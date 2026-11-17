"""Bounded policy/value-guided tree search over independent horizontal forecasts.

This module deliberately does not import BlueSky or any environment. Search
callbacks operate on copies of world state supplied by the caller; they must
never step, reset, or otherwise mutate the live simulator. The small predictor
is an approximation used to compare decisions, not a replacement for A320
dynamics in the actual competition environment.

PPO continues to collect and train on its own actor's actions. Search is an
inference improvement using that learned actor's priors and critic's values.
"""

from dataclasses import dataclass, field, replace
import math
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class LookaheadConfig:
    simulations: int = 24
    max_depth: int = 6
    exploration: float = 1.5
    discount: float = 0.998
    candidate_count: int = 7
    rollout_seconds: float = 15.0
    turn_rate_deg_s: float = 3.0
    speed_change_kt_s: float = 1.0
    coordination_passes: int = 1
    safety_filter: bool = False
    safety_buffer_nm: float = 3.0
    safety_horizon_seconds: float = 60.0

    def __post_init__(self):
        if not isinstance(self.safety_filter, bool):
            raise ValueError("safety_filter must be boolean")
        if not math.isfinite(self.safety_buffer_nm) or self.safety_buffer_nm < 0:
            raise ValueError("safety_buffer_nm must be finite and nonnegative")
        if not math.isfinite(self.safety_horizon_seconds) or not 0 < self.safety_horizon_seconds <= 120:
            raise ValueError("safety_horizon_seconds must be in (0, 120]")
        if (isinstance(self.coordination_passes, bool) or not isinstance(self.coordination_passes, int)
                or not 1 <= self.coordination_passes <= 2):
            raise ValueError("coordination_passes must be 1 or 2")
        for name, upper in (("simulations", 4096), ("max_depth", 32), ("candidate_count", 31)):
            value = getattr(self, name)
            minimum = 3 if name == "candidate_count" else 1
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{minimum}, {upper}]")
        if self.simulations < self.coordination_passes:
            raise ValueError("simulations must provide at least one simulation per coordination pass")
        for name in ("exploration", "rollout_seconds", "turn_rate_deg_s", "speed_change_kt_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if self.rollout_seconds > 120:
            raise ValueError("rollout_seconds must be <= 120")


def resolve_lookahead_config(saved=None, **overrides):
    config = saved if isinstance(saved, LookaheadConfig) else LookaheadConfig(**(saved or {}))
    return replace(config, **{key: value for key, value in overrides.items() if value is not None})


def resolve_inference_lookahead_config(saved, *, preset=None, simulations=None,
                                     max_depth=None, coordination_passes=None):
    """Resolve optional inference budgets without altering checkpoint defaults.

    Fast keeps every root manoeuvre and all predictor safety samples, but gives
    the learned tree fewer visits and less depth. Explicit budgets win over a
    preset. This changes planning behavior and needs separate evaluation.
    """
    if preset not in (None, "saved", "fast"):
        raise ValueError("lookahead preset must be saved or fast")
    overrides = dict(simulations=12, max_depth=3, coordination_passes=1) if preset == "fast" else {}
    overrides.update({key: value for key, value in dict(
        simulations=simulations, max_depth=max_depth, coordination_passes=coordination_passes,
    ).items() if value is not None})
    return resolve_lookahead_config(saved, **overrides)


@dataclass(frozen=True)
class PolicyExpansion:
    """Learned action priors and state value, in the target aircraft's reward units."""

    actions: np.ndarray
    priors: np.ndarray
    value: float
    terminal: bool = False

    def __post_init__(self):
        actions = np.array(self.actions, dtype=np.float64, copy=True)
        priors = np.array(self.priors, dtype=np.float64, copy=True)
        if actions.size == 0:
            actions = actions.reshape(0, 2)
        if actions.ndim != 2 or actions.shape[1:] != (2,):
            raise ValueError("Policy actions must have shape (N, 2)")
        if priors.shape != (len(actions),):
            raise ValueError("Policy priors must have one entry per candidate action")
        if not math.isfinite(float(self.value)):
            raise ValueError("Learned leaf value must be finite")
        if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError("Policy actions must be finite and in [-1, 1]")
        if not np.isfinite(priors).all() or np.any(priors < 0):
            raise ValueError("Policy priors must be finite and nonnegative")
        if not self.terminal and (len(actions) == 0 or not np.any(priors > 0)):
            raise ValueError("Nonterminal states require candidates and positive learned prior mass")
        if len(actions) and np.any(priors > 0):
            priors /= priors.max()
            priors /= priors.sum()
        actions.setflags(write=False)
        priors.setflags(write=False)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "priors", priors)
        object.__setattr__(self, "value", float(self.value))


@dataclass(frozen=True)
class Transition:
    state: Any
    reward: float
    terminal: bool = False
    # Allows a macro step to use gamma ** (duration / live decision interval).
    discount: float | None = None

    def __post_init__(self):
        if not math.isfinite(float(self.reward)):
            raise ValueError("Predicted reward must be finite")
        if self.discount is not None and (not math.isfinite(self.discount) or not 0 <= self.discount <= 1):
            raise ValueError("Transition discount must be in [0, 1]")


@dataclass(frozen=True)
class SearchResult:
    action: np.ndarray
    visit_counts: np.ndarray
    priors: np.ndarray
    action_values: np.ndarray
    value: float
    simulations: int
    expanded_nodes: int
    deepest_depth: int


@dataclass
class _Edge:
    prior: float
    visits: int = 0
    total_value: float = 0.0
    child: Any = None
    reward: float = 0.0
    discount: float = 1.0
    terminal: bool = False
    forecast_state: Any = None


@dataclass(frozen=True)
class ForecastStepDiagnostics:
    action: tuple
    state_before: Any
    state_after: Any
    reward: float
    discount: float
    terminal: bool


@dataclass(frozen=True)
class CandidateDiagnostics:
    index: int
    action: tuple
    prior: float
    visits: int
    mean_return: float | None
    first_reward: float | None
    first_discount: float | None
    first_leaf_value: float | None
    explored_depth: int
    principal_path: tuple


@dataclass(frozen=True)
class SearchDiagnostics:
    root_value: float
    selected_index: int
    candidates: tuple


def _search_diagnostics(root, selected_index, depth_limit):
    """Read the completed tree; never add visits, expansions or rollouts."""
    def explored_depth(edge):
        if not edge.visits:
            return 0
        children = edge.child.edges if edge.child is not None else []
        return 1 + max((explored_depth(child) for child in children), default=0)

    candidates = []
    for index, edge in enumerate(root.edges):
        node, chosen = root, index
        path = []
        for _ in range(depth_limit):
            current = node.edges[chosen]
            if not current.visits:
                break
            path.append(ForecastStepDiagnostics(
                tuple(float(v) for v in node.expansion.actions[chosen]), node.state,
                current.forecast_state, current.reward, current.discount, current.terminal,
            ))
            if current.terminal or current.child is None:
                break
            node = current.child
            visited = [i for i, child in enumerate(node.edges) if child.visits]
            if not visited:
                break
            chosen = max(visited, key=lambda i: (node.edges[i].visits,
                node.edges[i].total_value / node.edges[i].visits, node.edges[i].prior))
        candidates.append(CandidateDiagnostics(
            index, tuple(float(v) for v in root.expansion.actions[index]), edge.prior, edge.visits,
            edge.total_value / edge.visits if edge.visits else None,
            edge.reward if edge.visits else None, edge.discount if edge.visits else None,
            0.0 if edge.terminal else edge.child.expansion.value if edge.child is not None else None,
            explored_depth(edge), tuple(path),
        ))
    return SearchDiagnostics(root.expansion.value, selected_index, tuple(candidates))


@dataclass
class _Node:
    state: Any
    expansion: PolicyExpansion
    edges: list = field(default_factory=list)

    def __post_init__(self):
        self.edges = [_Edge(float(prior)) for prior in self.expansion.priors]


def policy_value_search(state, *, evaluate: Callable[[Any], PolicyExpansion],
                        transition: Callable[[Any, np.ndarray], Transition],
                        config=None, observer=None, action_selector=None):
    """Run deterministic PUCT with learned priors and bootstrapped leaf values.

    ``evaluate`` must consult the trained actor and critic, not a handcrafted
    substitute. ``transition`` must construct an independent forecast state and
    return the target aircraft's estimated reward. Terminal transitions already
    include their terminal reward and therefore have no continuation value.

    Search expands at most one new node per simulation, plus the root; both
    simulation count and depth are explicitly bounded. This is single-objective
    search: other aircraft's actions belong in the transition model, rather
    than being treated as adversarial turns. An optional ``action_selector``
    chooses among the completed root candidates without changing tree visits
    or values; the observer records that final choice.
    """
    config = resolve_lookahead_config(config)
    initial = evaluate(state)
    if not isinstance(initial, PolicyExpansion):
        raise TypeError("evaluate must return PolicyExpansion")
    if initial.terminal:
        raise ValueError("Cannot choose an action for a terminal root state")
    root = _Node(state, initial)
    expanded_nodes = 1
    deepest_depth = 0

    for _ in range(config.simulations):
        node = root
        path = []
        leaf_value = node.expansion.value
        for depth in range(1, config.max_depth + 1):
            parent_visits = sum(edge.visits for edge in node.edges)
            scores = [
                (edge.total_value / edge.visits if edge.visits else 0.0)
                + config.exploration * edge.prior * math.sqrt(parent_visits + 1) / (edge.visits + 1)
                for edge in node.edges
            ]
            # Give every root manoeuvre one trial when the simulation budget
            # permits. Otherwise a very confident stuck actor can prevent its
            # low-prior escape direction from ever being evaluated.
            unvisited = [index for index, edge in enumerate(node.edges) if edge.visits == 0]
            if node is root and unvisited:
                index = max(unvisited, key=lambda item: node.edges[item].prior)
            else:
                index = int(np.argmax(scores))
            edge = node.edges[index]
            path.append(edge)
            deepest_depth = max(deepest_depth, depth)
            if edge.child is None and not edge.terminal:
                predicted = transition(node.state, node.expansion.actions[index].copy())
                if not isinstance(predicted, Transition):
                    raise TypeError("transition must return Transition")
                edge.reward = float(predicted.reward)
                if observer is not None:
                    edge.forecast_state = predicted.state
                edge.discount = config.discount if predicted.discount is None else float(predicted.discount)
                edge.terminal = bool(predicted.terminal)
                if edge.terminal:
                    leaf_value = 0.0
                else:
                    expansion = evaluate(predicted.state)
                    if not isinstance(expansion, PolicyExpansion):
                        raise TypeError("evaluate must return PolicyExpansion")
                    edge.child = _Node(predicted.state, expansion)
                    expanded_nodes += 1
                    edge.terminal = bool(expansion.terminal)
                    leaf_value = 0.0 if edge.terminal else expansion.value
                break
            if edge.terminal:
                leaf_value = 0.0
                break
            node = edge.child
            leaf_value = node.expansion.value

        for edge in reversed(path):
            leaf_value = edge.reward + edge.discount * leaf_value
            edge.visits += 1
            edge.total_value += leaf_value

    visits = np.array([edge.visits for edge in root.edges], dtype=np.int64)
    values = np.array([edge.total_value / edge.visits if edge.visits else initial.value
                       for edge in root.edges], dtype=np.float64)
    # Prefer visits; use estimated value and learned prior only to break ties.
    best = max(range(len(visits)), key=lambda index: (visits[index], values[index], initial.priors[index]))
    if action_selector is not None:
        best = action_selector(initial.actions, visits, values, initial.priors)
        if isinstance(best, bool) or not isinstance(best, (int, np.integer)) or not 0 <= best < len(visits):
            raise ValueError("action_selector must return a valid candidate index")
    result = SearchResult(
        action=initial.actions[best].copy(), visit_counts=visits,
        priors=initial.priors.copy(), action_values=values,
        value=float(sum(edge.total_value for edge in root.edges) / config.simulations),
        simulations=config.simulations, expanded_nodes=expanded_nodes, deepest_depth=deepest_depth,
    )
    if observer is not None:
        observer(_search_diagnostics(root, best, config.max_depth))
    return result


def gaussian_action_candidates(mean, std, *, count=7, seed=0):
    """Discretize a learned Gaussian actor and retain its density as the prior.

    Include the actor mean, relative +/-90-degree turns and a U-turn so low
    actor variance cannot make a detour unavailable. Heading is periodic with
    period two because the v5 mapping spans +/-180 degrees. The actor still
    supplies every prior; these fixed proposals define the search's action
    discretization only.
    Returned arrays are new objects. Their length can be smaller than ``count``
    after removing duplicate clipped candidates.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    if mean.shape != (2,) or std.shape != (2,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Actor mean and standard deviation must be finite two-vectors")
    if np.any(std <= 0):
        raise ValueError("Actor standard deviation must be positive")
    if isinstance(count, bool) or not isinstance(count, int) or not 3 <= count <= 31:
        raise ValueError("Candidate count must be an integer in [3, 31]")
    centre = np.clip(mean, -1.0, 1.0)
    heading = lambda offset: (centre[0] + offset + 1.0) % 2.0 - 1.0
    candidates = [centre, [heading(-0.5), centre[1]], [heading(0.5), centre[1]],
                  [centre[0], -1.0], [centre[0], 1.0], [heading(1.0), centre[1]]]
    rng = np.random.default_rng(seed)
    candidates.extend(rng.normal(mean, std, size=(max(count - 6, 0), 2)))
    actions = []
    for candidate in candidates[:count]:
        action = np.clip(candidate, -1.0, 1.0)
        if not any(np.array_equal(action, previous) for previous in actions):
            actions.append(action)
    actions = np.array(actions, dtype=np.float64)
    # Clip standardized residuals before squaring so an extremely confident
    # but finite actor cannot overflow; probabilities retain a tiny floor.
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.clip((actions - mean) / np.maximum(std, np.finfo(float).tiny), -1e6, 1e6)
    log_priors = -0.5 * np.sum(normalized ** 2, axis=1)
    weights = np.exp(log_priors - np.max(log_priors))
    weights = np.maximum(weights, 1e-8)
    return actions, weights / weights.sum()


@dataclass(frozen=True)
class AircraftState:
    """Horizontal forecast state: local north/east NM, heading degrees, speed kt."""

    north_nm: float
    east_nm: float
    heading_deg: float
    speed_kt: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (
                self.north_nm, self.east_nm, self.heading_deg, self.speed_kt)) or self.speed_kt < 0:
            raise ValueError("Aircraft forecast state must be finite with nonnegative speed")


def predict_aircraft(state, target_heading_deg, target_speed_kt, seconds, config=None):
    """Return independent forecast samples at intervals of at most one second.

    Turns take the shortest heading change with a limited rate; speed changes
    have a limited acceleration. Midpoint heading/speed integrate each step.
    This intentionally omits altitude and makes no claim of exact A320 dynamics.
    The initial state is excluded; the last sample is at exactly ``seconds``.
    """
    config = resolve_lookahead_config(config)
    if not all(math.isfinite(value) for value in (target_heading_deg, target_speed_kt, seconds)):
        raise ValueError("Forecast targets and duration must be finite")
    if target_speed_kt < 0 or seconds <= 0 or seconds > 120:
        raise ValueError("Use nonnegative target speed and duration in (0, 120] seconds")
    samples = []
    elapsed = 0.0
    current = state
    while elapsed < seconds:
        dt = min(1.0, seconds - elapsed)
        error = (target_heading_deg - current.heading_deg + 180.0) % 360.0 - 180.0
        heading_change = float(np.clip(error, -config.turn_rate_deg_s * dt, config.turn_rate_deg_s * dt))
        speed_change = float(np.clip(target_speed_kt - current.speed_kt,
                                     -config.speed_change_kt_s * dt, config.speed_change_kt_s * dt))
        heading_mid = math.radians(current.heading_deg + heading_change / 2)
        distance_nm = (current.speed_kt + speed_change / 2) * dt / 3600.0
        current = AircraftState(
            current.north_nm + distance_nm * math.cos(heading_mid),
            current.east_nm + distance_nm * math.sin(heading_mid),
            (current.heading_deg + heading_change) % 360.0,
            current.speed_kt + speed_change,
        )
        samples.append(current)
        elapsed += dt
    return tuple(samples)
