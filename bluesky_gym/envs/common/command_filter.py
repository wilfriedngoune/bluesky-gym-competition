"""Optional command hysteresis; does not change aircraft flight dynamics."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CommandDeadbandConfig:
    heading_deadband_deg: float = 0.0
    speed_deadband_kt: float = 0.0

    def __post_init__(self):
        for name in ("heading_deadband_deg", "speed_deadband_kt"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.heading_deadband_deg > 180:
            raise ValueError("heading_deadband_deg must be <= 180")


def command_config(value=None):
    if isinstance(value, CommandDeadbandConfig):
        return value
    return CommandDeadbandConfig(**(value or {}))


def resolve_command_config(saved=None, *, heading_deadband_deg=None, speed_deadband_kt=None):
    config = command_config(saved)
    return CommandDeadbandConfig(
        config.heading_deadband_deg if heading_deadband_deg is None else heading_deadband_deg,
        config.speed_deadband_kt if speed_deadband_kt is None else speed_deadband_kt,
    )
