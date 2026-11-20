"""Cross-host features/mapping, memory reset, fixed scoring and opt-in profiles."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import bluesky as bs
import numpy as np
from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_gym.envs.single_agent_env import SingleAgentEnv
from bluesky_zoo.competition.competition import CompetitionZooEnv
from bluesky_zoo.strategic_navigation import MultiAgentStrategicEnv
from bluesky_zoo.transfer_navigation import (SingleAgentTransferEnv,MultiAgentTransferEnv,TransferConfig,
                                           deadline_features,MDP)
from scripts import evaluate_competition as evaluation
from scripts.train_transfer_navigation import train,ComponentInfo
from test_multi_agent_env import fixed_scenario

class TransferNavigationTests(unittest.TestCase):
    def test_shared_spaces_commands_features_and_masks(self):
        snapshot=[]
        for cls in (SingleAgentTransferEnv,MultiAgentTransferEnv):
            env=cls(n_obstacles=0,scenario=fixed_scenario())
            try:
                obs,_=env.reset(seed=17)
                if cls is SingleAgentTransferEnv:row=obs;space=env.observation_space;state=env.navigation_state
                else:row=obs['A'];space=env.observation_space('A');state=env.navigation_states['A']
                self.assertTrue(space.contains(row));env._get_action('A',np.array([.2,-.5]))
                snapshot.append((space,deepcopy(row),state.last_target_heading,state.last_target_speed))
                self.assertEqual(row['intruders'].shape,(10,7))
                self.assertEqual(sum(row['intruder_mask']),0 if cls is SingleAgentTransferEnv else 2)
            finally:env.close()
        self.assertEqual(snapshot[0][0],snapshot[1][0])
        self.assertEqual(snapshot[0][2:],snapshot[1][2:])
        for key in ('goal_direction','goal_distance','time_to_goal','time_slack','clearance_rays','navigation_memory'):
            np.testing.assert_array_equal(snapshot[0][1][key],snapshot[1][1][key])

    def test_sa_observes_all_ten_intruders_and_logs_rewards(self):
        env=SingleAgentTransferEnv()
        try:
            obs,_=env.reset(seed=101);self.assertEqual(sum(obs['intruder_mask']),10)
            metrics=deepcopy(env.metrics);again=env._get_obs(env.agent)
            self.assertEqual(metrics,env.metrics)
            np.testing.assert_array_equal(obs['navigation_memory'],again['navigation_memory'])
            _,reward,_,_,info=ComponentInfo(env).step(np.zeros(2))
            self.assertAlmostEqual(sum(info['navigation_reward_components'].values()),reward)
            self.assertTrue(env.observation_space.contains(env._get_obs(env.agent)))
        finally:env.close()

    def test_live_scoring_dynamics_and_termination_are_inherited(self):
        for cls,base in ((SingleAgentTransferEnv,CompetitionEnv),(MultiAgentTransferEnv,CompetitionZooEnv)):
            for method in ('step','_update_metrics','_get_info'):
                self.assertIs(getattr(cls,method),getattr(base,method))
            env=cls()
            try:
                self.assertEqual((env.intrusion_distance,env.distance_margin,env.episode_time_limit,env.sim_dt),(5.,5.,3000.,1))
                self.assertEqual(env.n_obstacles,5)
            finally:env.close()

    def test_deadline_features_and_invalid_configuration(self):
        eta,slack=deadline_features(12.7,2.7*1.852,120.,2700.,3000.)
        self.assertAlmostEqual(eta[0],.1);self.assertAlmostEqual(slack[0],0.)
        eta,slack=deadline_features(1.,5.,220.,2999.,3000.)
        self.assertEqual(eta[0],0);self.assertGreater(slack[0],0)
        for kwargs in (dict(min_target_speed_kt=300),dict(side_switch_after_s=0),dict(side_commitment=1),
                       dict(stagnant_rate_nm_s=float('nan'))):
            with self.assertRaises(ValueError):TransferConfig(**kwargs)

    def test_progress_is_idempotent_and_memory_resets_per_aircraft(self):
        env=MultiAgentTransferEnv(n_agents=2)
        try:
            env.reset(seed=101);a,b=(env.navigation_states[x] for x in env.agents)
            with patch.object(a,'_clock',return_value=10.),patch.object(a,'_goal_distance_nm',return_value=a.progress_anchor_distance):
                a._refresh_progress();self.assertEqual(a.stalled_seconds,10.)
                a._refresh_progress();self.assertEqual(a.stalled_seconds,10.)
            self.assertEqual(b.stalled_seconds,0.)
            env.reset(seed=102)
            self.assertEqual(a.stalled_seconds,0.);self.assertEqual(a.avoidance_side,0)
        finally:env.close()

    def test_side_commitment_switches_once_and_does_not_override_inside_escape(self):
        env=MultiAgentTransferEnv(n_agents=2)
        try:
            env.reset(seed=101);state=env.navigation_states[env.agents[0]]
            idx=bs.traf.id2idx(state.ac_id)
            obstacle=dict(index=0,inside=False,nearest_point=(bs.traf.lat[idx]+.01,bs.traf.lon[idx]))
            with patch.object(state,'_clock',return_value=0.):state._get_buffer_avoidance_heading(idx,obstacle)
            original=state.avoidance_side;state.stalled_seconds=150.
            with patch.object(state,'_clock',return_value=150.):
                state._get_buffer_avoidance_heading(idx,obstacle);self.assertEqual(state.avoidance_side,-original)
                state._get_buffer_avoidance_heading(idx,obstacle);self.assertEqual(state.avoidance_side,-original)
            self.assertTrue(state.avoidance_switched)
            inside=state._get_buffer_avoidance_heading(idx,dict(obstacle,inside=True))
            self.assertTrue(np.isfinite(inside));self.assertEqual(state.avoidance_side,-original)
        finally:env.close()

    def test_legacy_sa_imports_and_v5_contract_remain_available(self):
        legacy=SingleAgentEnv()
        try:obs,_=legacy.reset(seed=101);self.assertTrue(legacy.observation_space.contains(obs))
        finally:legacy.close()
        env=MultiAgentStrategicEnv(n_agents=2)
        try:
            obs,_=env.reset(seed=101);row=obs[env.agents[0]]
            self.assertEqual(row['intruders'].shape,(9,7));self.assertNotIn('time_slack',row)
        finally:env.close()

    def test_sa_to_ma_training_checkpoint_contract_and_saved_actor_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            first=train(kind='sa',output=root/'sa/model',timesteps=2,n_steps=2,batch_size=2,n_epochs=1,checkpoint_freq=0)
            second=train(kind='ma',output=root/'ma/model',init_model=root/'sa/model.zip',timesteps=4,
                         n_agents=2,n_steps=2,batch_size=2,n_epochs=1,checkpoint_freq=0)
            self.assertEqual(first['observation_shapes'],second['observation_shapes'])
            self.assertEqual(first['configs'],second['configs'])
            third=train(kind='sa',output=root/'returned/model',init_model=root/'ma/model.zip',timesteps=2,
                        n_steps=2,batch_size=2,n_epochs=1,checkpoint_freq=0)
            self.assertEqual(second['configs'],third['configs'])
            act=evaluation.load_policy('ma',str(root/'ma/model.zip'),mdp=MDP)
            self.assertTrue(act.disable_lookahead)
            with self.assertRaises(FileExistsError):train(kind='sa',output=root/'sa/model',dry_run=True)

    def test_old_checkpoint_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'old.zip';source.write_bytes(b'fixture')
            with patch('scripts.train_transfer_navigation.load_policy',return_value=SimpleNamespace(
                    model=SimpleNamespace(competition_mdp='navigation-v5'))):
                with self.assertRaisesRegex(ValueError,'Older observation/action contracts'):
                    train(kind='ma',output=root/'new/model',init_model=source,timesteps=4)
            self.assertFalse((root/'new').exists())

    def test_new_mdp_factory_restores_configs_and_rejects_search(self):
        for kind in ('sa','ma'):
            env=evaluation.make_env(kind,mdp=MDP)
            try:
                self.assertIsInstance(env,SingleAgentTransferEnv if kind=='sa' else MultiAgentTransferEnv)
            finally:env.close()
        with patch.object(evaluation,'load_policy',return_value=SimpleNamespace(disable_lookahead=True)):
            with self.assertRaisesRegex(ValueError,'actor-only'):
                evaluation.evaluate('ma',1,10,'unused','unused.csv',mdp=MDP,no_lookahead=False)
        for cls in (SingleAgentTransferEnv,MultiAgentTransferEnv):
            with self.assertRaises(ValueError):cls(action_frequency=17)
        with self.assertRaises(ValueError):SingleAgentTransferEnv(n_intruders=11)
