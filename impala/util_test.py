"""Util test."""
from absl.testing import absltest
from bsuite.experiments.catch import catch
import dm_env
import jax
import mock
import numpy as np
from jax.experimental import optix

from impala import actor as actor_lib
from impala import agent as agent_lib
from impala import haiku_nets
from impala import learner as learner_lib
from impala import util
import message_pb2

class CatchTest(absltest.TestCase):

  def setUp(self):
    super(CatchTest, self).setUp()
    self.env = catch.Catch()
    self.action_spec = self.env.action_spec()
    self.num_actions = self.action_spec.num_values
    self.obs_spec = self.env.observation_spec()
    self.agent = agent_lib.Agent(
        num_actions=self.num_actions,
        obs_spec=self.obs_spec,
        net_factory=haiku_nets.CatchNet,
    )

    self.key = jax.random.PRNGKey(42)
    self.key, subkey = jax.random.split(self.key)
    self.initial_params = self.agent.initial_params(subkey)

  def test_encode_decode_trajectory(self):
    # mock_learner = mock.MagicMock()
    traj_len = 10
    actor = actor_lib.Actor(
        agent=self.agent,
        env=self.env,
        unroll_length=traj_len,
    )
    self.key, subkey = jax.random.split(self.key)
    trajectory = actor.unroll(
        rng_key=subkey,
        frame_count=0,
        params=self.initial_params,
        unroll_length=traj_len)

    proto_trajectory = util.proto3_encoder(trajectory)
    self.assertIsInstance(proto_trajectory,
                          message_pb2.InsertTrajectoryRequest2)

    decoded_trajectory = util.proto3_decoder(proto_trajectory)
    np.testing.assert_almost_equal(trajectory.agent_state,
                                   decoded_trajectory.agent_state)
    np.testing.assert_almost_equal(trajectory.agent_out.action,
                                   decoded_trajectory.agent_out.action)

  def test_encode_weights(self):
    env = catch.Catch()
    action_spec = env.action_spec()
    num_actions = action_spec.num_values
    obs_spec = env.observation_spec()
    agent = agent_lib.Agent(
        num_actions=num_actions,
        obs_spec=obs_spec,
        net_factory=haiku_nets.CatchNet,
    )
    unroll_length = 20
    learner = learner_lib.Learner(
        agent=agent,
        rng_key=jax.random.PRNGKey(42),
        opt=optix.sgd(1e-2),
        batch_size=1,
        discount_factor=0.99,
        frames_per_iter=unroll_length,
    )
    actor = actor_lib.Actor(
        agent=agent,
        env=env,
        unroll_length=unroll_length,
    )
    frame_count, params = learner.params_for_actor()
    proto_weight = util.proto3_weight_encoder(frame_count, params)
    decoded_frame_count, decoded_params = \
      util.proto3_weight_decoder(proto_weight)
    self.assertEqual(frame_count, decoded_frame_count)
    np.testing.assert_almost_equal(decoded_params["catch_net/linear"]["w"],
                                   params["catch_net/linear"]["w"])
    act_out = actor.unroll_and_push(frame_count, params)
