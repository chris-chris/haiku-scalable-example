# Lint as: python3
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Single-process IMPALA wiring."""

from concurrent import futures
import time
import json

import grpc
import dm_env
from absl import app
import numpy as np
import jax
from jax.experimental import optix
from haiku._src.data_structures import to_mutable_dict
from bsuite.experiments.catch import catch

from impala import agent as agent_lib
from impala import haiku_nets
from impala import learner as learner_lib
from impala import util
import message_pb2
import message_pb2_grpc


ACTION_REPEAT = 1
BATCH_SIZE = 2
DISCOUNT_FACTOR = 0.99
MAX_ENV_FRAMES = 20000
NUM_ACTORS = 2
UNROLL_LENGTH = 20

FRAMES_PER_ITER = ACTION_REPEAT * BATCH_SIZE * UNROLL_LENGTH

class Information(message_pb2_grpc.InformationServicer):
  """gRPC protocol interface"""
  def __init__(self, learner):
    self.learner = learner

  def InsertTrajectory(self, request, context):
    trajectory = json.loads(request.trajectory,
                            object_hook=util.ndarray_decoder)
    traj = util.Transition(
        timestep=dm_env.TimeStep(
            step_type=np.array(trajectory[0][0]),
            reward=np.array(trajectory[0][1]),
            observation=np.array(trajectory[0][3]),
            discount=np.array(trajectory[0][2])
        ), agent_out=agent_lib.AgentOutput(
            policy_logits=np.array(trajectory[1][0]),
            action=np.array(trajectory[1][2]),
            values=np.array(trajectory[1][1])
        ),
        agent_state=np.array(trajectory[2])
    )
    self.learner.enqueue_traj(traj)
    return message_pb2.InsertTrajectoryReply(message='ID: %s' % id)

  def GetParams(self, request, context):
    frame_count, params = self.learner.params_for_actor()
    mutable_params = to_mutable_dict(params)
    params_json = json.dumps(mutable_params, cls=util.NumpyEncoder)

    return message_pb2.GetParamsReply(frame_count=frame_count,
                                      params=params_json)

def setup_learner():
  """Setup learner for distributed setting"""
  # A thunk that builds a new environment.
  # Substitute your environment here!
  build_env = catch.Catch

  # Construct the agent. We need a sample environment for its spec.
  env_for_spec = build_env()
  num_actions = env_for_spec.action_spec().num_values
  agent = agent_lib.Agent(num_actions, env_for_spec.observation_spec(),
                          haiku_nets.CatchNet)

  # Construct the optimizer.
  opt = optix.rmsprop(1e-1, decay=0.99, eps=0.1)

  # Construct the learner.
  learner = learner_lib.Learner(
      agent,
      jax.random.PRNGKey(428),
      opt,
      BATCH_SIZE,
      DISCOUNT_FACTOR,
      FRAMES_PER_ITER,
      max_abs_reward=1.,
      logger=util.AbslLogger(),  # Provide your own logger here.
  )
  return learner

def setup_server(learner):
  """Setup gRPC server for the communication between learners and actors."""

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
  message_pb2_grpc.add_InformationServicer_to_server(Information(learner),
                                                     server)
  server.add_insecure_port('[::]:50051')
  return server


def main(_):

  max_updates = MAX_ENV_FRAMES / FRAMES_PER_ITER
  learner = setup_learner()
  server = setup_server(learner)
  server.start()
  learner.run(int(max_updates))

  try:
    while not learner.is_done():
      time.sleep(1)
  except KeyboardInterrupt:
    server.stop(0)

if __name__ == '__main__':
  app.run(main)
