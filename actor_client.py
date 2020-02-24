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

import threading
import os
import json

from absl import app
from haiku._src.data_structures import to_immutable_dict
import grpc
from bsuite.experiments.catch import catch

from impala import actor as actor_lib
from impala import agent as agent_lib
from impala import haiku_nets
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

def run_actor(actor: actor_lib.Actor):
  """Runs an actor to produce num_trajectories trajectories."""
  host = os.getenv("GRPC_HOST", "localhost:50051")
  # print(host)
  channel = grpc.insecure_channel(host)
  stub = message_pb2_grpc.InformationStub(channel)

  while True:

    param_result = stub.GetParams(message_pb2.GetParamsRequest())
    frame_count = param_result.frame_count
    params = param_result.params
    params_obj = json.loads(params, object_hook=util.ndarray_decoder)
    params_frozen = to_immutable_dict(params_obj)

    trajectories = actor.unroll_and_push(frame_count, params_frozen)
    t_obj = json.dumps(trajectories, cls=util.NumpyEncoder)

    stub.InsertTrajectory(message_pb2.InsertTrajectoryRequest(
        trajectory=t_obj
    ))

def setup_actors(num_actors):
  """Setup actor threads for the execution."""
  # A thunk that builds a new environment.
  # Substitute your environment here!
  build_env = catch.Catch

  # Construct the agent. We need a sample environment for its spec.
  env_for_spec = build_env()
  num_actions = env_for_spec.action_spec().num_values
  agent = agent_lib.Agent(num_actions, env_for_spec.observation_spec(),
                          haiku_nets.CatchNet)

  # Construct the actors on different threads.
  # stop_signal in a list so the reference is shared.
  actor_threads = []
  stop_signal = [False]
  for i in range(num_actors):
    actor = actor_lib.Actor(
        agent,
        build_env(),
        UNROLL_LENGTH,
        rng_seed=i,
        logger=util.AbslLogger(),  # Provide your own logger here.
    )
    args = (actor, stop_signal)
    actor_threads.append(threading.Thread(target=run_actor, args=args))
  return actor_threads

def main(_): # pragma: no cover
  actor_threads = setup_actors(NUM_ACTORS)

  # Start the actors and learner.
  for t in actor_threads:
    t.start()

  # Stop.
  for t in actor_threads:
    t.join()


if __name__ == '__main__': # pragma: no cover
  app.run(main)
