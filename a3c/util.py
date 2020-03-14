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
"""Util."""
import collections
import json

from absl import logging
import dm_env
import numpy as np
import tree
from dm_env import TimeStep

from haiku._src.data_structures import to_immutable_dict
from impala import agent as agent_lib
import message_pb2

# Can represent either a single transition, a trajectory, or a batch of
# trajectories.
Transition = collections.namedtuple('Transition',
                                    ['timestep', 'agent_out', 'agent_state'])


def _preprocess_none(t) -> np.ndarray:
  if t is None:
    return np.array(0., dtype=np.float32)
  else:
    return np.asarray(t)


def preprocess_step(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  if timestep.discount is None:
    timestep = timestep._replace(discount=1.)
  return tree.map_structure(_preprocess_none, timestep)


class NullLogger:
  """Logger that does nothing."""

  def write(self, _):
    pass

  def close(self):
    pass


class AbslLogger:
  """Writes to logging.info."""

  def write(self, d):
    logging.info(d)

  def close(self):
    pass

class NumpyEncoder(json.JSONEncoder):
  def default(self, o):
    if isinstance(o, np.ndarray):
      return o.tolist()
    return json.JSONEncoder.default(self, o)

def ndarray_decoder(dct):
  if isinstance(dct, dict):
    for key in dct.keys():
      dct[key] = ndarray_decoder(dct[key])
  elif isinstance(dct, list):
    return np.array(dct)
  return dct


def tensor_int(tensor):
  return message_pb2.TensorInt32(data=tensor.flatten(),
                                 shape=tensor.shape)


def tensor_float(tensor):
  return message_pb2.TensorFloat(data=tensor.flatten(),
                                 shape=tensor.shape)


def tensor_int_decoder(tensor):
  data = np.array(tensor.data, dtype=np.int)
  data = data.reshape(tensor.shape)
  return data


def tensor_float_decoder(tensor):
  data = np.array(tensor.data, dtype=np.float)
  data = data.reshape(tensor.shape)
  return data


def proto3_encoder(trajectory):
  return message_pb2.Trajectory(
      agent_out=message_pb2.AgentOut(
          action=tensor_int(trajectory.agent_out.action),
          policy_logits=tensor_float(trajectory.agent_out.policy_logits),
          values=tensor_float(trajectory.agent_out.values),
      ),
      agent_state=tensor_float(trajectory.agent_state),
      timestep=message_pb2.Timestep(
          discount=tensor_float(trajectory.timestep.discount),
          observation=tensor_float(trajectory.timestep.observation),
          reward=tensor_float(trajectory.timestep.reward),
          step_type=tensor_int(trajectory.timestep.step_type),
      )
  )

def proto3_decoder(trajectory):
  decoded = Transition(
      agent_out=agent_lib.AgentOutput(
          action=tensor_int_decoder(trajectory.agent_out.action),
          policy_logits=tensor_float_decoder(trajectory.agent_out.policy_logits),
          values=tensor_int_decoder(trajectory.agent_out.values),
      ),
      agent_state=tensor_float_decoder(trajectory.agent_state),
      timestep=TimeStep(
          discount=tensor_float_decoder(trajectory.timestep.discount),
          observation=tensor_float_decoder(trajectory.timestep.observation),
          reward=tensor_float_decoder(trajectory.timestep.reward),
          step_type=tensor_int_decoder(trajectory.timestep.step_type),
      )
  )
  return decoded


def encode_layer_weight(params):
  layers = []
  for key in params:
    layer = message_pb2.LayerWeight(
        name=key,
        b=tensor_float(params[key]['b']),
        w=tensor_float(params[key]['w']),
    )
    layers.append(layer)
  return layers


def decode_layer_weight(layers):
  data = {}
  for layer in layers:
    data[layer.name] = to_immutable_dict({
      "b": tensor_float_decoder(layer.b),
      "w": tensor_float_decoder(layer.w)
    })
  return to_immutable_dict(data)


def proto3_weight_encoder(frame_count, params):
  layer_weights = encode_layer_weight(params)
  return message_pb2.ModelParams(
      frame_count=frame_count,
      params=layer_weights
  )


def proto3_weight_decoder(model_params):
  frame_count = model_params.frame_count
  params = decode_layer_weight(model_params.params)
  return frame_count, params
