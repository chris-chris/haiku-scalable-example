import gym

from threading import Thread, Lock
from actor import ActorCriticModel


class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    
  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    
  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []

class Worker(Thread):
  global_episode = 0
  global_moving_average_reward = 0
  best_score = 0
  save_lock = Lock()
  def __init__(self,
               state_size,
               action_size,
               global_model,
               opt,
               result_queue,
               idx,
               game_name='CartPole-v0',
               save_dir='/tmp'):
    super(Worker, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.result_queue = result_queue
    self.global_model = global_model
    self.opt = opt
    self.local_model = ActorCriticModel(self.state_size, self.action_size)
    self.worker_idx = idx
    self.game_name = game_name
    self.env = gym.make(self.game_name).unwrapped
    self.save_dir = save_dir
    self.ep_loss = 0.0