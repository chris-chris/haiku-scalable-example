import os
from queue import Queue
import argparse
import multiprocessing
import threading

import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class RandomAgent:
  """Random Agent that will play the game environment.
  """
  def __init__(self, env_name, max_eps):
    self.env = gym.make(env_name)
    self.max_episodes = max_eps
    self.global_moving_average_reward = 0
    self.res_queue = Queue()

  def run(self):
    reward_avg = 0
    for episode in range(self.max_episodes):
      done = False
      self.env.reset()
      reward_sum = 0.0
      steps = 0
      while not done:
        # Sample randomly from the action space and step
        _, reward, done, _ = self.env.step(self.env.action_space.sample())
        steps += 1
        reward_sum += reward
      # Record statistics
      self.global_moving_average_reward = record(episode,
                                                 reward_sum,
                                                 0,
                                                 self.global_moving_average_reward,
                                                 self.res_queue,
                                                 0, steps)
      reward_avg += reward_sum

    final_avg = reward_avg / float(self.max_episodes)
    print(f"Average score across {self.max_episodes} episodes: {final_avg}")
    return final_avg

class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.dense1 = layers.Dense(100, activation="relu")
    self.policy_logits = layers.Dense(action_size)
    self.dense2 = layers.Dense(100, activation="relu")
    self.values = layers.Dense(1)

  def __call__(self, inputs):
    x = self.dense1(inputs)
    logits = self.policy_logits(x)
    v1 = self.dense2(inputs)
    values = self.values(v1)
    return logits, values

class MasterAgent():
  def __init__(self):
    self.game_name = "CartPole-v0"
    save_dir = args.save_dir
    self.save_dir = save_dir
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    env = gym.make(self.game_name)
    self.state_size = env.observation_space.shape[0]
    self.action_size = env.action_space.n
    self.opt = tf.train.AdamOptimizer(args.lr, use_locking=True)
    print(self.state_size, self.action_size)

    self.global_model = ActorCriticModel(self.state_size, self.action_size)
    self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

  def train(self):
    if args.algorithm == "random":
      random_agent = RandomAgent(self.game_name, args.max_eps)
      random_agent.run()
      return

    res_queue = Queue()

    workers = [Worker(self.state_size,
                      self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i, game_name=self.game_name,
                      save_dir=self.save_dir) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
      print(f"Starting worker {i}")
      worker.start()

    moving_average_rewards = []
    while True:
      reward = res_queue.get()
      if reward is not None:
        moving_average_rewards.append(reward)
      else:
        break

    [w.join() for w in workers]


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


class Worker(threading.Thread):
  global_episode = 0
  global_moving_average_reward = 0
  best_score = 0
  save_lock = threading.Lock()

  def __init__(self, state_size, action_size, global_model,
      opt, result_queue, idx, game_name="CartPole-v0", save_dir="/tmp"):
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

  def run(self):
    total_step = 1
    mem = Memory()
    while Worker.global_episode < args.max_eps:
      current_state = self.env.reset()
      mem.clear()
      ep_reward = 0
      ep_steps = 0
      self.ep_loss = 0

      time_count = 0
      done = False
      while not done:
        logits, _ = self.local_model(
            tf.convert_to_tensor(current_state[None, :],
                                 dtype=tf.float32)
        )
        probs = tf.nn.softmax(logits)

        action = np.random.choice(self.action_size, p=probs.numpy()[0])
        new_state, reward, done, _ = self.env.step(action)
        if done:
          reward -= 1
        ep_reward += reward
        mem.store(current_state, action, reward)

        if time_count == args.update_freq or done:
          with tf.GradientTape() as tape:
            total_loss = self.compute_loss(done, new_state, mem, args.gamma)

          self.ep_loss += total_loss
          grads = tape.gradient(total_loss, self.local_model.trainable_weights)
          self.opt.apply_gradient(zip(grads, self.global_model.trainable_weights))
          self.local_model.set_weights(self.global_model.get_weights())

          mem.clear()
          time_count = 0

          if done:
            Worker.global_moving_average_reward = \
              record(Worker.global_episode, ep_reward, self.worker_idx,
                     Worker.global_moving_average_reward, self.result_queue,
                     self.ep_loss, ep_steps)