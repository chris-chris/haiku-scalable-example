import haiku as hk
import jax
import jax.numpy as jnp

class ActorCriticModel:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.actor = hk.transform(self.actor_fn)
    self.critic = hk.transform(self.critic_fn)
    self.actor_param = self.actor.init(jax.PRNGKey(42), pass) # TODO data input
    self.critic_param = self.critic.init(jax.PRNGKey(42), pass)

  def pi(self, states):
    return self.actor.apply(self.actor_param, states)
  
  def v(self, states):
    return self.critic.apply(self.critic_param, states)
  
  def actor_fn(self, batch):
    x = batch.astype(jnp.float32)
    actor = hk.Sequential([
        hk.Flatten(),
        hk.Linear(100), jax.nn.relu,
        hk.Linear(self.action_size),
    ])
    return actor(x)

  def critic_fn(self, batch):
    x = batch.astype(jnp.float32)
    critic = hk.Sequential([
        hk.Flatten(),
        hk.Linear(100), jax.nn.relu,
        hk.Linear(self.action_size),
    ])
    return critic(x)
  
    
  def get_actor_param(self):
    return self.actor_param

  def get_critic_param(self):
    return self.critic_param