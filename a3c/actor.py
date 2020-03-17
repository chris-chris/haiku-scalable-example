import haiku as hk
import jax
import jax.numpy as jnp

class ActorCriticModel:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.actor = hk.transform(self.actor_fn)
    self.critic = hk.transform(self.critic_fn)

  
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
  
    