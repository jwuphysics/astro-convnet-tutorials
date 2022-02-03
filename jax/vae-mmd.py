import jax
import jax.numpy as jnp

import quarks2cosmos
import tensorflow_datasets as tfds
import tensorflow as tf

import haiku as hk     
import optax           
import pickle

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

# initialize pseudo RNG
rng_seq = hk.PRNGSequence(42)

def augment_data(example):
    """ This applies random flipping to the input image.
    """
    x = example['image']
    x = x[...,tf.newaxis]
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = x[..., 0]
    return x 
  
class Encoder(hk.Module):
  """Simple Convolutional encoder model."""
  def __init__(self, latent_size=64):
    super().__init__()
    self._latent_size = latent_size

  def __call__(self, x):
    # Add channel dimension to x
    x = x[..., jnp.newaxis]
    
    x = hk.Conv2D(16, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')
    
    x = hk.Conv2D(32, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x,  window_shape=3, strides=2, padding='SAME')
    
    x = hk.Conv2D(64, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')
    
    x = hk.Conv2D(128, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')
    
    x = hk.Conv2D(128, kernel_shape=3)(x)
    x = jax.nn.leaky_relu(x)
    x = hk.avg_pool(x, window_shape=3, strides=2, padding='SAME')
    
    x = hk.Flatten()(x)
    
    x = hk.Linear(256)(x)
    x = jax.nn.leaky_relu(x)
    
    # Returns the variational distribution encoding the input image
    loc = hk.Linear(self._latent_size)(x)
    
    scale = jax.nn.softplus(hk.Linear(self._latent_size)(x)) + 1e-3
    return tfd.MultivariateNormalDiag(loc, scale) 

class Decoder(hk.Module):
  """Simple Convolutional decoder model."""
  def __call__(self, z, scale=1.0):
    
    # Reshape latent variable to an image
    x = hk.Linear(256)(z)
    x = jax.nn.leaky_relu(x)
    
    x = hk.Linear(3*3*128)(x)
    x = x.reshape([-1,3,3,128])
    
    x = hk.Conv2DTranspose(128, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)
    
    x = hk.Conv2DTranspose(64, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)
    
    x = hk.Conv2DTranspose(32, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)
    
    x = hk.Conv2DTranspose(16, kernel_shape=3, stride=2)(x)
    x = jax.nn.leaky_relu(x)
    
    x = hk.Conv2DTranspose(8, kernel_shape=3, stride=2)(x)
    
    x = hk.Conv2D(1, kernel_shape=5)(x)
    
    x = x[...,0]
    x = jnp.pad(x, [[0,0],[3,2],[3,2]])  # This step is to pad the image for the 101x101 expected size
    
    return tfd.Independent(tfd.Normal(loc=x, scale=scale), reinterpreted_batch_ndims=2)
  
def compute_kernel(x, y):
    x_size = jnp.shape(x)[0]
    y_size = jnp.shape(y)[0]
    dim = jnp.shape(x)[1]
    tiled_x = jnp.tile(jnp.reshape(x, jnp.stack([x_size, 1, dim])), jnp.stack([1, y_size, 1]))
    tiled_y = jnp.tile(jnp.reshape(y, jnp.stack([1, y_size, dim])), jnp.stack([x_size, 1, 1]))
    return jnp.exp(-jnp.mean(jnp.square(tiled_x - tiled_y), axis=2) / jnp.array(dim, dtype=jnp.float32))

# can't jit -> even `static_argnums` doesn't help here because the arrays are unhashable
def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return jnp.mean(x_kernel) + jnp.mean(y_kernel) - 2 * jnp.mean(xy_kernel)

def loss_mmd_fn(params, rng_key, batch, alpha=1., beta=1.):
    x = batch
    
    q = encoder.apply(params, x)
    z = q.sample(seed=rng_key)
    p = decoder.apply(params, z)
    kl = tfd.kl_divergence(q, tfd.MultivariateNormalDiag(jnp.zeros(64), scale_identity_multiplier=1))
    
    log_likelihood = p.log_prob(x)
    
    mmd = compute_mmd(
        tfd.MultivariateNormalDiag(jnp.zeros(64), scale_identity_multiplier=1).sample(200, seed=rng_key), 
        q.sample(seed=rng_key)
    )
    return -jnp.mean(log_likelihood - alpha*mmd - beta*kl)
 
# @jax.jit -- this will fail when called
def update(params, rng_key, opt_state, batch):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_mmd_fn)(params, rng_key, batch)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

if __name__ == '__main__':
  
  # prepare training data
  dset = tfds.load('Cosmos/23.5', split=tfds.Split.TRAIN)
  dset = dset.cache()
  dset = dset.repeat()
  dset = dset.shuffle(50000)
  dset = dset.batch(64)
  dset = dset.map(augment_data)
  dset = iter(tfds.as_numpy(dset))
  
  # initialize params
  encoder = hk.without_apply_rng(hk.transform(lambda x : Encoder()(x)))
  params_enc = encoder.init(next(rng_seq), jnp.zeros([1,101,101]))
  decoder = hk.without_apply_rng(hk.transform(lambda z : Decoder()(z)))
  params_dec = decoder.init(next(rng_seq), jnp.zeros([1,64]))
  params = hk.data_structures.merge(params_enc, params_dec)
  
  # get 200 samples from distribution p, compute MMD between p and q
  # q = encoder.apply(params, batch)
  # mmd = jnp.array(compute_mmd(
  #     tfd.MultivariateNormalDiag(jnp.zeros(64), scale_identity_multiplier=1).sample(200, seed=next(rng_seq)), 
  #     q.sample(seed=next(rng_seq))
  # ))
  # print(mmd)
  
  # initialize optimizer
  optimizer = optax.adam(3e-4)
  opt_state = optimizer.init(params)
  
  # training loop 
  for s in range(10000):
    loss, params, opt_state = update(params, next(rng_seq), opt_state, next(dset))
    step+=1
    
    if step %500 == 0:
        print(step, loss)
    
    if step%5000 ==0:
        with open('models/vae/model-%d.pckl'%step, 'wb') as file:
            pickle.dump([params, opt_state], file)