from jax import Array
import optax
import tensorflow_datasets as tfds
import numpy as np
import jax.numpy as jnp
import jax
import flax.nnx as nnx
BATCH_IN_SEQUENCES = 32
SEQUENCE_LENGTH = 128

VOCAB_DIM = 256
EMBED_DIM = 512
FF_DIM = 2048

LAYERS = 4

HEAD_DEPTH = 128
NUM_HEADS = 4

LEARNING_RATE = 1e-3

class OurModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.embedding = nnx.Embed(VOCAB_DIM, EMBED_DIM, rngs=rngs)

        ff1 = nnx.Linear(EMBED_DIM, FF_DIM, use_bias=False, rngs=rngs)
        ff2 = nnx.Linear(FF_DIM, EMBED_DIM, use_bias=False, rngs=rngs)
        layer = nnx.Sequential(ff1, nnx.relu, ff2, nnx.relu)
        layers = [layer] * LAYERS
        self.layers = nnx.Sequential(*layers)
      
    def __call__(self, input: Array):
        x = self.embedding(input)
        x = self.layers(x)
        return x @ self.embedding.embedding.value.T

# -------------------------------------------------------------------------------------------------
def convert_to_ascii(string_array, max_length):
  result = np.zeros((len(string_array), max_length), dtype=np.uint8)
  for i, string in enumerate(string_array):
    for j, char in enumerate(string):
      if j >= SEQUENCE_LENGTH:
         break
      result[i, j] = char
  return result

# -------------------------------------------------------------------------------------------------
def input_to_output(np_array):
   zero_array = np.zeros( (BATCH_IN_SEQUENCES,SEQUENCE_LENGTH), dtype = jnp.uint8)
   zero_array[:, 1:SEQUENCE_LENGTH] = np_array[:, 0:SEQUENCE_LENGTH-1]
   return zero_array

# -------------------------------------------------------------------------------------------------
def calculate_loss(model, inputs, outputs):
   proposed_outputs = model(inputs)
   one_hot = jax.nn.one_hot(outputs, VOCAB_DIM)
   loss = optax.softmax_cross_entropy(proposed_outputs, one_hot)
   return jnp.mean(loss)

# -------------------------------------------------------------------------------------------------
def main():
    ds = tfds.load('lm1b', split='train', shuffle_files=False)
    ds = ds.batch(BATCH_IN_SEQUENCES)

    model = OurModel(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE))

    for i, example in enumerate(ds):
        outputs = convert_to_ascii(example['text'].numpy(), SEQUENCE_LENGTH)
        inputs = input_to_output(outputs)
       
        grad_fn = nnx.value_and_grad(calculate_loss, has_aux=False)
        loss, grads = grad_fn(model, inputs, outputs)
        optimizer.update(grads)
        print(f"{i} -> {loss}")


if __name__ == "__main__":
    main()