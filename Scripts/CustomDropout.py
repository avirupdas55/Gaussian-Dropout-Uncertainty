import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, Dropout
import tensorflow.keras.backend as K
from keras import backend
from keras.utils import tf_utils

class myDropout(Dropout):
  def __init__(self, rate, training=None, noise_shape=None, seed=None, **kwargs):
    super(myDropout, self).__init__(rate, noise_shape=None, seed=None, **kwargs)
    self.training=training

  def call(self, inputs, training=None):
    if 0.<self.rate<1:
      noise_shape= self._get_noise_shape(inputs)

      def dropped_inputs():
        return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)

      if not training:
        return K.in_train_phase(dropped_inputs, inputs, training=self.training)
      return K.in_train_phase(dropped_inputs, inputs, training=self.training)

    return inputs

class myGaussianDropout(Layer):

  def __init__(self, rate, training=None, seed=None, **kwargs):
    super(myGaussianDropout, self).__init__(**kwargs)
    self.supports_masking = True
    self.rate = rate
    self.seed = seed
    self.training=training
    self._random_generator = backend.RandomGenerator(seed)

  def call(self, inputs, training=None):
    if 0 < self.rate < 1:

      def noised():
        stddev = np.sqrt(self.rate / (1.0 - self.rate))
        return inputs * self._random_generator.random_normal(
            shape=tf.shape(inputs),
            mean=1.0,
            stddev=stddev,
            dtype=inputs.dtype)
        
        if not training:
          return K.in_train_phase(noised, inputs, training=self.training)
        return K.in_train_phase(noised, inputs, training=self.training)
    return inputs

  def get_config(self):
    config = {'rate': self.rate, 'seed': self.seed}
    base_config = super(myGaussianDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape