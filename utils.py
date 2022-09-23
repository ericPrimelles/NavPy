import tensorflow as tf
from keras.layers import Flatten
def flatten(x : tf.Tensor):
    
    return Flatten()(x)