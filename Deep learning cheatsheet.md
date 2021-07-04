Based on https://github.com/mrdbourke/tensorflow-deep-learning/
thanks for the course materials mrdbourke

for the format I used this cheatsheet as inspiration: https://zerotomastery.io/cheatsheets/python-cheat-sheet/

Contents
important libraries
tensors in tensorflow


# Fast starter libraries

## Tensorflow is the end to end library for data preprocessing, modelling, model service
# Import TensorFlow
import tensorflow as tf
print(tf.__version__) # find the version number (should be 2.x+)


## Tensors in Tensorflow
# tf.Constant: immutable (something you don't change) tensor

# Create a scalar (rank 0 tensor)
scalar = tf.constant(7)
# Create a vector (more than 0 dimensions)
vector = tf.constant([10, 10])
# Create a matrix (more than 1 dimension)
matrix = tf.constant([[10, 7],
                      [7, 10]])
# Create another matrix and define the datatype
another_matrix = tf.constant([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16) # specify the datatype with 'dtype'

## tf.Variable mutable tensor (like tf.Constant, just changeable)
changeable_tensor = tf.Variable([10, 7])
changeable_tensor[0].assign(7)



