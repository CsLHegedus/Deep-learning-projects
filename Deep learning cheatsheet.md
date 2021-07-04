Based on https://github.com/mrdbourke/tensorflow-deep-learning/
thanks for the course materials mrdbourke

for the format I used this cheatsheet as inspiration (scroll down): https://zerotomastery.io/cheatsheets/python-cheat-sheet/

## Contents  
- Useful libraries, modules  
- tensors in tensorflow  
-   

#### Useful libraries, modules
Tensorflow is the end to end library for data preprocessing, modelling, model service
```
# Import TensorFlow  
import tensorflow as tf
print(tf.__version__) # find the version number (should be 2.x+)
```
Numpy is a library for numerical, tensor/array computations 
```
# Import numpy
import numpy as np
```

#### Tensors in Tensorflow
Note that:  
scalar: a single number.  
vector: a number with direction (e.g. wind speed with direction).  
matrix: a 2-dimensional array of numbers.  
tensor: an n-dimensional arrary of numbers (where n can be any number, a 0-dimension tensor is a scalar, a 1-dimension tensor is a vector).  

##### Constant tensor
tf.Constant: immutable tensor  (something you don't change)
```
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
```
##### Variable tensor
tf.Variable mutable tensor (like tf.Constant, just changeable)
```
changeable_tensor = tf.Variable([10, 7])
changeable_tensor[0].assign(7) # assign 7 to the first element

# Make a tensor of all ones
tf.ones(shape=(3, 2))

# Make a tensor of all zeros
tf.zeros(shape=(3, 2))
```
##### Random tensors and shuffling
Note: always set the random seed for reproducibility
```
# Create random tensor 
random_1 = tf.random.Generator.from_seed(42) # set the  seed for reproducibility
random_1 = random_1.normal(shape=(3, 2)) # create tensor from a normal distribution 

# Change the order of elements of a tensor with local seed
tf.random.shuffle(not_shuffled, seed=42) # local seed that only effects the codeline you write it in

# Change the order of elements of a tensor with global seed
# Set global random seed
tf.random.set_seed(42) #the global random seed that works for the entire code block

# Set the operation random seed
tf.random.shuffle(not_shuffled, seed=42)
```


