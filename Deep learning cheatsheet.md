Based on  (copied from) https://github.com/mrdbourke/tensorflow-deep-learning/
thanks for the course materials mrdbourke

for the format I used this cheatsheet as inspiration (scroll down): https://zerotomastery.io/cheatsheets/python-cheat-sheet/

# work in progress
it includes the cheat sheet for the notebooks below:
00_tensorflow_fundamentals sheet 
01_neural_network_regression_in_tensorflow

## Contents  
### Useful libraries, modules  
- [Tensorflow](#tensorflow)
- [Numpy](#numpy)  
### Tensors in tensorflow  
- [Constant tensor](#constant-tensor)  
- [Variable tensor](#variable-tensor) 
- [Random tensors and shuffling](#random-tensors-and-shuffling)  
- [Tensor attributes](#tensor-attributes)
- [Adding, removing dimensions of a tensor](#adding-removing-dimensions-of-a-tensor)  
### Manipulating tensors
- [Basic elementwise operations](#basic-elementwise-operations)
- [Tensor, tensor operations](#tensor-tensor-operations)
- [Reshape, transpose](#reshape-transpose)
- [Finding the min, max, mean, sum (aggregation)](#finding-the-min-max-mean-sum-aggregation)
- [Setting and changing datatype](#setting-and-changing-datatype)
- [Finding the positional maximum and minimum](#finding-the-positional-maximum-and-minimum)
### Data preprocessing
- [One hot encoding](#one-hot-encoding)
### Typical neural network architectures
- [Typical architecture of a regression neural network](#typical-architecture-of-a-regresison-neural-network)
### Utilities
- [Numpy Tensorflow tensor conversions](#numpy-tensorflow-tensor-conversions)
- [Using tensorflow decorator](#using-tensorflow-decorator)


### Useful libraries, modules
##### Tensorflow
Tensorflow is the end to end library for data preprocessing, modelling, model service
```
# Import TensorFlow  
import tensorflow as tf
print(tf.__version__) # find the version number (should be 2.x+)
[Back to top](#contents)
```
##### Numpy
Numpy is a library for numerical, tensor/array computations 
```
# Import numpy
import numpy as np
```
[Back to top](#contents)  
### Tensors in Tensorflow
##### Tensors in Tensorflow
Note that:  
scalar: a single number.  
vector: a number with direction (e.g. wind speed with direction).  
matrix: a 2-dimensional array of numbers.  
tensor: an n-dimensional arrary of numbers (where n can be any number, a 0-dimension tensor is a scalar, a 1-dimension tensor is a vector).  
[Back to top](#contents)  

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
# Create a variable tensor with numpy
I = tf.Variable(np.arange(0, 5)) # it outputs [0, 1, 2, 3, 4]
```
[Back to top](#contents)  
##### Variable tensor
tf.Variable mutable tensor (like tf.Constant, just changeable)
```
changeable_tensor = tf.Variable([10, 7])
changeable_tensor[0].assign(7) # assign 7 to the first element

another_matrix = tf.Variable([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16) # specify the datatype with 'dtype'

# Make a tensor of all ones
tf.ones(shape=(3, 2))

# Make a tensor of all zeros
tf.zeros(shape=(3, 2))
```
[Back to top](#contents)
##### Random tensors and shuffling
Note: always set the random seed for reproducibility
```
# Create random tensor 
random_1 = tf.random.Generator.from_seed(42) # set the  seed for reproducibility
random_1 = random_1.normal(shape=(3, 2)) # create tensor from a normal distribution 

# Create a tensor with 50 random values between 0 and 100
E = tf.constant(np.random.randint(low=0, high=100, size=50))

# Create a tensor with 50 values between 0 and 1
F = tf.constant(np.random.random(50))

# Change the order of elements of a tensor with local seed
G = tf.random.shuffle(not_shuffled, seed=42) # local seed that only effects the codeline you write it in

# Change the order of elements of a tensor with global seed
# Set global random seed
tf.random.set_seed(42) #the global random seed that works for the entire code block

# Set the operation random seed
tf.random.shuffle(not_shuffled, seed=42)
```
[Back to top](#contents)
##### Tensor attributes
Useful informations about tensors  

Note that:  
Shape: The length (number of elements) of each of the dimensions of a tensor.  
Rank: The number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix is rank 2, a tensor has rank n.  
Axis or Dimension: A particular dimension of a tensor.  
Size: The total number of items in the tensor.  

```
# Get various attributes of tensor
print("Datatype of every element:", rank_4_tensor.dtype)
print("Number of dimensions (rank):", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (2*3*4*5):", tf.size(rank_4_tensor).numpy()) # .numpy() converts to NumPy array
```
[Back to top](#contents)
### Manipulating tensors
##### Basic elementwise operations
Basic, element wise operations with Python
```
# Elementwise addition operator
tensor = tf.constant([[10, 7], [3, 4]])
tensor + 10

# Multiplication (known as element-wise multiplication)
tensor * 10

# Subtraction
tensor - 10
```

Basic, elementwise operations with Tensorflow
```
# multiplication with tensorflow 1st solution 
tf.multiply(tensor, 10) # it won't rewrite the original tensor

# multiplication with tensorflow 2nd solution 
tf.math.multiply(
    x, y, name=None
)

and others...
# Square it
tf.square(H)

# Find the squareroot (will error if it is integer/default), needs to be non-integer
tf.sqrt(H)

# Find the log (input also needs to be float)
tf.math.log(H)
```
[Back to top](#contents)
##### Tensor, tensor operations 
```
Matrix mutliplication with Python
tensor @ tensor

Matrix mutliplication with Tensorflow
tf.matmul(tensor, tensor)

The dot product
tf.tensordot(tf.transpose(X), Y, axes=1)
```
[Back to top](#contents)
##### Reshape, transpose
```
Reshape
tf.reshape(Y, shape=(2, 3))

Transpose
tf.transpose(X)

Example for more operations in one line
tf.matmul(a=X, b=Y, transpose_a=True, transpose_b=False)
```
[Back to top](#contents)
##### Finding the min, max, mean, sum (aggregation)
```
# Get the absolute values
tf.abs(D)

# Find the minimum
tf.reduce_min(E)

# Find the maximum
tf.reduce_max(E)

# Find the mean
tf.reduce_mean(E)

# Find the sum
tf.reduce_sum(E)
```
[Back to top](#contents)
##### Setting and changing tensor datatype
```
# Create a new tensor with default datatype (float32)
Default = tf.constant([1.7, 7.4])

# Create a new tensor with float16 datatype 
Non_Default = tf.constant([1.7, 7.4], dtype=tf.float16)

# Change from float32 to float16 (reduced precision)
Default_Changed = tf.cast(B, dtype=tf.float16)
```
[Back to top](#contents)
##### Finding the positional maximum and minimum
Max returns the index (or position) of the largest value in the tensor
Min is the same
```
# Find the maximum element position of F
tf.argmax(F)

# Find the minimum element position of F
tf.argmin(F)
```
[Back to top](#contents)
##### Adding, removing dimensions of a tensor
```
# Note that Python slicing also works on tensors
# Get the dimension from each index except for the final one
rank_4_tensor[:1, :1, :1, :]

# Add an extra dimension (to the end) 1st solution
rank_3_tensor = rank_2_tensor[..., tf.newaxis] # in Python "..." means "all dimensions prior to"

# Add an extra dimension (to the end) 2nd solution
tf.expand_dims(rank_2_tensor, axis=-1) # "-1" means last axis

# Squeezing a tensor (removing all single dimensions)
G_squeezed = tf.squeeze(G)
```
[Back to top](#contents)
### Data preprocessing
##### One hot encoding
Turn string categories (stuff, not_stuff) into numbers ([[1,0],[0,1]])
```
# Create a list of indices
some_list = [0, 1, 2, 3]

# One hot encode them
tf.one_hot(some_list, depth=4) # 4 is the number of different categories you have

# Specify custom values for on and off encoding
tf.one_hot(some_list, depth=4, on_value="We're live!", off_value="Offline")
```
[Back to top](#contents)
### Utilities
##### Numpy Tensorflow tensor conversions
```
# Create a tensor from a NumPy array
J = tf.constant(np.array([3., 7., 10.]))

# Convert tensor J to NumPy with np.array()
np.array(J)

# Convert tensor J to NumPy with .numpy()
J.numpy()
```
[Back to top](#contents)
##### Using Tensorflow decorator
In the @tf.function decorator case, it turns a Python function into a callable TensorFlow graph. Which is a fancy way of saying, if you've written your own Python function, and you decorate it with @tf.function, when you export your code (to potentially run on another device), TensorFlow will attempt to convert it into a fast(er) version of itself (by making it part of a computation graph).
```
# Create a  function and decorate it with tf.function
@tf.function
def tf_function(x, y):
  return x ** 2 + y

tf_function(x, y)
```
[Back to top](#contents)
##### Finding access to GPUs
```
# You can check if you've got access to a GPU using:
print(tf.config.list_physical_devices('GPU'))

You can also find information about your GPU using:
!nvidia-smi
```
[Back to top](#contents)

##### Typical architecture of a regression neural network
        "| **Hyperparameter** | **Typical value** |\n",
        "| --- | --- |\n",
        "| Input layer shape | Same shape as number of features (e.g. 3 for # bedrooms, # bathrooms, # car spaces in housing price prediction) |\n",
        "| Hidden layer(s) | Problem specific, minimum = 1, maximum = unlimited |\n",
        "| Neurons per hidden layer | Problem specific, generally 10 to 100 |\n",
        "| Output layer shape | Same shape as desired prediction shape (e.g. 1 for house price) |\n",
        "| Hidden activation | Usually [ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) (rectified linear unit) |\n",
        "| Output activation | None, ReLU, logistic/tanh |\n",
        "| Loss function | [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) (mean square error) or [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) (mean absolute error)/Huber (combination of MAE/MSE) if outliers |\n",
        "| Optimizer | [SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (stochastic gradient descent), [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) |\n",
        "\n",
        "***Table 1:*** *Typical architecture of a regression network.* ***Source:*** *Adapted from page 293 of [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow Book by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)*\n",



