Based (copied from) on https://github.com/mrdbourke/tensorflow-deep-learning/
thanks for the course materials mrdbourke

for the format I used this cheatsheet as inspiration (scroll down): https://zerotomastery.io/cheatsheets/python-cheat-sheet/

## Contents  
### Useful libraries, modules  
- Tensorflow  
- Numpy  
### Tensors in tensorflow  
- Constant tensor  
- Variable tensor  
- Random tensors and shuffling  
- Tensor attributes  
- Reshaping tensors  
### Manipulating tensors
- Tensors basic (algebraic) operations
- Tensor - tensor operations
- Reshape, transpose
- Finding the min, max, mean, sum (aggregation)
- Setting and changing tensor datatype

### Useful libraries, modules
##### Tensorflow
Tensorflow is the end to end library for data preprocessing, modelling, model service
```
# Import TensorFlow  
import tensorflow as tf
print(tf.__version__) # find the version number (should be 2.x+)
```
##### Numpy
Numpy is a library for numerical, tensor/array computations 
```
# Import numpy
import numpy as np
```

### Tensors in Tensorflow
##### Tensors in Tensorflow
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

another_matrix = tf.Variable([[10., 7.],
                              [3., 2.],
                              [8., 9.]], dtype=tf.float16) # specify the datatype with 'dtype'

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

# Create a tensor with 50 random values between 0 and 100
E = tf.constant(np.random.randint(low=0, high=100, size=50))

# Change the order of elements of a tensor with local seed
tf.random.shuffle(not_shuffled, seed=42) # local seed that only effects the codeline you write it in

# Change the order of elements of a tensor with global seed
# Set global random seed
tf.random.set_seed(42) #the global random seed that works for the entire code block

# Set the operation random seed
tf.random.shuffle(not_shuffled, seed=42)
```
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

##### Reshaping tensors
Note that slicing also works on tensors:
```
# Get the dimension from each index except for the final one
rank_4_tensor[:1, :1, :1, :]

# Add an extra dimension (to the end) 1st solution
rank_3_tensor = rank_2_tensor[..., tf.newaxis] # in Python "..." means "all dimensions prior to"

# Add an extra dimension (to the end) 2nd solution
tf.expand_dims(rank_2_tensor, axis=-1) # "-1" means last axis
```

### Manipulating tensors
##### Tensors basic (algebraic) operations
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
```
##### Tensor - tensor operations 
```
Matrix mutliplication with Python
tensor @ tensor

Matrix mutliplication with Tensorflow
tf.matmul(tensor, tensor)

The dot product
tf.tensordot(tf.transpose(X), Y, axes=1)
```
Reshape, transpose
```
Reshape
tf.reshape(Y, shape=(2, 3))

Transpose
tf.transpose(X)

Example for more operations in one line
tf.matmul(a=X, b=Y, transpose_a=True, transpose_b=False)
```
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
##### Setting and changing tensor datatype
```
# Create a new tensor with default datatype (float32)
Default = tf.constant([1.7, 7.4])

# Create a new tensor with float16 datatype 
Non_Default = tf.constant([1.7, 7.4], dtype=tf.float16)

# Change from float32 to float16 (reduced precision)
Default_Changed = tf.cast(B, dtype=tf.float16)

```
