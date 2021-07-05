Based on  (copied from) https://github.com/mrdbourke/tensorflow-deep-learning/  
thanks for the course materials mrdbourke  

for the format I used this cheatsheet as inspiration (scroll down): https://zerotomastery.io/cheatsheets/python-cheat-sheet/  

# work in progress  
it includes the cheat sheet for the notebooks below:  
00_tensorflow_fundamentals sheet   
01_neural_network_regression_in_tensorflow  

## Contents  
- [Useful libraries, modules](#useful-libraries-modules)
- [Tensors in tensorflow](#tensors-in-tensorflow)
- [Manipulating tensors](#manipulating-tensors)
- [Visualize any and everything](#visualize-any-and-everything)
- [Steps in preprocessing and modelling](#steps-in-preprocessing-and-modelling)
- [Data preprocessing](#data-preprocessing)
- [Typical neural network architectures](#typical-neural-network-architectures)
- [Utilities](#utilities)

#### Useful libraries, modules  
- [Tensorflow](#tensorflow)
- [Numpy](#numpy)  
- [Matplotlib](#matplotlib)
- [Scikitlearn](#scikitlearn)
#### Tensors in tensorflow  
- [Constant tensor](#constant-tensor)  
- [Variable tensor](#variable-tensor) 
- [Random tensors and shuffling](#random-tensors-and-shuffling)  
- [Tensor attributes](#tensor-attributes)
- [Adding, removing dimensions of a tensor](#adding-removing-dimensions-of-a-tensor)  
#### Manipulating tensors
- [Basic elementwise operations](#basic-elementwise-operations)
- [Tensor, tensor operations](#tensor-tensor-operations)
- [Reshape, transpose](#reshape-transpose)
- [Finding the min, max, mean, sum (aggregation)](#finding-the-min-max-mean-sum-aggregation)
- [Setting and changing datatype](#setting-and-changing-datatype)
- [Finding the positional maximum and minimum](#finding-the-positional-maximum-and-minimum)
#### Visualize any and everything
- [What to visualize](#what-to-visualize)
- [Visualizing the data, regression model](#visualizing-the-data-regression-model)
- [Visualizing the model](#visualize-the-model)
- [Visualizing the loss curve](#visualize-loss-curve)
#### Steps in preprocessing and modelling
- [Typical workflow for modelling](#typical-workflow)
#### Data preprocessing
- [Train test split with python](#train-test-split-with-python)
- [One hot encoding with pandas](#one-hot-encoding-with-pandas)
- [One hot encoding with tensorflow](#one-hot-encoding-with-tensorflow)
- [One hot encoding and data normalization with scikitlearn](#one-hot-encoding-and-data-normalization-with-scikitlearn)
#### Typical neural network architectures
- [Typical architecture of a regression neural network](#typical-architecture-of-a-regression-neural-network)
- [Regression model example](#regression-model-example)
#### Utilities
- [Numpy Tensorflow tensor conversions](#numpy-tensorflow-tensor-conversions)
- [Using tensorflow decorator](#using-tensorflow-decorator)
- [How to compare models](#how-to-compare-models)
- [How to save, load and check a saved a model](#how-to-save-load-and-check-a-saved-a-model)
- [How to download a model from google colab](#how-to-download-a-model-from-google-colab)


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
##### Matplotlib
Matplotlib is a library to visualize, evaulate your data
```
# Import matplotlib
import matplotlib.pyplot as plt
```
[Back to top](#contents)  
##### Scikitlearn
Scikitlearn is a library for data preprocessing and machine learning models
```
#For data preprocessing
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
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
### Visualizing any and everything
##### What to visualize
- The data - what data are you working with? What does it look like?
- The model itself - what does the architecture look like? What are the different shapes?
- The training of a model - how does a model perform while it learns?
- The predictions of a model - how do the predictions of a model line up against the ground truth (the original labels)?
##### Visualizing the data, regression model
Visualize a regression model's data
```
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=y_preds):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training data") 
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  # Plot the predictions in red (predictions were made on the test data)
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  # Show the legend
  plt.legend();
```
[Back to top](#contents)
##### Visualizing the model
```
# Visualize the model in commandline style
model.summary()
```
Total params - total number of parameters in the model.
Trainable parameters - these are the parameters (patterns) the model can update as it trains.
Non-trainable parameters - these parameters aren't updated during training (this is typical when you bring in the already learned patterns from other models during transfer learning).

Visualize the data with 2D plot
```
from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True)
```
##### Visualizing the loss curve
```
# Plot history (also known as a loss curve)
pd.DataFrame(history_1.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs");
```
[Back to top](#contents)
### Data preprocessing
##### Train test split with python
Get your data into train and test sets. Train for training, test for testing, easy isn't it?
```
# Split 50 data rows into train and test sets
X_train = X[:40] # first 40 examples (80% of data)
y_train = y[:40]

X_test = X[40:] # last 10 examples (20% of data)
y_test = y[40:]

len(X_train), len(X_test) # check the sets
```
##### Train test split with scikitlearn
Get your data into train and test sets. Train for training, test for testing, easy isn't it?
```
# Create training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42) # set random state for reproducible splits
```
##### One hot encoding with pandas
Turn string categories (stuff, not_stuff) into numbers ([[1,0],[0,1]])
```
insurance_one_hot = pd.get_dummies(insurance)
```
##### One hot encoding with tensorflow
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
##### One hot encoding and data normalization with scikitlearn
Turn string categories (stuff, not_stuff) into numbers ([[1,0],[0,1]])
```
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Create column transformer (this will help us normalize/preprocess our data)
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi", "children"]), # get all values between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

# Create X & y
X = insurance.drop("charges", axis=1)
y = insurance["charges"]

# Build our train and test sets (use random state to ensure same split as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit column transformer on the training data only (doing so on test data would result in data leakage)
ct.fit(X_train)

# Transform training and test data with normalization (MinMaxScalar) and one hot encoding (OneHotEncoder)
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

```
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

##### How to compare models
```
# Create shorter function names (why not)
def mae(y_test, y_pred):
  """
  Calculuates mean absolute error between y_test and y_preds.
  """
  return tf.metrics.mean_absolute_error(y_test,
                                        y_pred)
  
def mse(y_test, y_pred):
  """
  Calculates mean squared error between y_test and y_preds.
  """
  return tf.metrics.mean_squared_error(y_test,
                                       y_pred)

# Calculate model_1 metrics
mae_1 = mae(y_test, y_preds_3.squeeze()).numpy()
mse_1 = mse(y_test, y_preds_3.squeeze()).numpy()
mae_1, mse_3

# Repeat with the rest of the models
# Create a nested list for the results
model_results = [["model_1", mae_1, mse_1],
                 ["model_2", mae_2, mse_2],
                 ["model_3", mae_3, mae_3]]

# Turn it into pandas dataframe and print it
import pandas as pd
all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
all_results
```
TensorBoard - a component of the TensorFlow library to help track modelling experiments (we'll see this later).
Weights & Biases - a tool for tracking all kinds of machine learning experiments (the good news for Weights & Biases is it plugs into TensorBoard).
[Back to top](#contents)
##### How to save load and check a saved a model
```
There are two ways to save a model in TensorFlow:

The SavedModel format (default).
The HDF5 format.
The main difference between the two is the SavedModel is automatically able to save custom objects (such as special layers) without additional modifications when loading the model back in.

It's a good practice to load back the saved model and compare it to the original, just to be sure nothing went wrong
```
```
# Save a model using the SavedModel format
model_2.save('best_model_SavedModel_format')

# Check it out - outputs a protobuf binary file (.pb) as well as other files
!ls best_model_SavedModel_format

# Load a model from the SavedModel format
loaded_saved_model = tf.keras.models.load_model("best_model_SavedModel_format")
loaded_saved_model.summary()
```
```
# Save a model using the HDF5 format
model_2.save("best_model_HDF5_format.h5") # note the addition of '.h5' on the end

# Check it out
!ls best_model_HDF5_format.h5

# Load a model from the HDF5 format
loaded_h5_model = tf.keras.models.load_model("best_model_HDF5_format.h5")
loaded_h5_model.summary()
```
```
# Compare model_2 with the SavedModel version (should return True)
model_2_preds = model_2.predict(X_test)
saved_model_preds = loaded_saved_model.predict(X_test)
mae(y_test, saved_model_preds.squeeze()).numpy() == mae(y_test, model_2_preds.squeeze()).numpy()
```
### How to download a model from google colab
```
# Download the model (or any file) from Google Colab
from google.colab import files
files.download("best_model_HDF5_format.h5")
```
### Typical workflow
Check the data, figure out the problem type (regression, binary or multi classification etc.)

Preprocess the data
```
Visualize your data when possible.

Load the data with pandas
Is it too big? use chunks

Check for missing data
A feature vector (column) has less than 10% data coverage (lines with text)? Drop it.
A feature vector has missing data? use data imputation (fill missing values) use pandas or scikitlearn.

A feature has 
-numerical values only? 
Leave it.
-categorical or Ordinal values? 
One_Hot encode it. In pandas you can do it in one step with imputation, use column transformer.  
Scikit_learn and tensorflow also has functions for imputation with variable abilities

Normalize the data if its values isn't between 0 and 1 for faster convergence (learning).
Use scikitlearn or a normalization layer in the model
```
Modeling
```
Create a model (layers, activation functions)
Compile the model (loss, optimizer, metrics)
Fit the model (save history, set number of epochs, callbacks)
Evaluate the model (metrics, predictions)

Compare the different models
Save the model

To improve use
- more neurons
- more layers
- different layers
- different optimizer
- more epochs
```
Optional
```
If you train the model for long time (1000+ epochs) you might want to
- determine the optimal starting learning rate
- use early callback to stop training when the model doesn't learn much anymore
- use checkpoints the save the progress so if something happens you don't have to train from scratch
```
[Back to top](#contents)
### Typical architecture of neural networks
The input shape is the shape of your data that goes into the model.  
The output shape is the shape of your data you want to come out of your model.  
##### Typical architecture of a regression neural network
| **Hyperparameter** | **Typical value** |
| --- | --- |
| Input layer shape | Same shape as number of features (e.g. 3 for # bedrooms, # bathrooms, # car spaces in housing price prediction) |
| Hidden layer(s) | Problem specific, minimum = 1, maximum = unlimited |
| Neurons per hidden layer | Problem specific, generally 10 to 100 |
| Output layer shape | Same shape as desired prediction shape (e.g. 1 for house price) |
| Hidden activation | Usually [ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) (rectified linear unit) |
| Output activation | None, ReLU, logistic/tanh |
| Loss function | [MSE](https://en.wikipedia.org/wiki/Mean_squared_error) (mean square error) or [MAE](https://en.wikipedia.org/wiki/Mean_absolute_error) (mean absolute error)/Huber (combination of MAE/MSE) if outliers |
| Optimizer | [SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (stochastic gradient descent), [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) |

original source: Adapted from page 293 of [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow Book by Aurélien Géron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
        
[Back to top](#contents)  
##### Regression model example
```
# let's use the insurance dataset (you )
# Set random seed
tf.random.set_seed(42)

# Build the model (3 layers, 100, 10, 1 units)
insurance_model_3 = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation="Relu"), # hidden layer with Relu activation
  tf.keras.layers.Dense(10, activation="Relu"), # hidden layer with Relu activation
  tf.keras.layers.Dense(1, activation="tanh") # output layer with tanh activation
])

# Compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(lr=0.001), # using default learning rate
                          metrics=['mae'])

# Fit the model for 200 epochs (same as insurance_model_2)
# verbose=0 means that it won't write out the training log
insurance_model_3.fit(X_train_normal, y_train, epochs=200, verbose=0)

# Evaluate the model
insurance_model_3.evaluate(X_test_normal, y_test)

# Do some predictions
insurance_model_3.preds(Y_test_normal[0]) # let's use the first row of the test data
```
[Typical architecture of a regression neural network](#typical-architecture-of-a-regression-neural-network)  
[Back to top](#contents)  




