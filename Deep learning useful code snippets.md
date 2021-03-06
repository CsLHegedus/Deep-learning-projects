Based on  (copied from) https://github.com/mrdbourke/tensorflow-deep-learning/  
thanks for the course materials mrdbourke  

as inspiration (scroll down): https://zerotomastery.io/cheatsheets/python-cheat-sheet/  

an even bigger cheatsheet for pros https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#
the cheatsheet for anything http://www.cheat-sheets.org/  
cnn explainer https://poloclub.github.io/cnn-explainer/

it includes the useful code snippets for the notebooks below:  
00. tensorflow_fundamentals sheet   
01. neural_network_regression_in_tensorflow  
02. Neural Network Classification with TensorFlow  
03. Convolutional Neural Networks and Computer Vision with TensorFlow  
04. Transfer Learning with TensorFlow Part 1: Feature Extraction   
05. transfer_learning_in_tensorflow_part_2_fine_tuning


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
- [Visualizing the data, classification model](#visualizing-the-data-classification-model)
- [Visualizing the data, image multiclass classification](#visualizing-the-data-image-multiclass-classification)
- [Visualizing the model](#visualizing-the-model)
- [Visualizing the loss curve](#visualizing-the-loss-curve)
- [Visualizing transfer learnin feature extraction vs fine tuning](#feature_extraction_vs_fine_tuning)
#### Steps in preprocessing and modelling
- [Typical workflow for modelling](#typical-workflow)
#### Data preprocessing
- [File structure of data](#file-structure-of-data) 
- [Train test split with python](#train-test-split-with-python)
- [One hot encoding with pandas](#one-hot-encoding-with-pandas)
- [One hot encoding with tensorflow](#one-hot-encoding-with-tensorflow)
- [One hot encoding and data normalization with scikitlearn](#one-hot-encoding-and-data-normalization-with-scikitlearn)
#### Typical neural network architectures and examples
- [Typical architecture of a regression neural network](#typical-architecture-of-a-regression-neural-network)
- [Regression model example](#regression-model-example)
- [Typical architecture of a classification neural network](#typical-architecture-of-a-classification-neural-network)
- [Typical architecture of a convolutional neural network](#typical-architecture-of-a-convolutional-neural-network)
- [Convolutional model example with less typing than in regression](Convolutional-model-example-with-less-typing-than-in-regression)
- [Types of transfer learning](#types-of-transfer-learning)
- [Transfer learning feature extraction end to end example](#transfer_learning_feature_extraction)
- [Transfer learning feature fine tuning end to end example](#transfer_learning_fine_tuning)
#### Callbacks
- [Tensorboard](#tensorboard)
- [Find ideal learning rate with learning scheduler](#find-ideal-learning-rate)
- [Checkpoint](#checkpoint)

#### Utilities
- [Numpy Tensorflow tensor conversions](#numpy-tensorflow-tensor-conversions)
- [Using tensorflow decorator](#using-tensorflow-decorator)
- [Evaluation metrics](#evaluation_metrics)
- [How to compare models](#how-to-compare-models)
- [How to save, load and check a saved a model](#how-to-save-load-and-check-a-saved-a-model)
- [How to download a model from google colab](#how-to-download-a-model-from-google-colab)
- [Download and extract zip](#download-and-extract-zip)
- [Datasets, toy datasets](#toy_datasets)


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
```

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
##### Visualizing the data, classification model
Binary classification
```
# Visualize data with a plot
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu);
```
```
def plot_decision_boundary(model, X, y):
  """
  Plots the decision boundary created by a model predicting on X.
  This function has been adapted from two phenomenal resources:
   1. CS231n - https://cs231n.github.io/neural-networks-case-study/
   2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
  """
  # Define the axis boundaries of the plot and create a meshgrid
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                       np.linspace(y_min, y_max, 100))
  
  # Create X values (we're going to predict on all of these)
  x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html
  
  # Make predictions using the trained model
  y_pred = model.predict(x_in)

  # Check for multi-class
  if len(y_pred[0]) > 1:
    print("doing multiclass classification...")
    # We have to reshape our predictions to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classifcation...")
    y_pred = np.round(y_pred).reshape(xx.shape)
  
  # Plot decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
```
```
# Example call
# Check out the predictions our model is making 
plot_decision_boundary(model_3, X, y)
```
``` 
# Check train and test data
# Plot the decision boundaries for the training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_8, X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_8, X=X_test, y=y_test)
plt.show()
```
custom confusion matrix
```
# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")
```
```
# Get the class names
class_names = test_data.class_names
class_names
```
```
# Plot a confusion matrix with all 25250 predictions, ground truth labels and 101 classes
make_confusion_matrix(y_true=y_labels,
                      y_pred=pred_classes,
                      classes=class_names,
                      figsize=(100, 100),
                      text_size=20,
                      norm=False,
                      savefig=True)```
##### Visualizing the data, image multiclass classification
```
```
# Plot an example image and its label
plt.imshow(train_data[17], cmap=plt.cm.binary) # change the colours to black & white
plt.title(class_names[train_labels[17]]);
```
```
# Plot multiple random images of fashion MNIST
import random
plt.figure(figsize=(7, 7))
for i in range(4):
  ax = plt.subplot(2, 2, i + 1)
  rand_index = random.choice(range(len(train_data)))
  plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
  plt.title(class_names[train_labels[rand_index]])
  plt.axis(False)
```
```
import random

# Create a function for plotting a random image along with its prediction
def plot_random_image(model, images, true_labels, classes):
  """Picks a random image, plots it and labels it with a predicted and truth label.

  Args:
    model: a trained model (trained on data similar to what's in images).
    images: a set of random images (in tensor form).
    true_labels: array of ground truth labels for images.
    classes: array of class names for images.
  
  Returns:
    A plot of a random image from `images` with a predicted class label from `model`
    as well as the truth class label from `true_labels`.
  """ 
  # Setup random integer
  i = random.randint(0, len(images))
  
  # Create predictions and targets
  target_image = images[i]
  pred_probs = model.predict(target_image.reshape(1, 28, 28)) # have to reshape to get into right size for model
  pred_label = classes[pred_probs.argmax()]
  true_label = classes[true_labels[i]]

  # Plot the target image
  plt.imshow(target_image, cmap=plt.cm.binary)

  # Change the color of the titles depending on if the prediction is right or wrong
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  # Add xlabel information (prediction/true label)
  plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                   100*tf.reduce_max(pred_probs),
                                                   true_label),
             color=color) # set the color to green or red
```
```
# Example call
# Check out a random image as well as its prediction
plot_random_image(model=model_14, 
                  images=test_data, 
                  true_labels=test_labels, 
                  classes=class_names)
```
```
# View a random image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off");

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img
```
```
# View a random image from the training dataset
img = view_random_image(target_dir="pizza_steak/train/",
                        target_class="steak")
```
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
```
# Plot the validation and training data separately
def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();
```
[Back to top](#contents)
### Data preprocessing
##### File structure of data
```
# Check invidual folders
!ls pizza_steak
!ls pizza_steak/train/

# Walk through pizza_steak directory and list number of files
import os

for dirpath, dirnames, filenames in os.walk("pizza_steak"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))

num_steak_images_train

# Get the class names (programmatically, this is much more helpful with a longer list of classes)
import pathlib
import numpy as np
data_dir = pathlib.Path("pizza_steak/train/") # turn our training path into a Python path
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # created a list of class_names from the subdirectories
print(class_names)
```
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
##### Download and extract zip
```
import zipfile

# Download zip file of pizza_steak images
!wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip 

# Unzip the downloaded file
zip_ref = zipfile.ZipFile("pizza_steak.zip", "r")
zip_ref.extractall()
zip_ref.close()
```
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
[Back to top](#contents)
##### Evaluation metrics
tf.keras.losses.your_loss

Regression:
mae mean of absolute errors
mse mean of squared errors, sensitive to outliners
huber combination of the above two

Binary classification
Accuracy Out of 100 predictions, how many does your model get correct? E.g. 95% accuracy means it gets 95/100 predictions correct.	
Precision Proportion of true positives over total number of samples. Higher precision leads to less false positives (model predicts 1 when it should've been 0).	 
Recall Proportion of true positives over total number of true positives and false negatives (model predicts 0 when it should've been 1). Higher recall leads to less false negatives.	
F1-score Combines precision and recall into one metric. 1 is best, 0 is worst.	
Confusion matrix	Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line).	
Classification report	Collection of some of the main classification metrics such as precision, recall and f1-score.	
##### Find ideal learning rate
```
# Create a learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20)) # traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch

# Fit the model (passing the lr_scheduler callback)
history = model_9.fit(X_train, 
                      y_train, 
                      epochs=100,
                      callbacks=[lr_scheduler])

# Plot the learning rate versus the loss
lrs = 1e-4 * (10 ** (np.arange(100)/20))
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history.history["loss"]) # we want the x-axis (learning rate) to be log scale
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning rate vs. loss");
```
To figure out the ideal value of the learning rate (at least the ideal value to begin training our model), the rule of thumb is to take the learning rate value where the loss is still decreasing but not quite flattened out (usually about 10x smaller than the bottom of the curve).

Example of other typical learning rate values
10**0, 10**-1, 10**-2, 10**-3, 1e-4

[Back to top](#contents)  
### How to download a model from google colab
```
# Download the model (or any file) from Google Colab
from google.colab import files
files.download("best_model_HDF5_format.h5")
```
[Back to top](#contents)

##### Toy datasets
For practice or as warmup exercises:
https://scikit-learn.org/stable/datasets/toy_dataset.html
https://scikit-learn.org/stable/datasets.html
[Back to top](#contents)
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
(set an experiment to find the ideal learning rate)
Evaluate the model (metrics, predictions)

Compare the different models
Save the model

To improve use
- more neurons
- more layers
- different layers
- different optimizer
- more epochs

- more data
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

original source: Adapted from page 293 of [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow Book by Aur??lien G??ron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
        
[Back to top](#contents)  
##### Typical architecture of a classification neural network
| **Hyperparameter** | **Binary classification** | **Multiclass classification** |
| --- | --- | --- |
| Input layer shape | Same as number of features (e.g. 5 for age, sex, height, weight, smoking status in heart disease prediction) | Same as binary classification |
| Hidden layer(s) | Problem specific, minimum = 1, maximum = unlimited | Same as binary classification |
| Neurons per hidden layer | Problem specific, generally 10 to 100 | Same as binary classification |
| Output layer shape | 1 (one class or the other) | 1 per class (e.g. 3 for food, person or dog photo) |
| Hidden activation | Usually [ReLU](https://www.kaggle.com/dansbecker/rectified-linear-units-relu-in-deep-learning) (rectified linear unit) | Same as binary classification |
| Output activation | Sigmoid	| Softmax |
| Loss function | Cross entropy (tf.keras.losses.BinaryCrossentropy in TensorFlow)	(mean absolute error)/Huber (combination of MAE/MSE) if outliers | Cross entropy (tf.keras.losses.CategoricalCrossentropy in TensorFlow)
|
| Optimizer | [SGD](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD) (stochastic gradient descent), [Adam](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam) | Same as binary classification |

original source: Adapted from page 295 of [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow Book by Aur??lien G??ron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
             
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

# Create a learning rate scheduler callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20)) # traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch

# Fit the model for 200 epochs (same as insurance_model_2)
# verbose=0 means that it won't write out the training log
insurance_model_3.fit(X_train_normal, y_train, epochs=200, verbose=0,
                      callbacks=[lr_scheduler])

# Evaluate the model
insurance_model_3.evaluate(X_test_normal, y_test)

# Do some predictions
insurance_model_3.preds(Y_test_normal[0]) # let's use the first row of the test data
```
[Typical architecture of a regression neural network](#typical-architecture-of-a-regression-neural-network)  
[Back to top](#contents)  

##### Typical architecture of a convolutional neural network
| **Hyperparameter/Layer type** | **What does it do?** | **Typical values** |
| --- | --- | --- |
| Input image(s) | Target images you'd like to discover patterns in	 | Whatever you can take a photo (or video) of |
| Input layer	 | Takes in target images and preprocesses them for further layers | input_shape = [batch_size, image_height, image_width, color_channels] |
| Convolution layer	 | Extracts/learns the most important features from target images	 | Multiple, can create with tf.keras.layers.ConvXD (X can be multiple values) |
| Hidden activation	| Adds non-linearity to learned features (non-straight lines)	| Usually ReLU (tf.keras.activations.relu) |
|Pooling layer	|Reduces the dimensionality of learned image features	|Average (tf.keras.layers.AvgPool2D) or Max (tf.keras.layers.MaxPool2D)|
|Fully connected layer	|Further refines learned features from convolution layers	|tf.keras.layers.Dense|
|Output layer	|Takes learned features and outputs them in shape of target labels	|output_shape = [number_of_classes] (e.g. 3 for pizza, steak or sushi)|
|Output activation	|Adds non-linearities to output layer	|tf.keras.activations.sigmoid (binary classification) or tf.keras.activations.softmax|

detailed explanation of layers and their respective parameters:  
https://poloclub.github.io/cnn-explainer/
[Back to top](#contents)  

##### Convolutional model example with less typing than in regression
```
# Make the creating of our model a little easier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential

# Create the model (this can be our baseline, a 3 layer Convolutional Neural Network)
model_4 = Sequential([
  Conv2D(filters=10, 
         kernel_size=3, 
         strides=1,
         padding='valid',
         activation='relu', 
         input_shape=(224, 224, 3)), # input layer (specify input shape)
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  Flatten(),
  Dense(1, activation='sigmoid') # output layer (specify output shape)
])
```
And it follows the typical CNN structure of:

Input -> Conv + ReLU layers (non-linearities) -> Pooling layer -> Fully connected (dense layer) as Output
Let's discuss some of the components of the Conv2D layer:

- The "2D" means our inputs are two dimensional (height and width), even though they have 3 colour channels, the convolutions are run on each channel invididually.
- filters - these are the number of "feature extractors" that will be moving over our images.
- kernel_size - the size of our filters, for example, a kernel_size of (3, 3) (or just 3) will mean each filter will have the size 3x3, meaning it will look at a space of 3x3 pixels each time. The smaller the kernel, the more fine-grained features it will extract.
- stride - the number of pixels a filter will move across as it covers the image. A stride of 1 means the filter moves across each pixel 1 by 1. A stride of 2 means it moves 2 pixels at a time.
- padding - this can be either 'same' or 'valid', 'same' adds zeros the to outside of the image so the resulting output of the - --- convolutional layer is the same as the input, where as 'valid' (default) cuts off excess pixels where the filter doesn't fit (e.g. 224 pixels wide divided by a kernel size of 3 (224/3 = 74.6) means a single pixel will get cut off the end.

end to end example:
https://colab.research.google.com/drive/1tA-6oY2EUPg96EF3-mIpPB6bs9jckcsc#scrollTo=lRiun3m5PwZL
[Back to top](#contents)  

##### Types of transfer learning
    "As is" transfer learning is when you take a pretrained model as it is and apply it to your task without any changes.

        For example, many computer vision models are pretrained on the ImageNet dataset which contains 1000 different classes of images. This means passing a single image to this model will produce 1000 different prediction probability values (1 for each class).
            This is helpful if you have 1000 classes of image you'd like to classify and they're all the same as the ImageNet classes, however, it's not helpful if you want to classify only a small subset of classes (such as 10 different kinds of food). Model's with "/classification" in their name on TensorFlow Hub provide this kind of functionality.

    Feature extraction transfer learning is when you take the underlying patterns (also called weights) a pretrained model has learned and adjust its outputs to be more suited to your problem.

        For example, say the pretrained model you were using had 236 different layers (EfficientNetB0 has 236 layers), but the top layer outputs 1000 classes because it was pretrained on ImageNet. To adjust this to your own problem, you might remove the original activation layer and replace it with your own but with the right number of output classes. The important part here is that only the top few layers become trainable, the rest remain frozen.
            This way all the underlying patterns remain in the rest of the layers and you can utilise them for your own problem. This kind of transfer learning is very helpful when your data is similar to the data a model has been pretrained on.

    Fine-tuning transfer learning is when you take the underlying patterns (also called weights) of a pretrained model and adjust (fine-tune) them to your own problem.
        This usually means training some, many or all of the layers in the pretrained model. This is useful when you've got a large dataset (e.g. 100+ images per class) where your data is slightly different to the data the original model was trained on.

A common workflow is to "freeze" all of the learned patterns in the bottom layers of a pretrained model so they're untrainable. And then train the top 2-3 layers of so the pretrained model can adjust its outputs to your custom data (feature extraction).

After you've trained the top 2-3 layers, you can then gradually "unfreeze" more and more layers and run the training process on your own data to further fine-tune the pretrained model.

[Back to top](#contents)  

##### Transfer learning feature extraction
end to end example:
https://github.com/CsLHegedus/Deep-learning-projects/blob/main/Transfer_learning_feature_extraction_end_to_end_example.ipynb

##### Tensorboard
```
Only continue if you are okay with that your experiments are publicly available  
Comparing models with tensorflow (in this example just one architecture is shown)

# Upload Tensorboard dev records
!tensorboard dev upload --logdir ./tensorflow_hub/ \
--name "ResNet50V2" \
--description "Showing the result of one TF Hub feature extraction model architectures"
--one_shot
```
Treat authorization code like a password
```
example 
https://tensorboard.dev/experiment/6JvGksvdTVqUz2OZtRnE5A/#scalars&runSelectionState=eyJyZXNuZXQ1MFYyLzIwMjEwNzE5LTE3NDI1Ny90cmFpbiI6ZmFsc2UsInJlc25ldDUwVjIvMjAyMTA3MTktMTc0MjU3L3ZhbGlkYXRpb24iOnRydWV9
```
```
Check out your experiments on TensorBoard
!tensorboard dev list
```
```
# Delete experiment
!tensorboard dev delete --experiment_id u3pYYrxnSZKWE69yoKkF9g
```

##### Feature extraction vs fine tuning
```
def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares feature extraction results to fine tuning.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
```
```
compare_historys(original_history=history_10_percent_data_aug, 
                 new_history=history_fine_10_percent_data_aug, 
                 initial_epochs=5)
```
##### Checkpoint
It is used to save weights (or even the whole model) per given number of epochs
```
# Setup checkpoint path
checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.ckpt" # note: remember saving directly to Colab is temporary

# Create a ModelCheckpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, # set to False to save the entire model
                                                         save_best_only=False, # set to True to save only the best model instead of a model every epoch 
                                                         save_freq="epoch", # save every epoch
                                                         verbose=1)
```
```
# Fit the model saving checkpoints every epoch
initial_epochs = 5
history_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                          epochs=initial_epochs,
                                          validation_data=test_data,
                                          validation_steps=int(0.25 * len(test_data)), # do less steps per validation (quicker)
                                          callbacks=[create_tensorboard_callback("transfer_learning", "10_percent_data_aug"), 
                                                     checkpoint_callback])
```
```
# load saved weights
model_2.load_weights(checkpoint_path)
```
##### Transfer learning fine tuning
Transfer learning fine tuning example:  
https://github.com/CsLHegedus/Deep-learning-projects/blob/main/Transfer_learning_Fine_tuning_end_to_end_example.ipynb
