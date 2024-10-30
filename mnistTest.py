import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the images

# Pulling off the first entry and printing it to see what it looks like
# It happens to be a 5 (which you can double-check with the corresponding
# label
# I'm using matplotlib.pyplot.spy to print it...
# This probably isn't the best for this data since there's actually
# some grayscale data embedded, but we're going to use spy for matrices
# since we care less about the scale/magnitude of the entries and more just about
# where there are non-zeros
plt.spy(x_train[0])
print(y_train[0])
plt.show()

# Build the CNN model
# This is equivalent to the build_model function in train.py
# Please experiment with this, and change things around
# The only things that need to stay (or something close to them):
#   1. The shape of the Input layer should be (28, 28, 1)
#       All the handwritten digits are stored as 28x28 pixel arrays and we're taking a
#       single slice
#   2. The last layer ("Output layer") should map to the 10 digits
#       This is our prediction:
#           Is the digit we analyzed a: 0, 1, 2, 3, 4, 5, 6, 7, 8, or 9
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Input layer
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),  # Convolutional layer
    layers.MaxPooling2D(pool_size=(2, 2)),  # Max-pooling layer
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Convolutional layer
    layers.MaxPooling2D(pool_size=(2, 2)),  # Max-pooling layer
    layers.Flatten(),  # Flatten the output
    layers.Dense(128, activation='relu'),  # Fully connected layer
    layers.Dense(10, activation='softmax')  # Output layer
])

# Compile the model
# I think of this as an extra step in between "build" and "train", but it's really just
# having Python/Tensorflow Glue the whole thing together along with definitions for how
# we "optimize", categorize "loss", and what metrics we care about
# In train.py, this is just the last line inside of build_model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# This is equivalent to the function train_network in train.py
# More epochs tends to lead to higher accuracy, but there's definitely a "knee in the curve"
# where more training doesn't lead to better results
#
# This data set (MNIST) is simple enough that a small number of training Epochs is
# probably sufficient
# The block-prediction CNN should probably be trained for quite a few more Epochs, but
# the default (that, honestly, I probably set) is low to help the program run quickly
#   ***Just a note of caution that this can be misleading since some neural networks may
#      have significantly worse performance, but may continue making progress as they're
#      trained for longer
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Evaluate the model
# I don't see a whole lot of "evaluation" stored in what's in the initial github repo
# The whole file "predict.py" preprocesses the data to make it more appropriate for use in
# the model developed in train.py and then creates a set of predictions
#
# To my eye, it looks like the evaluation occurred either offline (e.g., interactively
# in the Python shell) or in a separate file that wasn't loaded in the github
#   The latter seems more likely to me; there are definitely files on the front end
#   that aren't included in the repo
#
# I'm leaving the last bit here like this since I think this will more closely match
# references/artifacts you see "out in the wild"
#   We could "manually" find the evaluation by first creating a prediction from the model
#   using the test data and the compare it to the test labels
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# Everything below here is just some simple graphs
# Feel free to play around with this (and anything else you find)


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()