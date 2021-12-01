import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist  # Object of the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Load data

# Normalize the train dataset
x_train = tf.keras.utils.normalize(x_train, axis=1)
# Normalize the test dataset
x_test = tf.keras.utils.normalize(x_test, axis=1)

n_input = 784  # input layer (28x28 pixels)
n_hidden1 = 150  # 1st hidden layer
n_hidden2 = 150  # 2nd hidden layer
n_output = 10  # output layer (0-9 digits)

# Build the model object
model = tf.keras.models.Sequential()
# Add the Flatten Layer
model.add(tf.keras.layers.Flatten())
# Build the input and the hidden layers
model.add(tf.keras.layers.Dense(n_input, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(n_hidden1, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(n_hidden2, activation=tf.nn.relu))
# Build the output layer
model.add(tf.keras.layers.Dense(n_output, activation=tf.nn.softmax))

# Compile the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x=x_train, y=y_train, epochs=3)  # Start training process

# Evaluate the model performance
test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
# Print out the model accuracy
print('\nTest accuracy:', test_acc)

model.save('./model/mnistLearner.h5')
