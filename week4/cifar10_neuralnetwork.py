""" 
Introduction to Pattern Recognition and Machine Learning
Sara Hirvonen
Exercise 4 Visual classification with NNets

"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import keras
import numpy as np

def main():
    training_data, testing_data = tf.keras.datasets.cifar10.load_data()
    X = training_data[0]
    Y = training_data[1]

    test_data = testing_data[0]
    test_labels = testing_data[1]

    X = X.astype('float32')
    test_data = test_data.astype('float32')
    X = X / 255.0
    test_data = test_data / 255.0

    # Creating neural network model
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation ='sigmoid'))


    # Compiling the model
    keras.optimizers.SGD(lr=0.7)
    model.compile(optimizer='sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

    # Training
    model.fit(X, Y, epochs=10)

    # Testing
    test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=1)
    print("Test accuracy:", test_acc)

main()