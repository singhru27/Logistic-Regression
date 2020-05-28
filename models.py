#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
   This file contains the Logistic Regression classifier

   Brown CS142, Spring 2020
'''
import random
import numpy as np
import math


def softmax(x):
    '''
    Apply softmax to an array
    Given an array of inner products, it applys the exponentiation algorithm
    to return the probabilities of those inner products

    @params:
        x: the original array
    @return:
        an array with softmax applied elementwise.
    '''
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

class LogisticRegression:
    '''
    Multinomial Logistic Regression that learns weights using
    stochastic gradient descent.
    '''
    def __init__(self, n_features, n_classes, batch_size, conv_threshold):
        '''
        Initializes a LogisticRegression classifer.

        @attrs:
            n_features: the number of features in the classification problem
            n_classes: the number of classes in the classification problem
            weights: The weights of the Logistic Regression model
            alpha: The learning rate used in stochastic gradient descent
        '''
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.zeros((n_features + 1, n_classes))  # An extra row added for the bias
        self.alpha = 0.03  # tune this parameter
        self.batch_size = batch_size
        self.conv_threshold = conv_threshold

    def train(self, X, Y):
        '''
        Trains the model, using stochastic gradient descent

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            num_epochs: integer representing the number of epochs taken to reach convergence
        '''
        # Initializing variables
        num_epochs = 0
        num_examples = Y.size
        convergence_boolean = 0

        # Scrubbing the data to make even batch sizes
        remainder = num_examples % self.batch_size
        for i in range (0, remainder):
            X = np.delete (X, 0, 0)
            Y = np.delete (Y, 0, 0)

        # Finding the loss with unmodified weights
        loss = self.loss (X, Y)

        # Finding the number of batches to be created
        num_examples = X.shape[0]
        num_batches = num_examples/self.batch_size


        # Running SGD until convergence is achieved
        while convergence_boolean == 0:

            ## Shuffling the batches
            random_state = np.random.get_state ()
            np.random.shuffle (X)
            np.random.set_state (random_state)
            np.random.shuffle (Y)
            # Splitting the data into even batches of the desired size
            batched_data = np.vsplit(X, num_batches)


            for i in range (0, int(num_batches)):

                # Initializing array of partial derivatives and results
                derivatives_array = np.zeros ((self.batch_size, self.n_classes))

                # batch_array refers to a single batch of data
                batch_array = batched_data [i]
                batch_inner_products = np.matmul (batch_array, self.weights)

                # Computing the probabilities via the softmax function.
                probabilities = np.apply_along_axis (softmax, 1, batch_inner_products)

                # Setting partial derivatives for the individual weights
                for k in range (0, self.batch_size):
                    for l in range (0, self.n_classes):

                        # Case for when the class corresponds to the actual labeled class
                        if l == Y[(i * self.batch_size) + k]:
                            derivatives_array[k][l] = (probabilities[k][l] - 1)
                        else:
                            derivatives_array[k][l] = (probabilities[k][l])


                # Finding the outer product to be subtracted from the weights array
                weights_derivatives = np.dot (np.transpose(batch_array), derivatives_array)
                # Resetting the weights
                self.weights = self.weights - ((self.alpha * weights_derivatives)/self.batch_size)

            # incrementing the num_epochs
            num_epochs = num_epochs + 1
            # Checking if convergence has been reached
            if abs (loss - self.loss (X, Y)) < self.conv_threshold:
                return num_epochs

            else:
                loss = self.loss (X,Y)

    def loss(self, X, Y):
        '''
        Returns the total log loss on some dataset (X, Y), divided by the number of datapoints.
        @params:
            X: 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: 1D Numpy array containing the corresponding labels for each example
        @return:
            A float number which is the squared error of the model on the dataset
        '''

        ## Finding the number of examples in the 2D array
        num_examples = X.shape[0]

        # Computing the inner products
        inner_products = np.matmul (X, self.weights)

        # Computing probabilities via the softmax function
        probabilities = np.apply_along_axis (softmax, 1, inner_products)

        # Array containing logs of all the probabilities for the correct classification
        log_array = np.zeros ((num_examples))

        # Looping through each example, and adding the log of the associated probability
        # into the log_array
        for i in range (0, num_examples):

            ## Handling the case for which the probability has been set to 0
            if probabilities[i][Y[i]] == 0:
                probabilities[i][Y[i]] = .0001

            log_array[i] = (-1 * math.log(probabilities[i][Y[i]]))

        # Summing the logs and returning
        return (np.sum (log_array) / num_examples )

    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        # TODO. Looks to be complete

        # Number of examples in the data set
        num_examples = X.shape[0]

        # Creating a 1D numpy array to hold all the data
        result_array = np.zeros ((num_examples))

        # Computing the inner products
        inner_products = np.matmul (X, self.weights)

        # Computing the probabilities via the softmax function
        for i in range (0, num_examples):
            probabilities = softmax (inner_products[i])
            # Setting the class of each data set
            result_array[i] = int(probabilities.argmax())

        # Returning the predicted class
        return result_array

    def accuracy(self, X, Y):
        '''
        Outputs the accuracy of the trained model on a given testing dataset X and labels Y.

        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''

        # TODO. Completed but not debugged. Looks to be complete
        num_elements = X.shape[0]
        predicted_y = self.predict (X)

        num_correct = 0

        for i in range (num_elements):
            if predicted_y [i] == Y[i]:
                num_correct += 1

        return (num_correct/num_elements)
