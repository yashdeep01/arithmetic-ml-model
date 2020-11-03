# coding=utf-8

"""
Command line interface for elementary arithmetic operations using ML
"""

import torch
import torch.nn as nn
import argparse
import numpy as np
import math
from sklearn.model_selection import train_test_split

# Sincere thanks for inspiring: https://github.com/python-engineer/pytorchTutorial

# TODO: Create separate files for each class representing each arithmetic model
# TODO: Create dump files (pickles) for storing training, testing and results
# TODO: Try DOCKERFILE to containerize this git repo

parser = argparse.ArgumentParser(allow_abbrev=False,
                                 description='Arithmetic operation ML model')
parser.add_argument('-m', '--model',
                    type=str,
                    choices=['mul', 'add', 'sub'],
                    default='mul',
                    help='Arithmetic operation to perform')
parser.add_argument('-o', '--operand',
                    type=float,
                    default=3.0,
                    help='Set operand to perform operation with')
parser.add_argument('-l', '--lr',
                    type=float,
                    default=0.2212,
                    help='Set learning rate')
parser.add_argument('-w', '--weight',
                    type=float,
                    default=0.0,
                    help='Set initial weight (operand)')
parser.add_argument('-e', '--epochs',
                    type=int,
                    default=150,
                    help='Set number of epochs')
parser.add_argument('--step',
                    type=int,
                    default=10,
                    help='Print every \'skip\'th epoch')
parser.add_argument('-t', '--test_size',
                    type=float,
                    default=0.2,
                    help='Test ratio')

args = parser.parse_args()


class MachineLearningModel:
    """
    ML Models for simple arithmetic operations. Supports:
        - Multiplication
        - Addition
        - Subtraction
    """
    # TODO: Allow other operations like division, squaring, sq. root, etc.

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.model_name = args.model
        self.learning_rate = args.lr
        self.weight = torch.tensor(args.weight, dtype=torch.float32, requires_grad=True)
        self.n_epochs = args.epochs
        self.test_size = args.test_size
        self.loss = None

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam([self.weight], lr=self.learning_rate)

        # TODO: Apply n-fold cross-validation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_size,
                                                                                shuffle=False)
        self.print_params()

    def print_params(self):
        print("*************** Hyper-Parameters *****************")
        print(f"\tArithmetic model: {self.model_name}")
        print(f"\tLearning rate: {self.learning_rate}")
        print(f"\tInitial weight: {self.weight}")
        print(f"\tEpochs: {self.n_epochs}")
        print(f"\tTrain-test ratio: {1-self.test_size}-{self.test_size}\n")

        print("****************** Dataset ***********************")
        print(f"\tX_train: {self.X_train}")
        print(f"\ty_train: {self.y_train}")
        print(f"\tX_test : {self.X_test}")
        print(f"\ty_test : {self.y_test}\n")

    def forward(self, X):
        """
        Computes arithmetic operation for numpy array `X`
        :param X: numpy array of train/test data points
        :return:
        """
        if self.model_name == 'mul':
            return True, self.weight * X
        elif self.model_name == 'add':
            return True, self.weight + X
        elif self.model_name == 'sub':
            return True, self.weight + X
        else:
            return False, None

    def compute_loss(self, y_pred, y):
        """
        Computes loss in the form of MSE
        :param y_pred: predicted values from the model
        :param y: ground truth values of the arithmetic operation
        :return: None
        """
        self.loss = ((y_pred - y)**2).mean()

    def backprop(self, X, Y):
        """
        Computes gradient manually using chain rule, redundant with pytorch.
        :param X: Training data points
        :param Y: Training labels (Ground truth)
        :return dw: Gradient
        """
        if self.model_name == 'mul':
            dw = np.multiply(2 * X, self.weight * X - Y).mean()
        elif self.model_name == 'add':
            dw = 2 * (X + self.weight - Y).mean()
        elif self.model_name == 'sub':
            dw = 2 * (X - self.weight - Y).mean()
        else:
            dw = None
        return dw

    def gradient_descent(self, dw):
        """
        Applying non-stochastic gradient descent by manually updating values of the weight.
        :param dw: Gradient of loss wrt weight
        :return:
        """
        # TODO: Apply SGD and optimizer (Adam)
        # We do not want this operation to be part of our computation graph for grad calculation
        # Hence, omitting grad calculation.
        with torch.no_grad():
            self.weight -= self.learning_rate * dw

    def train(self):
        """
        Training workflow of the arithmetic operation model.
        In each epoch:
            1. Forward propagation: Compute arithmetic operation
            2. Compute loss: Square-mean-error between label and train/test
            3. Backward propagation: Compute gradients
            4. Gradient descent: Update weight using learning rate
        """
        # TODO: Introduce batch size, optimizer

        print("*********************** Training ***********************\n")
        for epoch in range(self.n_epochs):

            ### Forward prop
            ret, y_pred = self.forward(self.X_train)
            if ret is False:
                print(f"Error: 'forward()' returned false.")
                break

            ### Compute loss
            # self.compute_loss(y_pred, self.y_train)         # manual computation
            self.loss = self.loss_fn(self.y_train, y_pred)

            ### Update weights
            # dw = self.backprop(self.X_train, self.y_train)  # manual computation
            self.loss.backward()
            # dw = self.weight.grad
            # self.gradient_descent(dw)
            self.optimizer.step()

            ### Making gradients zero so as to prevent old values in grad calculation
            # dw.zero_()
            self.optimizer.zero_grad()

            if epoch % args.step == 0:
                print(f"\tEpo {epoch + 1}\t: weight = {self.weight:.5f}, loss = {math.sqrt(self.loss):.8f}")

        print("\n*******************************************************")

    def test(self):
        if self.model_name == 'mul':
            y_pred = self.weight * self.X_test
        elif self.model_name == 'add':
            y_pred = self.weight + self.X_test
        elif self.model_name == 'sub':
            y_pred = self.X_test - self.weight
        else:
            y_pred = None

        self.compute_loss(y_pred, self.y_test)
        print(f"Test results:\n\ty_pred: {y_pred}\n\ty_test: {self.y_test}\n\tLoss  : {math.sqrt(self.loss):.5f}")


def main():
    """
    Main logic of the program
    """
    # TODO: Allow arithmetic operations between arrays, instead of constant and array

    # Input data
    X = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)

    # Generating labels (ground truth)
    if args.model == 'mul':
        Y = args.operand * X
    elif args.model == 'add':
        Y = args.operand + X
    elif args.model == 'sub':
        Y = X - args.operand
    else:
        print(f"Error: \'{args.model}\' operation does not exist!")
        return
    print("-------------------------------------------------------")
    print(f"Arithmetic operation: {args.model}")
    print(f"Operand: {args.operand}")
    print(f"X: {X}")
    print(f"Y: {Y}\n")

    # Defining the model, training and testing it.
    ml_model = MachineLearningModel(X, Y)
    print(f"\nPrediction before training : \n\t{ml_model.weight * ml_model.X_train}\n")
    ml_model.train()
    print(f"\nPrediction after training  : \n\t{ml_model.weight * ml_model.X_train}\n")
    print("=======================================================")
    ml_model.test()
    print("=======================================================")


# Starting point of execution
if __name__ == '__main__':
    main()
