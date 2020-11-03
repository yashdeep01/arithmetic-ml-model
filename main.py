import numpy as np
import math
from sklearn.model_selection import train_test_split

# TODO: Write requirements.txt, README.md
# TODO: Create separate files for each class representing each arithmetic model
# TODO: Create dump files (pickles) for storing training, testing and results
# TODO: Try DOCKERFILE to containerize this git repo


class MachineLearningModel:
    """
    ML Models for simple arithmetic operations. Supports:
        - Multiplication
        - Addition
        - Subtraction
    """
    # TODO: Allow other operations like division, squaring, sq. root, etc.

    X_train, X_test, y_train, y_test = [], [], [], []

    def __init__(self, model_name, X, Y, lr=0.016, weight=0.0, n_epochs=20, test_size=0.2):
        self.X = X
        self.Y = Y
        self.model_name = model_name
        self.learning_rate = lr
        self.weight = weight
        self.n_epochs = n_epochs
        self.test_size = test_size
        self.loss = None

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
        if self.model_name == 'multiplication':
            return True, self.weight * X
        elif self.model_name == 'addition':
            return True, self.weight + X
        elif self.model_name == 'subtraction':
            return True, self.weight + X
        else:
            return False, None

    def compute_loss(self, y_pred, y):
        """
        Computes loss in the form of SME
        :param y_pred: predicted values from the model
        :param y: ground truth values of the arithmetic operation
        :return: None
        """
        self.loss = np.square((y_pred - y)).mean()

    def backprop(self, X, Y):
        """
        Computes gradient manually using chain rule
        :param X: Training data points
        :param Y: Training labels (Ground truth)
        :return dw: Gradient
        """
        if self.model_name == 'multiplication':
            dw = np.multiply(2 * X, self.weight * X - Y).mean()
        elif self.model_name == 'addition':
            dw = 2 * (X + self.weight - Y).mean()
        elif self.model_name == 'subtraction':
            dw = 2 * (X - self.weight - Y).mean()
        else:
            dw = None
        return dw

    def gradient_descent(self, dw):
        """
        Applying non-stochastic gradient descent by updating values of the weight.
        :param dw:
        :return:
        """
        # TODO: Apply SGD and optimizer (Adam)
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
            ret, y_pred = self.forward(self.X_train)
            if ret is False:
                print(f"Error: 'forward()' returned false.")
                break

            self.compute_loss(y_pred, self.y_train)
            dw = self.backprop(self.X_train, self.y_train)
            self.gradient_descent(dw)

            if epoch % 1 == 0:
                print(f"\tEpo {epoch + 1}\t: weight = {self.weight:.5f}, loss = {math.sqrt(self.loss):.8f}")
        print("\n*******************************************************")

    def test(self):
        if self.model_name == 'multiplication':
            y_pred = self.weight * self.X_test
        elif self.model_name == 'addition':
            y_pred = self.weight + self.X_test
        elif self.model_name == 'subtraction':
            y_pred = self.X_test - self.weight
        else:
            y_pred = None

        self.compute_loss(y_pred, self.y_test)
        print(f"Test results:\n\ty_pred: {y_pred}\n\ty_test: {self.y_test}\n\tLoss  : {math.sqrt(self.loss):.5f}")


def main():
    """
    Main logic of the program
    """
    # TODO: Use argparser to take input from command-line arguments

    operation = 'multiplication'
    operand = 3
    # TODO: Allow arithmetic operations between arrays, instead of constant and array

    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

    # Generating labels (ground truth)
    if operation == 'multiplication':
        Y = operand * X
    elif operation == 'addition':
        Y = operand + X
    elif operation == 'subtraction':
        Y = X - operand
    else:
        # TODO: Exception handling
        print(f"Error: \'{operation}\' operation doesn\'t exist!")
        return
    print("-------------------------------------------------------")
    print(f"Arithmetic operation: {operation}")
    print(f"Operand: {operand}")
    print(f"X: {X}")
    print(f"Y: {Y}\n")

    # Defining the model, training and testing it.
    # TODO: Apply multiprocessing to exploit multi-core CPUs
    model = MachineLearningModel(operation, X, Y, lr=0.015, n_epochs=15)
    print(f"\nPrediction before training : \n\t{model.weight * model.X_train}\n")
    model.train()
    print(f"\nPrediction after training  : \n\t{model.weight * model.X_train}\n")
    print("=======================================================")
    model.test()
    print("=======================================================")


# Starting point of execution
if __name__ == '__main__':
    # TODO: Create a .yaml or config file to fetch input arguments
    main()
