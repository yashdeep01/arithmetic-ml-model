# coding=utf-8

# Streamlit app for elementary arithmetic operations using ML

import streamlit as st
import altair as alt
import pandas as pd
import argparse, time
import numpy as np
import math
from sklearn.model_selection import train_test_split
from urllib.error import URLError

# Sincere thanks for inspiring: https://github.com/python-engineer/pytorchTutorial

# TODO: Create separate files for each class representing each arithmetic model
# TODO: Create dump files (pickles) for storing training, testing and results
# TODO: Try DOCKERFILE to containerize this git repo

st.sidebar.write("""
## Model   
Pick the arithmetic operation you want to perform.
""")
model = st.sidebar.selectbox('Operation', ['Multiply','Add','Subtract'], 0)
n = st.sidebar.number_input('Data size', 10, 100, 20, 1, help='Natural numbers in train-val set')
operand = st.sidebar.number_input('Operand', 1, 20, 3, 1)

st.sidebar.write("""
## Hyperparameters
Tune the hyperparameters for the ML model.
""")
lr = st.sidebar.slider('Learning rate', 0.000, 0.100, 0.006, 0.001, '%3f')
epochs = st.sidebar.slider('Epochs', 0, 200, 55, 5)
train_size = st.sidebar.slider('Train-val split', 0.05, 0.95, 0.8, 0.05)
test_size = 1 - train_size

st.sidebar.write('Initialising learned weight (operand) as 0.0')
weight = 0.0
bias = 0.0

st.title("Arithmetics with Machine Learning")
st.write("""We all know simple addition and multiplication is no big deal. 
But what if we overkilled this small problem with Machine Learning? 
I create a single-neuron Deep Learning model (basically linear regression :P) to \'predict\' or estimate the outcome.  

Simply toggle the values on the sidebar and see *\"machine learning\"* in action!
""")

class MachineLearningModel:
    # ML Models for simple arithmetic operations. Supports:
    #     - Multiplication
    #     - Addition
    #     - Subtraction
    # TODO: Allow other operations like division, squaring, sq. root, etc.

    X_train, X_test, y_train, y_test = [], [], [], []

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.model_name = model
        self.learning_rate = lr
        self.weight = weight
        self.bias = bias
        self.n_epochs = epochs
        self.test_size = test_size
        self.loss = None
        self.t2 = None

        # TODO: Apply n-fold cross-validation
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=self.test_size,
                                                                                shuffle=True, random_state=42)

    def forward(self, X):
        # Computes arithmetic operation for numpy array `X`
        
        # if self.model_name == 'Multiply':
        #     return True, self.weight * X
        # elif self.model_name == 'Add':
        #     return True, self.weight + X
        # elif self.model_name == 'Subtract':
        #     return True, X - self.weight
        if self.model_name in ['Multiply','Add','Subtract']:
            return True, self.weight * X + self.bias
        else:
            return False, None

    def compute_loss(self, y_pred, y):
        # Computes loss in the form of SME
        self.loss = np.square((y_pred - y)).mean()

    def backprop(self, X, Y):
        # Computes gradient manually using chain rule
        
        # if self.model_name == 'Multiply':
        #     dw = np.multiply(2 * X, self.weight * X - Y).mean()
        # elif self.model_name == 'Add':
        #     dw = 2 * (X + self.weight - Y).mean()
        # elif self.model_name == 'Subtract':
        #     dw = 2 * (X - self.weight - Y).mean()
        if self.model_name in ['Multiply','Add','Subtract']:
            dw = np.multiply(2 * X, self.weight * X + self.bias - Y).mean()
            db = np.multiply(2, self.weight * X + self.bias - Y).mean()
        else:
            dw = None
            db = None
        return dw, db

    def gradient_descent(self, dw, db):
        # Applying non-stochastic gradient descent by updating values of the weight.
       
        # TODO: Apply SGD and optimizer (Adam)
        self.weight -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def train(self):
        # Training workflow of the arithmetic operation model.
        # In each epoch:
        #     1. Forward propagation: Compute arithmetic operation
        #     2. Compute loss: Square-mean-error between label and train/test
        #     3. Backward propagation: Compute gradients
        #     4. Gradient descent: Update weight using learning rate
        # TODO: Introduce batch size, optimizer
    
        # progress_bar = st.progress(0)
        # status_text = st.empty()

        chart_grad = st.empty() 
        chart_regr = st.empty()
        heading = st.empty() 
        table = st.empty()
        df = pd.DataFrame(columns=['Weight','Loss'])
        df_gt = pd.DataFrame({'X (Train data)': self.X_train, 
                             f'Y ({self.model_name} output)': self.y_train})
        
        t1 = time.time()
        for epoch in range(self.n_epochs):
            # status_text.text("%i%% Complete" % int(100*(epoch+1)/self.n_epochs))
            # progress_bar.progress(int(100*(epoch+1)/self.n_epochs))
            ret, y_pred = self.forward(self.X_train)
            if ret is False:
                print(f"Error: 'forward()' returned false.")
                break

            self.compute_loss(y_pred, self.y_train)
            dw,db = self.backprop(self.X_train, self.y_train)
            self.gradient_descent(dw, db)

            df = df.append(pd.DataFrame([[self.weight, self.loss]], columns=['Weight', 'Loss']))
            c = alt.Chart(df, title='Gradient Descent in action', width=350, height=350).mark_line(point=True, color='green').encode(
                x='Weight', y='Loss')
            
            df_reg = pd.DataFrame({'X (Train data)': self.X_train, 
                                  f'Y ({self.model_name} output)': y_pred})
            
            d = alt.Chart(df_gt, title='Linear Regression', width=350, height=300).mark_circle(color='red',radius=10).encode(
                x='X (Train data)', y=f'Y ({self.model_name} output)')
            e = alt.Chart(df_reg, title='Linear Regression', width=350, height=300).mark_line(point=False, color='blue').encode(
                x='X (Train data)', y=f'Y ({self.model_name} output)')
            
            # Gradient descent plot
            chart_grad.altair_chart(c.interactive(), use_container_width=True)
            
            # Regression plot
            chart_regr.altair_chart((d+e).interactive(), use_container_width=True)
            
            res_df = pd.DataFrame({'X (Train data)': self.X_train,
                                   'Operation': [self.model_name]*len(self.X_train),
                                   'Operand': [operand]*len(self.X_train),
                                   'Target (expected)': self.y_train,
                                   'Prediction': y_pred}
                                   ).sort_values(by=['X (Train data)'])
            
            # Training data results table
            heading.write("""### Results""")
            table.write(res_df)

            if epoch % 1 == 0:
                print(f"\tEpo {epoch + 1}\t: weight = {self.weight:.5f}, loss = {math.sqrt(self.loss):.8f}")
            time.sleep(0.05)
        self.t2 = time.time() - t1
        # progress_bar.empty()

    def test(self):
        ret, y_pred = self.forward(self.X_test)
        if ret is False:
            print(f"Error: 'forward()' returned false.")

        self.compute_loss(y_pred, self.y_test)
        val_df = pd.DataFrame({'X (Val data)': self.X_test,
                                'Operation': [self.model_name]*len(self.X_test),
                                'Operand': [operand]*len(self.X_test),
                                'Target (expected)': self.y_test,
                                'Prediction': y_pred}
                                ).sort_values(by=['X (Val data)'])
        st.write(val_df)
        st.write('Validation error (MSE Loss) = ', self.loss)

def main():
    # Main logic of the program
    # TODO: Allow arithmetic operations between arrays, instead of constant and array

    # Input data
    num_list = [i for i in range(1,n,1)]
    X = np.array(num_list, dtype=np.float32)

    # Generating labels (ground truth)
    if model == 'Multiply':
        Y = operand * X
    elif model == 'Add':
        Y = operand + X
    elif model == 'Subtract':
        Y = X - operand
    else:
        # TODO: Exception handling
        print(f"Error: \'{model}\' operation does not exist!")
        return
    
    # Defining the model, training and testing it.
    ml_model = MachineLearningModel(X, Y)
    st.write("""
    ## Training
    """)
    
    ml_model.train()

    st.write('Train-val set: Natural numbers till ', n)
    st.write('Predicted weight = ', ml_model.weight, ' and bias = ', ml_model.bias)
    st.write('Training error (MSE Loss) = ', ml_model.loss)
    st.write('Training time (sec): ', ml_model.t2)

    st.write("""
    ## Validation
    ### Results
    """)
    ml_model.test()

    st.write("""
    .  
    .  
    .  
    .  
    .  
    .  
    .  
    .  
    .  
    .  
    """)
    st.image('./assets/images/deep-learning.png')

    st.sidebar.button('Re-run')

# Starting point of execution
if __name__ == '__main__':
    main()
