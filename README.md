# Arithmetic ML model
A *really* simple machine learning model to grasp all major steps that make up a typical ML pipeline. The code here implements simple arithmetic operations on an array of floating numbers, viz multiplication, addition, and subtraction, with a constant operand. Machine learning is actually not required here since these elementary math operations can be implemented directly also, however it's a good way to grasp and appreciate machine learning.

`main` branch consists of manual implementation of the machine learning pipeline with **numpy**.

`pytorch` branch implements the above model using the **pytorch** library.

## Run
> It's advised to create a `conda` environment before running the code. Install the anaconda package from [here](https://docs.anaconda.com/anaconda/install/) and follow these commands:
``` bash
conda init
conda create -n arithmetic-ml-pytorch python=3.8
conda activate arithmetic-ml-pytorch
```

Run the following commands on terminal to set up the repo locally on your machine:

``` zsh
git clone https://github.com/yashdeep01/arithmetic-ml-model.git
cd arithmetic-ml-model/
pip install -r requirements.txt
```
Simply use this command to run the code:
~~~ bash
python main.py
~~~
Note that `main.py` also allows command line arguments to set hyperparameters, arithmetic operators and operands. 
For example, to set operand=2, learning rate=0.018 and number of epochs=20, we can run:
``` zsh
python main.py --operand 2 --lr 0.018 --epochs 20
```
Run this command for getting help on command line arguments to set various hyperparameters and other arguments:
``` zsh
python main.py -h
```

---

## Pipeline

The pipeline here consists of following steps:
1. Initialising the data, labels and parameters:
    - Input array of floating point numbers
    - Arithmetic operation to perform with ML
    - Operand (constant) to perform operation with
    - Hyperparameters, namely initial weight, learning rate, train-test ratio, and number of epochs.
2. Forward propagation
    - Compute the result of arithmetic operation of training set with given weight(s)
3. Loss computation
    - Mean square error (MSE) between label and prediction
4. Backward propagation
    - <del>Computing gradient manually using chain rule<del>
    - **pytorch** gradient computation
5. Gradient descent
    - Updating weight(s) based on a learning rate (Adam optimizer)

---

## Results
We apply a simple scalar multiplication operation on an array (tensor actually) of floating point numbers. In our example here, we have a tensor of 10 numbers `X`:

``` python
X: tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
```

We multiply them with our operand `3` to get our *labels* for training and testing `Y`:

```python
Y: tensor([ 3.,  6.,  9., 12., 15., 18., 21., 24., 27., 30.])
```
In our model, we intend to *learn* this operand from `Y`. We split our dataset into training and testing set in `4:1` ratio. We initialise our target operand `weight=0.0` and so initially our learned training set `X_train` looks like:

``` python
X_train: tensor([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
```
Finally, running the program produces the results below:

``` zsh
-------------------------------------------------------
Arithmetic operation: mul
Operand: 3.0
X: tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])
Y: tensor([ 3.,  6.,  9., 12., 15., 18., 21., 24., 27., 30.])

*************** Hyper-Parameters *****************
	Arithmetic model: mul
	Learning rate: 0.2212
	Initial weight: 0.0
	Epochs: 150
	Train-test ratio: 0.8-0.2

****************** Dataset ***********************
	X_train: tensor([1., 2., 3., 4., 5., 6., 7., 8.])
	y_train: tensor([ 3.,  6.,  9., 12., 15., 18., 21., 24.])
	X_test : tensor([ 9., 10.])
	y_test : tensor([27., 30.])


Prediction before training : 
	tensor([0., 0., 0., 0., 0., 0., 0., 0.], grad_fn=<MulBackward0>)

*********************** Training ***********************

	Epo 1	: weight = 0.22120, loss = 15.14925741
	Epo 11	: weight = 2.29846, loss = 4.47826647
	Epo 21	: weight = 3.52433, loss = 2.34178140
	Epo 31	: weight = 3.46506, loss = 2.57739488
	Epo 41	: weight = 2.95292, loss = 0.02822666
	Epo 51	: weight = 2.81649, loss = 0.96575477
	Epo 61	: weight = 2.99305, loss = 0.12585197
	Epo 71	: weight = 3.06633, loss = 0.34419884
	Epo 81	: weight = 3.00045, loss = 0.03727183
	Epo 91	: weight = 2.97624, loss = 0.12666698
	Epo 101	: weight = 3.00377, loss = 0.00693976
	Epo 111	: weight = 3.00756, loss = 0.04333535
	Epo 121	: weight = 2.99656, loss = 0.01431760
	Epo 131	: weight = 2.99848, loss = 0.01058778
	Epo 141	: weight = 3.00178, loss = 0.00896338

*******************************************************

Prediction after training  : 
	tensor([ 3.0000,  6.0000,  9.0000, 12.0000, 15.0000, 18.0000, 21.0000, 24.0000],
       grad_fn=<MulBackward0>)

=======================================================
Test results:
	y_pred: tensor([26.9999, 29.9999], grad_fn=<MulBackward0>)
	y_test: tensor([27., 30.])
	Loss  : 0.00006
=======================================================
```
