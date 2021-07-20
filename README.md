<h1 align="center">
    Arithmetic with ✨ Machine Learning ✨
</h1>

<p align="center">
    A <i>silly</i> use-case of machine learning model. Watch <b>ML</b> ⚡ in <i>action!</i> 
</p>


## 🤔 What is it?
Implementation of simple arithmetic operations—multiplication, addition, and subtraction—on an array of natural numbers with a single-neuron deep learning network (basically _linear regression!_). 

`main` branch consists of manual implementation of the model with **numpy**.

`pytorch` branch implements the above model using the **pytorch** library.

## 🚀 Open with Streamlit
Visualise the gradient descent and linear regression happening right in front of you on your browser. Tune the hyperparamters as you watch your model **learn** *live!*

![Demo](./assets/images/streamlit-streamlit_app-2021-07-20-15-07-10.gif)

## 💻 Running with CLI
### Create conda environment 🐍
> Install the anaconda package from [here](https://docs.anaconda.com/anaconda/install/) and run these commands on terminal:
```
conda init
conda create -n arithmetic-ml python=3.8
conda activate arithmetic-ml
```
### Clone this repo 🔗
```
git clone https://github.com/yashdeep01/arithmetic-ml-model.git
cd arithmetic-ml-model/
pip install -r requirements.txt
```
### Run 🛠️
```
python main.py
```
Run with custom hyperparameters:
```
python main.py --operand 2 --lr 0.018 --epochs 20
```
Get full list of command-line arguments:
```
python main.py -h
```

---

## ℹ️ About the model
A single neuron deep learning neural network with identity activation function; in other words, *linear regression*. 

**Input** `1,2,3,...,n`  
**Target** `c,2c,3c,...nc`, *where c is a constant operand*

### ⚙️ Training 
Steps in each epoch:

1. **Forward propagation** ➡️  
Computes the standard equation `w*X + b` in forward propagation. Identity activation function.

2. **Loss computation** ⚠️  
Calculates Mean Squared Error (MSE) loss at the end of the forward propagation.

3. **Back propagation** ⬅️  
Gradient of loss function with respect to weight `w` and bias `b` obtained: `dw` and `db` respectively.

4. **Gradient descent** 📉  
Updates weight and bias by stepping against the gradient direction in the magnitude of learning rate `lr`.
    ```
    w' = w - lr*dw
    b' = b - lr*db
    ```
Since training dataset is small, batch size is the full training set.

### 🏁 Validation
Predicts targets for numbers in validation set using the final weight and bias, then reports validation loss.

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

## 📚 Acknowledgement
Inspired from tutorial: https://github.com/python-engineer/pytorchTutorial
