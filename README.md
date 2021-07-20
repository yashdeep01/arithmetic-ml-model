# Arithmetic ML model
A *silly* use-case of machine learning model to watch ML in action. 

## What is it?
Implementation of simple arithmetic operations—multiplication, addition, and subtraction—on an array of natural numbers with a single-neuron deep learning network (basically linear regression!). 

`main` branch consists of manual implementation of the model with **numpy**.

`pytorch` branch implements the above model using the **pytorch** library.

## Open with Streamlit
Visualise the gradient descent and linear regression on your browser. Tune the hyperparamters as you watch your model **learn** *live!*

![Demo](./assets/images/streamlit-streamlit_app-2021-07-20-15-07-10.gif)

## Running with CLI
### Create conda environment
> Install the anaconda package from [here](https://docs.anaconda.com/anaconda/install/) and run these commands on terminal:
```
conda init
conda create -n arithmetic-ml python=3.8
conda activate arithmetic-ml
```
### Clone this repo
```
git clone https://github.com/yashdeep01/arithmetic-ml-model.git
cd arithmetic-ml-model/
pip install -r requirements.txt
```
### Run
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

## About the model
A single neuron deep learning neural network with identity activation function; in other words, *linear regression*. 

**Input** `1,2,3,...,n`  
**Target** `c,2c,3c,...nc`, *where c is a constant operand*

### Training 
Steps in each epoch:

1. **Forward propagation**  
Computes the standard equation `w*X + b` in forward propagation. Identity activation function.

2. **Loss computation**  
Calculates Mean Squared Error (MSE) loss at the end of the forward propagation.

3. **Back propagation**  
Gradient of loss function with respect to weight `w` and bias `b` obtained: `dw` and `db` respectively.

4. **Gradient descent**  
Updates weight and bias by stepping against the gradient direction in the magnitude of learning rate `lr`.
    ```
    w' = w - lr*dw
    b' = b - lr*db
    ```
Since training dataset is small, batch size is the full training set.

### Validation
Predicts targets for numbers in validation set using the final weight and bias, then reports validation loss.

