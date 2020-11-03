# Arithmetic ML model
A *really* simple machine learning model to grasp all major steps that make up a typical ML pipeline. The code here implements simple arithmetic operations on an array of floating numbers, viz multiplication, addition, and subtraction, with a constant operand. Machine learning is actually not required here since these elementary math operations can be implemented directly also, however it's a good way to grasp and appreciate machine learning.

`main` branch consists of manual implementation of the machine learning pipeline with **numpy**.

`pytorch` branch implements the above model using the **pytorch** library.

## Run
> It's advised to create a `conda` environment before running the code. Install the anaconda package from [here](https://docs.anaconda.com/anaconda/install/) and follow these commands:
```
conda init
conda create -n arithmetic-ml-pytorch python=3.8
conda activate arithmetic-ml-pytorch
```

Run the following commands on terminal to set up the repo locally on your machine:

```
git clone https://github.com/yashdeep01/arithmetic-ml-model.git
cd arithmetic-ml-model/
pip install -r requirements.txt
```
Simply use this command to run the code:
```
python main.py
```
Note that `main.py` also allows command line arguments to set hyperparameters, arithmetic operators and operands. 
For example, to set operand=2, learning rate=0.018 and number of epochs=20, we can run:
```
python main.py --operand 2 --lr 0.018 --epochs 20
```
Run this command getting help on command line arguments to set various hyperparameters and other arguments:
```
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
    - Computing gradient manually using chain rule
5. Gradient descent
    - Updating weight(s) based on a learning rate

