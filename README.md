# grad
*A tiny educational library for auto-differentiation and neural nets training.*

# Features
- Define arbitrary scalar or tensor functions
- Requires definition of forward pass only
- Automatic function differentiation
- Automatic and dynamic computational graph creation
- Automatic forward pass over computational graph
- Automatic backward pass over computational graph
- Simple API for training neural nets
- `NumPy` backend
- `PyTorch` inspired

# Ops
- Add
- Multiply
- Linear
- Sigmoid
- ReLU

# Losses
- MSE

# Optimizers
- SGD

# Installation
*Built with python 3.8.1*
1. Clone the repo into `<dir>`
2. `cd <dir>`
3. `python -m venv .venv`
4. `source .venv/bin/activate`
5. `pip install pip --upgrade`
6. `pip install -r requirements.txt`

# Examples
- `examples/scalar_func.py`: define, evaluate and differentiate custom scalar function
- `examples/regression.py`: build and train simple neural net for regression task 

# Tests
*Ensure correct installation by running tests:*  
- `python -m unittest discover tests/native -bv`  

*Ensure correct implementation by running tests against `PyTorch` (requires `pytest` and `torch` installed):*  
- `pytest tests/torch/test_ops -vs --tb=no`  

# Roadmap
- Matmul, Softmax
- Cross-entropy loss
- Conv2D, MaxPool2D
- Tensor ops (reshape, transpose, max etc.)
- Attention, Self-Attention
