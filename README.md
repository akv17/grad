# grad
*A tiny educational library for auto-differentiation and neural nets training.*

# Features
- Define arbitrary scalar or tensor functions
- Requires definition of forward pass only
- Automatic function differentiation
- Automatic computational graph creation
- Automatic forward pass over computational graph
- Automatic backward pass over computational graph
- Simple API for training neural nets
- `NumPy` backend
- `PyTorch` inspired

# Examples
- `examples/computational_graph.py`: define, evaluate and differentiate custom function
- `examples/regression.py`: build and train simple neural net for regression task 


# Installation
*Built with python 3.8.1*
1. Clone the repo into `<dir>`
2. `cd <dir>`
3. `python -m venv .venv`
4. `source .venv/bin/activate`
5. `pip install pip --upgrade`
6. `pip install -r requirements.txt`

# Test
*Ensure correct installation via running tests*
1. `python -m unittest discover tests -bv`
