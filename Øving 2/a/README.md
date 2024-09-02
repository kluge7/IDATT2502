# PyTorch example performing linear regression

## Prerequisites

Python3 should be installed, as well as the following python libraries: numpy, matplotlib and torch.

On Arch Linux/Manjaro you can install these packages using `pacman`:

```sh
sudo pacman -S python-numpy python-matplotlib python-pytorch
```

On other systems `pip3` can be used:

```sh
pip3 install numpy matplotlib torch torchvision
```

Note that CUDA optimized packages exist in case you have an Nvidia graphics card.

## Run example

```sh
python3 main.py
```

## Introduction

PyTorch is a machine learning library that include popular machine learning methods, but also reuses
much of the Numpy library API. Although, there are some differences:

```py
import torch
import numpy as np

# In Numpy one would use a multidimensional array to represent a matrix:
print(np.array([[1, 2], [3, 4]]))
# Output:
# [[1 2]
#  [3 4]]

# In PyTorch one instead use tensor, which basically is the same as np.array, to represent a matrix:
print(torch.tensor([[1, 2], [3, 4]]))
# Output:
# tensor([[1, 2],
#         [3, 4]])

# However, the usage of functions like matrix multiplication are similar:
print(np.array([[1, 2], [3, 4]]) @ np.array([[1, 2], [3, 4]]))
# Output:
# [[ 7 10]
#  [15 22]]
print(torch.tensor([[1, 2], [3, 4]]) @ torch.tensor([[1, 2], [3, 4]]))
# Output:
# tensor([[ 7 10]
#         [15 22]])
```

## Transforming data

### Reshape

The function `reshape()` is a commonly used in numpy/pytorch, and is very useful when working with
datasets:

```py
import torch

# Create a 2x2 matrix from a 1 dimensional table:
print(torch.reshape(torch.tensor([1, 2, 3, 4]), (2, 2)))
# Output:
# tensor([[1, 2],
#         [3, 4]])

# Can also specify one dimension as -1, and let reshape infer this dimension size
print(torch.reshape(torch.tensor([1, 2, 3, 4]), (-1, 2)))
# Output:
# tensor([[1, 2],
#         [3, 4]])
print(torch.reshape(torch.tensor([1, 2, 3, 4, 5, 6]), (-1, 2)))
# Output:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6])

# Another possibility is to convert an 1 dimensional table to a matrix with a single row:
print(torch.reshape(torch.tensor([1, 2, 3, 4, 5, 6]), (1, -1)))
# Output:
# tensor([[1, 2, 3, 4, 5, 6]])

# Or an 1 dimensional table to a single column matrix:
print(torch.reshape(torch.tensor([1, 2, 3, 4, 5, 6]), (-1, 1)))
# Output:
# tensor([[1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6]])

# You can also reshape a matrix to a vector (a method also called squeeze):
print(torch.reshape(torch.tensor([[1, 2, 3, 4, 5, 6]]), (-1, )))
# Output:
# tensor([1, 2, 3, 4, 5, 6])

# You can call reshape, and many other methods, on the tensor objects directly:
print(torch.tensor([[1, 2, 3, 4, 5, 6]]).reshape(-1, ))
# Output:
# tensor([1, 2, 3, 4, 5, 6])

# Reshape can act as transpose (T), but the resulting matrix shape is specified and easier to see
print(torch.tensor([[1, 2, 3, 4, 5, 6]]).T)
print(torch.tensor([[1, 2, 3, 4, 5, 6]]).reshape(-1, 1))
# Both outputs:
# tensor([[1],
#         [2],
#         [3],
#         [4],
#         [5],
#         [6]])

```

### Indexing

Although not used in main.py, PyTorch also supports Numpy-style indexing:

```py
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Read single value:
print(A[0, 1])
# Output:
# tensor(2)

# Read second row:
print(A[1, :])
# Output:
# tensor([4, 5, 6])

# Read first column:
print(A[:, 0])
# Output:
# tensor([1, 4])

# Read first and last columns:
print(A[:, [0, 2]])
# Output:
# tensor([[1, 3],
#         [4, 6]])
```

## Matrix multiplication recap

Consider a simple model `f(x) = xW`, where `f(x)` is simply an identity function `f(x) = x`, that is
`W = [[1]]`:

```py
import torch


class Model:
    def __init__(self):
        self.W = torch.tensor([[1]])

    def f(self, x):
        return x @ self.W


model = Model()

# We can call f(x) for each new observation x ([[1]] and [[3]]):
print(model.f(torch.tensor([[1]])))  # Equal to: [[1]] @ [[1]]
# Output: tensor([[1]])
print(model.f(torch.tensor([[3]])))  # Equal to: [[3]] @ [[1]]
# Output: tensor([[3]])

# Or we can call f(x) once where x contains all of the new observations ([[1]] and [[3]])
# in different rows:
print(model.f(torch.tensor([[1], [3]])))  # Equal to: [[1],
# Output: tensor([[1],                                 [3]] @ [[1]]
#                 [3]])
```

Another example making use of a model `f(x) = xW` that sums the two elements of the input
`x = [[x_1, x_2]]` (`W = [[1], [1]]`):

```py
import torch


class Model:
    def __init__(self):
        self.W = torch.tensor([[1], [1]])

    def f(self, x):
        return x @ self.W


model = Model()

# We can call f(x) for each new observation x ([[1, 2]] and [[3, 4]]):
print(model.f(torch.tensor([[1, 2]])))  # Equal to: [[1, 2]] @ [[1],
# Output: tensor([[3]])                                         [1]]
print(model.f(torch.tensor([[3, 4]])))  # Equal to: [[3, 4]] @ [[1],
# Output: tensor([[7]])                                         [1]]

# Or we can call f(x) once where x contains all of the new observations ([[1, 2]] and [[3, 4]])
# in different rows:
print(model.f(torch.tensor([[1, 2], [3, 4]])))  # Equal to: [[1, 2],
# Output: tensor([[3],                                       [3, 4]] @ [[1],
#                 [7]])                                                 [1]]
```

The same can be done during optimization of a model as well. The following example shows how the
loss, which can be used to optimize a model, can be calculated from all of the training data at
once.

```py
import torch

# Training data for a model that should sum the input elements x_1 and x_2,
# for instance from the last row of x_train and y_train: 2.0 + 3.0 = 5.0
x_train = torch.tensor([[1.0, 2.0],
                        [2.0, 2.0],
                        [2.0, 3.0]])
y_train = torch.tensor([[3.0],
                        [4.0],
                        [5.0]])


class Model:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.5], [0.5]])

    def f(self, x):
        return x @ self.W

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = Model()

# Demonstrates calculating loss of a model on the training data. The loss is used to optimize the model.
print(model.loss(x_train, y_train))  # model.W is now [[0.5], [0.5]]
# Output: tensor(4.1667)

# By changing the model manually we can achieve lower loss, which is better:
model.W = torch.tensor([[1.0], [1.0]])
print(model.loss(x_train, y_train))
# Output: tensor(0.)
```

## Linear regression

In [main.py](main.py), the linear regression model `f(x) = xW + b` is found by minimizing a loss
function `torch.mean(torch.square(f(x_train) - y_train))` called Mean Squared Error, where `x_train`
and `y_train` are the observed input and output we want to model. To minimize the loss function in
PyTorch, the model variables `W` and `b` are adjusted through a method called Stochastic Gradient
Descent (`torch.optim.SGD`). After the model variables `W` and `b` have converged, we can make new
predictions from new observations `x` using the model `f(x)`.

## Visualization

Below are two simple plot examples for 2D and 3D data.

### 2D

```py
import torch
import matplotlib.pyplot as plt

# Plot 3 random points
plt.plot(torch.rand(3), torch.rand(3), 'o')

# Line plot of x * x
x = torch.arange(start=0., end=1., step=0.01)  # Create the vector [0., 0.01, ..., 0.99]
plt.plot(x, x * x, label="$x^2$")  # x * x is element-wise multiplication resulting in a vector with the same shape as x

plt.xlabel('$x$')  # $$ activates LaTeX math notation
plt.ylabel('$y$')
plt.legend()
plt.show()
```

### 3D

```py
import torch
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')  # 3D plots require a subplot with 3D projection enabled

# Plot 3 random points (might need to convert to numpy array to avoid error...)
ax.plot(torch.rand(3).numpy(), torch.rand(3).numpy(), torch.rand(3).numpy(), 'o')

# Surface plot of x * x + y * y
# Create x and y matrices that will be used by the drawing function instead of a for-loop like:
# for y in torch.arange(start=0., end=1., step=0.01):
#     for x in torch.arange(start=0., end=1., step=0.01):
#         #  draw point at x, y, x * x + y * y
x = torch.arange(start=0., end=1., step=0.01)  # Create the vector [0., 0.01, ..., 0.99]
x = x.expand(x.shape[0], -1)  # Create the square matrix [[0., 0.01, ..., 0.99], ..., [0., 0.01, ..., 0.99]]
y = x.T  # Transpose of x: [[0., ..., 0.], [0.01, ..., 0.01], ... [0.99, ..., 0.99]]
ax.plot_wireframe(x, y, x * x + y * y, label="$x^2 + y^2$")  # * and + are element-wise operators resulting in a matrix with the same shape as x and y

ax.set_xlabel('$x$')  # $$ activates LaTeX math notation
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
plt.legend()
plt.show()
```

## Further reading

There are many PyTorch tutorials, but one that goes into slightly more detail is
[Understanding PyTorch with an example: a step-by-step tutorial](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
