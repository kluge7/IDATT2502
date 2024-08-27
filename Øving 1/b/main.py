import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load the data
data = np.loadtxt('https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_length_weight.csv', delimiter=',', skiprows=1)

# Separate the data into three arrays and convert to PyTorch tensors
x_train = torch.tensor(data[:, 0], dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data[:, 1], dtype=torch.float32).reshape(-1, 1)
z_train = torch.tensor(data[:, 2], dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel3D:
    def __init__(self):
        # Model variables
        self.W1 = torch.tensor([[0.0]], requires_grad=True)  # Weight for x
        self.W2 = torch.tensor([[0.0]], requires_grad=True)  # Weight for y
        self.b = torch.tensor([[0.0]], requires_grad=True)   # Bias

    # Predictor
    def f(self, x, y):
        return x @ self.W1 + y @ self.W2 + self.b  # Linear function for z

    # Mean Squared Error loss function
    def loss(self, x, y, z):
        return torch.mean(torch.square(self.f(x, y) - z))

# Instantiate the model
model = LinearRegressionModel3D()

# Set up the optimizer
optimizer = torch.optim.SGD([model.W1, model.W2, model.b], lr=0.0000001)

# Train the model
for epoch in range(15000):
    loss_value = model.loss(x_train, y_train, z_train)
    loss_value.backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization
    optimizer.zero_grad()  # Clear gradients

# Print final model parameters and loss
print("W1 = %s, W2 = %s, b = %s, loss = %s" % (model.W1, model.W2, model.b, model.loss(x_train, y_train, z_train)))

# Visualize the result
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Scatter plot of the data points
ax.scatter(x_train.numpy(), y_train.numpy(), z_train.numpy(), color='blue', label='Data Points')

# Create a grid of x and y values
x_grid, y_grid = torch.meshgrid(torch.linspace(torch.min(x_train), torch.max(x_train), 100),
                                torch.linspace(torch.min(y_train), torch.max(y_train), 100))

# Calculate the corresponding z values based on the model
z_grid = model.f(x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)).detach().numpy().reshape(100, 100)

# Plot the regression plane
ax.plot_surface(x_grid.numpy(), y_grid.numpy(), z_grid, color='orange', alpha=0.5)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
