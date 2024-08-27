import torch
import matplotlib.pyplot as plt
import numpy as np

# Observed/training input and output
data = np.loadtxt('https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_head_circumference.csv', delimiter=',', skiprows=1)

# Normalize the input data to avoid sigmoid saturation
x_train = torch.tensor(data[:, 0], dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(data[:, 1], dtype=torch.float32).reshape(-1, 1)

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.01]], requires_grad=True)  # Initialize W with a small value
        self.b = torch.tensor([[0.01]], requires_grad=True)  # Initialize b with a small value

    # Predictor
    def f(self, x):
        return 20 * sigmoid(x @ self.W + self.b) + 31  # Sigmoid applied to linear combination of xW + b

    # Loss function
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Mean Squared Error

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam([model.W, model.b], lr=0.005)  # Further reduced learning rate
for epoch in range(5000):
    optimizer.zero_grad()
    loss_value = model.loss(x_train, y_train)
    loss_value.backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b

    # Print the intermediate values to debug
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: W = {model.W.item()}, b = {model.b.item()}, loss = {loss_value.item()}")

def f(x):
    return 20 * sigmoid(x @ model.W + model.b) + 31
# Print model variables and final loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result

# Plot the original data points
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')  # Corrected line

# Generate a range of x values for plotting the function f(x)
x_values = torch.linspace(torch.min(x_train), torch.max(x_train), 100).reshape(-1, 1)

# Plot the non-linear function f(x)
plt.plot(x_values, model.f(x_values).detach(), label='$f(x) = 20 * sigmoid(xW + b) + 31$', color='orange')

# Add labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()