import torch
import matplotlib.pyplot as plt
import numpy as np

# Observed/training input and output for XOR operator
x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

class XOROperatorModel:
    def __init__(self):
        # Model variables initialized randomly between -1 and 1
        self.W1 = torch.nn.Parameter(torch.rand((2, 2), dtype=torch.float32) * 2 - 1)  
        self.b1 = torch.nn.Parameter(torch.rand((1, 2), dtype=torch.float32) * 2 - 1)  
        self.W2 = torch.nn.Parameter(torch.rand((2, 1), dtype=torch.float32) * 2 - 1)  
        self.b2 = torch.nn.Parameter(torch.rand((1, 1), dtype=torch.float32) * 2 - 1)  

    # Predictor
    def f(self, x):
        hidden = torch.sigmoid(x @ self.W1 + self.b1)  # First layer with sigmoid activation
        output = torch.sigmoid(hidden @ self.W2 + self.b2)  # Second layer with sigmoid activation
        return output

    # Uses Binary Cross Entropy Loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)

model = XOROperatorModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], lr=0.5)
num_epochs = 10000

for epoch in range(num_epochs):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step
    if (epoch % 1000 == 0):
        print(f"Epoch {epoch}/{num_epochs}, Loss: {model.loss(x_train, y_train)})")


# Print model variables and loss
print("W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" % (model.W1, model.b1, model.W2, model.b2, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure()
plot1 = fig.add_subplot(111, projection='3d')

# Plot the original training points
plot1.scatter(x_train[:, 0].numpy(), x_train[:, 1].numpy(), y_train[:, 0].numpy(), color="red", label="$(x_1^{(i)}, x_2^{(i)},y^{(i)})$")

# Generate a grid of points
x1_grid, x2_grid = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_grid = np.empty([10, 10])

# Calculate the model's output for each point in the grid
for i in range(x1_grid.shape[0]):
    for j in range(x1_grid.shape[1]):
        y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]], dtype=torch.float32)).item()

# Plot the wireframe representing the model's predictions
plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="blue", label="$\\hat y=f(\\mathbf{x})=\\sigma(\\mathbf{xW}+b)$")

plot1.legend()
plt.show()
