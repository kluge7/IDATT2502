import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

# Load MNIST data
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()
y_train = mnist_train.targets

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()
y_test = mnist_test.targets

# Define the model
class SoftmaxModel(nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        self.linear = nn.Linear(784, 10)  # W and b

    def f(self, x):
        return torch.softmax(self.linear(x), dim=1)
    
    def logits(self, x):
        return self.linear(x)
    
    def accuracy(self, x, y):
        predictions = self.f(x).argmax(dim=1)
        return torch.mean((predictions == y).float())

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y)

model = SoftmaxModel()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # Compute loss
    loss = model.loss(x_train, y_train)
    
    # Backpropagation
    loss.backward()
    optimizer.step()

    # Calculate and print accuracy on the test set
    with torch.no_grad():
        test_accuracy = model.accuracy(x_test, y_test)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {test_accuracy.item() * 100}%")

    # Early stopping if accuracy is 90% or higher
    if test_accuracy.item() >= 0.91:
        break

print(f"Final weights and biases: {model.linear.weight.data}, {model.linear.bias.data}")
print(f"Final Loss: {model.loss(x_train, y_train).item()}, Final Accuracy: {model.accuracy(x_test, y_test).item() * 100}%")


# Visualize and save the weights W after training
weights = model.linear.weight.data

for i in range(10):
    weight_image = weights[i, :].reshape(28, 28)
    plt.imshow(weight_image, cmap='viridis')
    plt.title(f'Weights for digit {i}')
    plt.colorbar()
    plt.savefig(f'weight_digit_{i}.png')
    plt.close()