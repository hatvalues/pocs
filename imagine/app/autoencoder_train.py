import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from autoencoder import DynAutoencoder
from datetime import datetime
# This script trains an autoencoder on the Fashion MNIST dataset.

# Normalize does the following for each channel:
# image = (image - mean) / std
# The parameters mean, std are passed as 0.5, 0.5 in your case.
# This will normalize the image in the range [-1,1]. For example,
# the minimum value 0 will be converted to (0-0.5)/0.5=-1,
# the maximum value of 1 will be converted to (1-0.5)/0.5=1.
# if you would like to get your image back in [0,1] range, you could use,
# image = ((image * std) + mean)


train_dataset = datasets.FashionMNIST(
    root="./data", train=True, transform=None, download=True
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, transform=None, download=True
)

channels, height, width = train_dataset[0][0].shape
print("Image shape:", channels, height, width)
n_pixels = channels * height * width
print("Number of pixels:", n_pixels)


def channel_based_normaliser(n_channels: int):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * n_channels, (0.5,) * n_channels),
        ]
    )
    return transform


transform = channel_based_normaliser(channels)
train_dataset.transform = transform
test_dataset.transform = transform


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Instantiate the autoencoder model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_autoencoder = DynAutoencoder(channels=channels, height=height, width=width).to(device)

# # Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(my_autoencoder.parameters(), lr=1e-3)

print("Autoencoder model initialized.")
print("Training on device:", device)
print("Training data size:", len(train_dataset))
print("Test data size:", len(test_dataset))
print("Batch size:", 64)
print("Number of batches in train loader:", len(train_loader))
print("Number of batches in test loader:", len(test_loader))
print("Autoencoder architecture:")
print(my_autoencoder)
print("Loss function:", criterion)
print("Optimizer:", optimizer)
print("Starting training...")
# # Training loop
epochs = 20
for epoch in range(epochs):
    my_autoencoder.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = my_autoencoder(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

# # Save the trained model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save(my_autoencoder.state_dict(), f"models/autoencoder_{timestamp}.pth")
