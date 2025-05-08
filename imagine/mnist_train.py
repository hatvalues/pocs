from app.fixed_models.cnn_classifier import OneTwentyEightClassifier
from app.mnist_demo_helpers import (
    dev,
    batch_size,
    epochs,
    channels,
    height,
    width,
    train_dataset,
    train_loader,
    criterion,
)
from torch import optim, save as torch_save
from datetime import datetime

net = OneTwentyEightClassifier().to(dev)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

print("Image shape:", channels, height, width)
n_pixels = channels * height * width
print("Number of pixels:", n_pixels)
print("Autoencoder model initialized.")
print("Training on device:", dev)
print("Training data size:", len(train_dataset))
print("Batch size:", batch_size)
print("Number of batches in train loader:", len(train_loader))
print("Autoencoder architecture:")
print(net)
print("Loss function:", criterion)
print("Optimizer:", optimizer)
print("Starting training...")

# # Training loop
for epoch in range(epochs):
    net.train()
    train_loss = 0
    for batch, (data, labels) in enumerate(train_loader):
        data, labels = data.to(dev), labels.to(dev)
        optimizer.zero_grad()
        outputs = net(data)
        loss = criterion(outputs, labels)
        loss.backward()
        if batch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Batch: {batch}, Batch Loss: {loss.item():.4f}"
            )
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

# # Save the trained model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
torch_save(net.state_dict(), f"models/onetwentyeight_{timestamp}.pth")
