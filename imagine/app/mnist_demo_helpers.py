from app.training_helpers import channel_based_normaliser
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch import device, cuda

batch_size = 64
epochs = 20

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, transform=None, download=True
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, transform=None, download=True
)

first_image = train_dataset[0][0]
height, width = first_image.size
channels = 0
for i in range(3):
    try:
        first_image.getchannel(i)
        channels += 1
    except ValueError:
        break

transform = channel_based_normaliser(channels)
train_dataset.transform = transform
test_dataset.transform = transform

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

dev = device("cuda" if cuda.is_available() else "cpu")
criterion = CrossEntropyLoss()
