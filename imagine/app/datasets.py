import os
from torch.utils.data import Dataset
from torchvision.io import read_image # type: ignore


class CustomImageDataset(Dataset):
    def __init__(self, image_directory, transform=None, target_transform=None):
        self.image_labels = [0]  # not using for classification at the moment
        self.image_directory = image_directory
        self.image_file_names = os.listdir(self.image_directory)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_directory, self.image_file_names[idx])
        image = read_image(img_path)
        label = self.image_labels[0]  # Placeholder for actual label
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
