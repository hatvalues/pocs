import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomImageDataset(Dataset):
    def __init__(self, image_directory, transform=None, target_transform=None):
        self.img_labels = [0]  # not using for classification at the moment
        self.image_directory = image_directory
        self.image_file_names = os.listdir(self.image_directory)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    # fix this error:
    #     src/datasets.py:18: in __getitem__
    #     img_path = os.path.join(self.image_directory, self.img_labels[idx])
    # <frozen posixpath>:90: in join
    #     ???
    # _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

    # funcname = 'join', args = ('mock_directory', 0), hasstr = True, hasbytes = False, s = 0

    # >   ???
    # E   TypeError: join() argument must be str, bytes, or os.PathLike object, not 'int'

    # <frozen genericpath>:164: TypeError

    def __getitem__(self, idx):
        if idx >= len(self.img_labels):
            raise IndexError("Index out of range")
        if idx < 0:
            idx += len(self.img_labels)
        img_path = os.path.join(self.image_directory, self.image_file_names[idx])
        image = read_image(img_path)
        label = self.img_labels[0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
