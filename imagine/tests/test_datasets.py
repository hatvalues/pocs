from app.datasets import CustomImageDataset
import os
import pytest
import torch
from torchvision import transforms
from torchvision.io import read_image
from unittest.mock import patch, MagicMock


# Mocking the os.listdir function to return a controlled list of files
@patch("os.listdir", return_value=["image1.jpg", "image2.jpg"])
# Mocking the read_image function to return a dummy tensor
@patch("src.datasets.read_image", return_value=torch.zeros(3, 224, 224))
def test_custom_image_dataset(mock_listdir, mock_read_image):
    # Setup
    image_directory = "mock_directory"
    dataset = CustomImageDataset(image_directory=image_directory)

    # Test if the dataset is initialized correctly
    assert len(dataset) == 1
    assert dataset.image_directory == image_directory
    assert dataset.image_file_names == ["image1.jpg", "image2.jpg"]
    assert dataset.img_labels == [0]
    assert dataset.transform is None
    assert dataset.target_transform is None
    # Test if the __getitem__ method works correctly
    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert label == 0
    # Test if the transform and target_transform are applied correctly
    transform = transforms.Resize((128, 128))
    dataset_with_transform = CustomImageDataset(
        image_directory=image_directory, transform=transform
    )
    image, label = dataset_with_transform[0]
    assert image.shape == (3, 128, 128)
    # Test if the target_transform is applied correctly
    target_transform = transforms.Lambda(lambda x: x + 1)
    dataset_with_target_transform = CustomImageDataset(
        image_directory=image_directory, target_transform=target_transform
    )
    image, label = dataset_with_target_transform[0]
    assert label == 1
    # Test if the transform and target_transform are applied correctly together
    dataset_with_both_transforms = CustomImageDataset(
        image_directory=image_directory,
        transform=transform,
        target_transform=target_transform,
    )
    image, label = dataset_with_both_transforms[0]
    assert image.shape == (3, 128, 128)
    assert label == 1
    # Test if the dataset can handle an empty directory
    empty_directory = "empty_directory"
    with patch("os.listdir", return_value=[]):
        empty_dataset = CustomImageDataset(image_directory=empty_directory)
        assert len(empty_dataset) == 1
        assert empty_dataset.image_file_names == []
        assert empty_dataset.img_labels == [0]
        with pytest.raises(IndexError):
            empty_dataset[0]
