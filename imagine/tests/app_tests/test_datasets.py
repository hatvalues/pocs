from app.datasets import CustomImageDataset
import os
import pytest # type: ignore
import torch
import numpy as np
from PIL import Image
from torchvision import transforms # type: ignore
from unittest.mock import patch
from pytest_unordered import unordered


# Mocking the os.listdir function to return a controlled list of files
@patch("os.listdir", return_value=["image1.jpg", "image2.jpg"])
# Mocking the read_image function to return a dummy tensor
@patch("app.datasets.read_image", return_value=torch.zeros(3, 224, 224))
def test_custom_image_dataset_with_mock(mock_listdir, mock_read_image):
    # Setup
    image_directory = "mock_directory"
    dataset = CustomImageDataset(image_directory=image_directory)

    # Test if the dataset is initialized correctly
    assert dataset.image_directory == image_directory
    assert dataset.image_file_names == ["image1.jpg", "image2.jpg"]
    assert dataset.image_labels == [0]
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
        assert len(empty_dataset) == 0
        assert empty_dataset.image_file_names == []
        assert empty_dataset.image_labels == [0]
        with pytest.raises(IndexError):
            empty_dataset[0]


def test_custom_image_dataset():
    # Create a temporary directory for testing
    image_dir = "images"
    n_files = len(os.listdir(image_dir))
    temp_dir = "temp_test_directory"
    os.makedirs(temp_dir, exist_ok=True)

    # copy the image directory to the temporary directory
    for fn, filename in enumerate(os.listdir(image_dir)):
        src_file = os.path.join(image_dir, filename)
        dst_file = os.path.join(temp_dir, filename)
        if os.path.isfile(src_file):
            with open(src_file, "rb") as fsrc:
                with open(dst_file, "wb") as fdst:
                    fdst.write(fsrc.read())

    # Create some dummy image files
    for i in range(n_files):
        # Create a random image array with dummy data
        # Using 3 channels (RGB) with size square root of the file size / channels

        # Generate random values between 0-255 for RGB channels
        dummy_data = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        image = Image.fromarray(dummy_data, "RGB")
        dummy_image = os.path.join(temp_dir, f"image{i}.jpg")
        image.save(dummy_image, format="JPEG")

    # Test the CustomImageDataset class
    dataset = CustomImageDataset(image_directory=temp_dir)
    print(len(dataset))
    print(dataset.image_directory)
    print(dataset.image_file_names)
    for i in range(4):
        print(f"Image {i}:")
        print(dataset[i][0].shape)
        print(dataset[i][1])
        print(dataset[i][0].dtype)
        print(dataset[i][0].device)
        print(dataset.image_file_names[i])
        print(dataset[i])

    # Clean up the temporary directory before assertions and potentionally failing tests
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(temp_dir)

    assert len(dataset) == n_files * 2
    assert len(dataset.image_file_names) == n_files * 2
    assert dataset.image_directory == temp_dir
    assert [
        fn for fn in dataset.image_file_names if fn.startswith("image")
    ] == unordered([f"image{fn}.jpg" for fn in range(n_files)])
    assert dataset.image_labels == [0]
