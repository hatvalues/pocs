from torchvision import transforms


def get_image_shape(image):
    """
    Get the shape of the image.

    Args:
        image: The image to get the shape of.

    Returns:
        A tuple containing the number of channels, height, and width of the image.
    """
    return image.shape


def channel_based_normaliser(n_channels: int):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * n_channels, (0.5,) * n_channels),
        ]
    )
