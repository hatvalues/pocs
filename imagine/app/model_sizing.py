def conv2d_output_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Calculate the output size of a square image after applying a Conv2d operation.

    Args:
        input_size (int): Side length of the square input image/feature map
        kernel_size (int): Size of the square convolutional kernel
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides. Default is 0.
        dilation (int, optional): Spacing between kernel elements. Default is 1.

    Returns:
        int: Side length of the square output feature map
    """
    return ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def maxpool2d_output_size(input_size, kernel_size, stride=None, padding=0, dilation=1):
    """
    Calculate the output size of a square image after applying a MaxPool2d operation.

    Args:
        input_size (int): Side length of the square input image/feature map
        kernel_size (int): Size of the square pooling window
        stride (int, optional): Stride of the pooling operation. Default is kernel_size.
        padding (int, optional): Zero-padding added to both sides. Default is 0.
        dilation (int, optional): Spacing between kernel elements. Default is 1.

    Returns:
        int: Side length of the square output feature map
    """
    # If stride is None, set it equal to kernel_size (PyTorch default)
    if stride is None:
        stride = kernel_size

    return ((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1


# Calculate required padding for "same" output size (maintaining spatial dimensions)
def size_preserving_padding(kernel_size, dilation=1):
    """Calculate padding needed to maintain spatial dimensions with given kernel and dilation"""
    return dilation * (kernel_size - 1) // 2
