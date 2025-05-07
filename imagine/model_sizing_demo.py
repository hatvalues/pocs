from app.model_sizing import (
    conv2d_output_size,
    maxpool2d_output_size,
    size_preserving_padding,
)

# Example usage
if __name__ == "__main__":
    # Test with some common configurations
    input_size = 32
    print("Input size:", input_size)

    # Conv2d examples
    print("Conv2d Output Size Examples:")
    print(
        f"Standard conv (k=3, s=1, p=1, d=1): {conv2d_output_size(input_size, 3, 1, 1, 1)}"
    )
    print(
        f"Strided conv (k=3, s=2, p=1, d=1): {conv2d_output_size(input_size, 3, 2, 1, 1)}"
    )
    print(
        f"Dilated conv (k=3, s=1, p=2, d=2): {conv2d_output_size(input_size, 3, 1, 2, 2)}"
    )
    print(
        f"No padding (k=3, s=1, p=0, d=1): {conv2d_output_size(input_size, 3, 1, 0, 1)}"
    )

    # MaxPool2d examples
    print("\nMaxPool2d Output Size Examples:")
    print(
        f"Standard pool (k=2, s=2, p=0, d=1): {maxpool2d_output_size(input_size, 2, 2, 0, 1)}"
    )
    print(
        f"Overlapping pool (k=3, s=2, p=0, d=1): {maxpool2d_output_size(input_size, 3, 2, 0, 1)}"
    )
    print(
        f"Non-strided pool (k=2, s=1, p=0, d=1): {maxpool2d_output_size(input_size, 2, 1, 0, 1)}"
    )
    print(
        f"Dilated pool (k=2, s=2, p=0, d=2): {maxpool2d_output_size(input_size, 2, 2, 0, 2)}"
    )

    print("\nRequired padding for 'same' output size:")
    for kernel in [1, 3, 5, 7]:
        for dilation in [1, 2, 3]:
            padding = size_preserving_padding(kernel, dilation)
            output = conv2d_output_size(input_size, kernel, 1, padding, dilation)
            print(f"k={kernel}, d={dilation}: padding={padding}, output={output}")
