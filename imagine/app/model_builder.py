import torch.nn as nn
from app.model_sizing import (
    conv2d_output_size,
    maxpool2d_output_size,
    size_preserving_padding,
)

activations_lookup = nn.ModuleDict(
    [
        ["sig", nn.Sigmoid()],
        ["tanh", nn.Tanh()],
        ["lrelu", nn.LeakyReLU()],
        ["relu", nn.ReLU()],
        ["elu", nn.ELU()],
    ]
)


def lin_layer(input_dim, output_dim, activation="relu"):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim), activations_lookup[activation]
    )


def conv_block(
    in_conv_channels: int,
    out_conv_channels: int,
    conv_kernel: int = 4,
    conv_stride: int = 1,
    conv_padding: int = 0,
    conv_dilation: int = 1,
    activation: str = "relu",
    max_pool_kernel: int = 2,
    max_pool_stride: int = 1,
    max_pool_padding: int = 0,
    max_pool_dilation: int = 1,
    *args,
    **kwargs,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_conv_channels,
            out_channels=out_conv_channels,
            kernel_size=conv_kernel,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            *args,
            **kwargs,
        ),
        activations_lookup[activation],
        nn.MaxPool2d(
            kernel_size=max_pool_kernel,
            stride=max_pool_stride,
            padding=max_pool_padding,
            dilation=max_pool_dilation,
        ),
    )


class CNN(nn.Module):
    def __init__(
        self,
        input_channels,
        n_conv_filters,
        conv_kernel_sizes,
        conv_strides,
        conv_paddings,
        conv_dilations,
        activations,
        max_pool_kernel_sizes,
        max_pool_strides,
        max_pool_paddings,
        max_pool_dilations,
    ):
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.n_conv_filters = [input_channels, *n_conv_filters]
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_paddings = conv_paddings
        self.conv_dilations = conv_dilations
        self.activations = activations
        self.max_pool_kernel_sizes = max_pool_kernel_sizes
        self.max_pool_strides = max_pool_strides
        self.max_pool_paddings = max_pool_paddings
        self.max_pool_dilations = max_pool_dilations

        conv_blocks = [
            conv_block(
                in_conv_channels,
                out_conv_channels,
                conv_kernel_size,
                conv_stride,
                conv_padding,
                conv_dilation,
                activation,
                max_pool_kernel,
                max_pool_stride,
                max_pool_padding,
                max_pool_dilation,
            )
            for in_conv_channels, out_conv_channels, conv_kernel_size, conv_stride, conv_padding, conv_dilation, activation, max_pool_kernel, max_pool_stride, max_pool_padding, max_pool_dilation in zip(
                self.n_conv_filters,
                self.n_conv_filters[1:],
                self.conv_kernel_sizes,
                self.conv_strides,
                self.conv_paddings,
                self.conv_dilations,
                self.activations,
                self.max_pool_kernel_sizes,
                self.max_pool_strides,
                self.max_pool_paddings,
                self.max_pool_dilations,
            )
        ]

        self.encoder = nn.Sequential(*conv_blocks)
        self.decoder = nn.Sequential(
            # * 4 * 4 is from two lots of max pooling kernel size 2
            # base value should match last channel size
            nn.Linear(self.n_conv_filters[-1], out_features=2)
        )

    def add_linear_layer(
        self,
        initial_input_dim,
        output_dim,
        activation="relu",
    ):
        _, output_sizes = self.conv_layers_sizing(initial_input_dim)
        # The last output size is the input size for the linear layer after flattening
        # because we have square images
        last_output_size = output_sizes[-1]
        input_dim = self.input_channels * last_output_size * last_output_size

        self.fully_connected = lin_layer(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
        )

        self.encoder = nn.Sequential(
            self.encoder,
            nn.Flatten(),
            self.fully_connected,
        )
        # self.decoder = nn.Sequential(
        #     self.fully_connected,
        #     nn.Unflatten(1, (self.n_conv_filters[-1], last_output_size, last_output_size)),
        #     *reversed(self.encoder),
        # )

    def conv_layers_sizing(self, input_size):
        """
        Calculate the output size of the convolutional layers.
        For a given input size, it computes the output size after each convolutional layer.
        Args:
            input_size (int): The size of the input image (assumed square).
        Returns:
            list: A list of output sizes after each convolutional layer.
        """
        conv_output_sizes = []
        maxp_output_sizes = []
        for i in range(len(self.n_conv_filters) - 1):
            conv_output_size = conv2d_output_size(
                input_size,
                self.conv_kernel_sizes[i],
                self.conv_strides[i],
                self.conv_paddings[i],
            )
            conv_output_sizes.append(conv_output_size)
            input_size = maxpool2d_output_size(
                conv_output_size,
                self.max_pool_kernel_sizes[i],
                self.max_pool_strides[i],
                self.max_pool_paddings[i],
            )
            maxp_output_sizes.append(input_size)
        return conv_output_sizes, maxp_output_sizes

    def forward(self, x):
        x = self.encoder(x)
        # x = x.view(x.size(0), -1)
        # x = self.decoder(x)
        return x


class DynAutoencoder(nn.Module):
    def __init__(self, channels: int, height: int, width: int):
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Example usage
# if __name__ == "__main__":
#     input_shape = (1, 28, 28)  # Example for grayscale images (e.g., MNIST)
#     model = Autoencoder()
#     print(model)

#     # Example input
#     example_input = torch.randn(1, *input_shape)  # Batch size of 1
#     output = model(example_input)
#     print("Output shape:", output.shape)
