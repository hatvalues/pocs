import torch
import torch.nn as nn
import torch.nn.functional as F

activations = nn.ModuleDict(
    [
        ["sig", nn.Sigmoid()],
        ["tanh", nn.Tanh()],
        ["lrelu", nn.LeakyReLU()],
        ["relu", nn.ReLU()],
        ["elu", nn.ELU()],
    ]
)

# using nn.Sequential
hidden1_dim = 60
hidden2_dim = 40
hidden3_dim = 20


def add_layer(input_dim, output_dim, activation="relu"):
    return nn.Sequential(nn.Linear(input_dim, output_dim), activations[activation])


def conv_block(
    in_conv_channels: int,
    out_conv_channels: int,
    conv_kernel: int = 4,
    conv_stride: int = 1,
    conv_padding: int = 0,
    activation: str = "relu",
    max_pool_kernel: int = 2,
    max_pool_stride: int = 1,
    max_pool_padding: int = 0,
    *args,
    **kwargs,
):
    return nn.Sequential(
        nn.Conv2d(
            in_conv_channels,
            out_conv_channels,
            conv_kernel,
            conv_stride,
            conv_padding,
            *args,
            **kwargs,
        ),
        activations[activation],
        nn.MaxPool2d(
            kernel_size=max_pool_kernel,
            stride=max_pool_stride,
            padding=max_pool_padding,
        ),
    )


class CNN(nn.Module):
    def __init__(
        self,
        input_channels,
        n_conv_filters,
        conv_kernel_sizes,
        conv_strides,
        activations,
        max_pool_kernels,
        max_pool_strides,
        max_pool_paddings,
    ):
        super(CNN, self).__init__()
        self.n_conv_filters = [input_channels, *n_conv_filters]
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.activations = activations
        self.max_pool_kernels = max_pool_kernels
        self.max_pool_strides = max_pool_strides
        self.max_pool_paddings = max_pool_paddings

        conv_blocks = [
            conv_block(
                in_conv_channels,
                out_conv_channels,
                conv_kernel_size,
                conv_stride,
                conv_padding,
                activation,
                max_pool_kernel,
                max_pool_stride,
                max_pool_padding,
            )
            for in_conv_channels, out_conv_channels, conv_kernel_size, conv_stride, conv_padding, activation, max_pool_kernel, max_pool_stride, max_pool_padding in zip(
                self.n_conv_filters,
                self.n_conv_filters[1:],
                self.conv_kernel_sizes,
                self.conv_strides,
                self.conv_strides,
                self.activations,
                self.max_pool_kernels,
                self.max_pool_strides,
                self.max_pool_paddings,
            )
        ]

        self.encoder = nn.Sequential(*conv_blocks)

        self.decoder = nn.Sequential(
            # * 4 * 4 is from two lots of max pooling kernel size 2
            # base value should match last channel size
            nn.Linear(self.n_conv_filters[-1], out_features=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# net = Net()


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
