from app.model_builder import CNN
from tests.helpers import assert_text_matches_fixture
import torch
import pytest


def test_CNN_init():
    model = CNN(
        input_channels=3,
        n_conv_filters=[16, 32],
        conv_kernel_sizes=[3, 5],
        conv_strides=[1, 2],
        conv_paddings=[0, 1],
        conv_dilations=[1, 1],
        activations=["relu", "relu"],
        max_pool_kernel_sizes=[2, 3],
        max_pool_strides=[2, 1],
        max_pool_paddings=[0, 1],
        max_pool_dilations=[1, 1],
    )
    assert model.n_conv_filters == [3, 16, 32]
    assert model.conv_kernel_sizes == [3, 5]
    assert model.conv_strides == [1, 2]
    assert model.conv_paddings == [0, 1]
    assert model.conv_dilations == [1, 1]
    assert model.activations == ["relu", "relu"]
    assert model.max_pool_kernel_sizes == [2, 3]
    assert model.max_pool_strides == [2, 1]
    assert model.max_pool_paddings == [0, 1]
    assert model.max_pool_dilations == [1, 1]
    assert_text_matches_fixture("cnn_init", str(model))
    assert model.conv_layers_sizing(28) == ([26, 6], [13, 6])
    assert model.conv_layers_sizing(32) == ([30, 7], [15, 7])
    assert model.conv_layers_sizing(1024) == ([1022, 255], [511, 255])

    model.add_linear_layer(
        initial_input_dim=1024,
        output_dim=10,
        activation="relu",
    )
    assert_text_matches_fixture("cnn_init_add_layer", str(model))

@pytest.mark.parametrize(
    "batch_size,input_channels,input_dim,n_conv_filters,max_pool_factor",
    [
        (4, 3, 28, [16], 2),
        (4, 3, 32, [16], 2),
    ],
)
def test_CNN(batch_size, input_channels, input_dim, n_conv_filters, max_pool_factor):
    model = CNN(
        input_channels=input_channels,
        n_conv_filters=n_conv_filters,
        conv_kernel_sizes=[5],
        conv_strides=[1],
        conv_paddings=[2],
        conv_dilations=[1],
        activations=["relu", "relu"],
        max_pool_kernel_sizes=[max_pool_factor],
        max_pool_strides=[max_pool_factor],
        max_pool_paddings=[0],
        max_pool_dilations=[1],
    )

    assert model.conv_layers_sizing(input_dim) == ([input_dim], [input_dim // max_pool_factor])
    x = torch.randn(batch_size, input_channels, input_dim // max_pool_factor, input_dim // max_pool_factor)
    y = model(x)
    assert y.shape == torch.Size([batch_size, n_conv_filters[-1], input_dim // (2 * max_pool_factor), input_dim // (2 * max_pool_factor)])
