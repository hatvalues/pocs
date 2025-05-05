# from app.autoencoder_train import my_autoencoder
from app.autoencoder import CNN #, DynAutoencoder
import torch.nn as nn


# def test_autoencoder_init():
#     type(my_autoencoder) == nn.Module
#     type(my_autoencoder) == DynAutoencoder
#     assert my_autoencoder is not None
#     assert my_autoencoder.encoder is not None
#     assert my_autoencoder.decoder is not None


def test_CNN_init():
    model = CNN(
        input_channels=3,
        n_conv_filters=[16, 32],
        conv_kernel_sizes=[3, 3],
        conv_strides=[1, 1],
        activations=["relu", "relu"],
        max_pool_kernels=[2, 2],
        max_pool_strides=[2, 2],
        max_pool_paddings=[0, 0],
    )
    assert model.n_conv_filters == [3, 16, 32]
    assert model.conv_kernel_sizes == [3, 3]
    assert model.conv_strides == [1, 1]
    assert model.activations == ["relu", "relu"]
    assert model.max_pool_kernels == [2, 2]
    assert model.max_pool_strides == [2, 2]
    assert model.max_pool_paddings == [0, 0]
    # for i in range(len(model.n_conv_filters) - 1):
    #     assert isinstance(model[i], nn.Sequential)
    #     assert len(model[i]) == 3
    #     assert isinstance(model[i][0], nn.Conv2d)
    #     assert model[i][0].in_channels == model.n_conv_filters[i]
    #     assert model[i][0].out_channels == model.n_conv_filters[i + 1]
    #     assert model[i][0].kernel_size == (
    #         model.conv_kernel_sizes[i],
    #         model.conv_kernel_sizes[i],
    #     )
    #     assert model[i][0].stride == (model.conv_strides[i], model.conv_strides[i])
    #     assert model[i][0].padding == (
    #         model.max_pool_paddings[i],
    #         model.max_pool_paddings[i],
    #     )
    #     assert isinstance(model[i][1], nn.ReLU)
    #     assert isinstance(model[i][2], nn.MaxPool2d)
    #     assert model[i][2].kernel_size == (
    #         model.max_pool_kernels[i],
    #         model.max_pool_kernels[i],
    #     )
    #     assert model[i][2].stride == (
    #         model.max_pool_strides[i],
    #         model.max_pool_strides[i],
    #     )
    #     assert model[i][2].padding == (
    #         model.max_pool_paddings[i],
    #         model.max_pool_paddings[i],
    #     )
    #     assert model[i][2].return_indices == False
