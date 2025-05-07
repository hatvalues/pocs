# from app.autoencoder_train import my_autoencoder
# from app.model_builder import CNN #, DynAutoencoder


# def test_autoencoder_init():
#     type(my_autoencoder) == nn.Module
#     type(my_autoencoder) == DynAutoencoder
#     assert my_autoencoder is not None
#     assert my_autoencoder.encoder is not None
#     assert my_autoencoder.decoder is not None


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
#         model.max_pool_kernel_sizes[i],
#         model.max_pool_kernel_sizes[i],
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
