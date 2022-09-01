"""
The implementation is based on https://keras.io/examples/vision/deeplabv3_plus/ with minor modifications
Note that original DeepLabV3+ uses ResNet-101 or Xception as encoders, not ResNet-50
"""
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.applications as applications
from dlmodels.unet import conv


def conv_block(x, n_filters=256, kernel_size=3, dilation_rate=1, batch_norm=True):
    y = conv(n_filters, kernel_size, dilation_rate)(x)
    if batch_norm:
        y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU()(y)
    return y


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(dspp_input)
    x = conv_block(x, kernel_size=1)
    out_pool = layers.UpSampling2D(
        size=(dims[1] // x.shape[1], dims[2] // x.shape[2]), 
        interpolation="bilinear",
    )(x)

    out_1 = conv_block(dspp_input, kernel_size=1)
    out_6 = conv_block(dspp_input, dilation_rate=6)
    out_12 = conv_block(dspp_input, dilation_rate=12)
    out_18 = conv_block(dspp_input, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = conv_block(x, kernel_size=1)
    return output


def deeplab(input_shapes, n_classes):
    inputs = []
    for input_name, input_shape in input_shapes.items():
        input_layer = layers.Input(input_shape, name=input_name)
        inputs.append(input_layer)

    conv = []
    for input_layer in inputs:
        conv_ = conv_block(input_layer, 32)
        conv.append(conv_)
    conv = layers.concatenate(conv)
    dims = conv.shape

    resnet50 = applications.ResNet50(
        weights=None, include_top=False, input_tensor=conv
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(dims[1] // 4 // x.shape[1], dims[2] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)

    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = conv_block(input_b, n_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = conv_block(x)
    x = conv_block(x)
    x = layers.UpSampling2D(
        size=(dims[1] // x.shape[1], dims[2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model
