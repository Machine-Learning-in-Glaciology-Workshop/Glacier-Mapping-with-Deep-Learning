"""
The implementation is based on https://keras.io/examples/vision/deeplabv3_plus/ with minor modifications
Note that original DeepLabV3+ uses ResNet-101 or Xception as encoders, not ResNet-50
"""
import tensorflow as tf
import layers.general


def DilatedSpatialPyramidPooling(dspp_input, dropout=0):
    dims = dspp_input.shape
    x = tf.keras.layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(dspp_input)
    x = layers.general.ConvBatchNormAct(n_filters=128, kernel_size=1)(x)
    out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[1] // x.shape[1], dims[2] // x.shape[2]), 
        interpolation="bilinear",
    )(x)

    out_1 = layers.general.ConvBatchNormAct(n_filters=128, kernel_size=1)(dspp_input)
    out_6 = layers.general.ConvBatchNormAct(n_filters=128, dilation_rate=6)(dspp_input)
    out_12 = layers.general.ConvBatchNormAct(n_filters=128, dilation_rate=12)(dspp_input)
    out_18 = layers.general.ConvBatchNormAct(n_filters=128, dilation_rate=18)(dspp_input)

    x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = layers.general.ConvBatchNormAct(n_filters=128, kernel_size=1, spatial_dropout=dropout)(x)
    return output


def DeepLabMini(
    input_shape, n_outputs, last_activation="softmax", dropout=0,
    name="DeepLabMini", **kwargs
):
    inputs = tf.keras.layers.Input(input_shape, name="features")
    dims = inputs.shape

    resnet50 = tf.keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=inputs
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x, dropout=dropout)

    input_a = tf.keras.layers.UpSampling2D(
        size=(dims[1] // 4 // x.shape[1], dims[2] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)

    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = layers.general.ConvBatchNormAct(n_filters=48, kernel_size=1, spatial_dropout=dropout)(input_b)

    x = tf.keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = layers.general.ConvBatchNormAct_x2(n_filters=64, spatial_dropout=dropout)(x)
    x = tf.keras.layers.UpSampling2D(
        size=(dims[1] // x.shape[1], dims[2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    outputs = tf.keras.layers.Conv2D(n_outputs, 1, activation=last_activation)(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name, **kwargs)
    return model
