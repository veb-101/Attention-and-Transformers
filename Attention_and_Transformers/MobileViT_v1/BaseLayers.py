import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, DepthwiseConv2D

# https://www.tensorflow.org/guide/mixed_precision#ensuring_gpu_tensor_cores_are_used
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvLayer(Layer):
    def __init__(self, num_filters=16, kernel_size=3, strides=2, use_activation=True, use_bn=True, use_bias=None, **kwargs):
        super().__init__(**kwargs)

        self.conv_layer = Sequential(name="Conv_layer")

        use_bias = use_bias if use_bias is not None else (False if use_bn else True)

        self.conv_layer.add(Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding="same", use_bias=use_bias))

        if use_bn:
            self.conv_layer.add(BatchNormalization())

        if use_activation:
            self.conv_layer.add(Activation("swish"))

    def call(self, x, **kwargs):
        return self.conv_layer(x, **kwargs)


# Code taken from: https://github.com/veb-101/Training-Mobilenets-From-Scratch/blob/main/mobilenet_v2.py
class InvertedResidualBlock(Layer):
    def __init__(
        self,
        in_channels=32,
        out_channels=64,
        depthwise_stride=1,
        expansion_factor=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Input Parameters

        self.num_in_channels = in_channels
        self.num_out_channels = int(_make_divisible(out_channels, divisor=8))
        self.expansion_channels = int(_make_divisible(expansion_factor * self.num_in_channels))

        self.depthwise_stride = depthwise_stride

        # Layer Attributes
        self.apply_expansion = self.expansion_channels > self.num_in_channels
        self.residual_connection = True if (self.num_in_channels == self.num_out_channels) and (self.depthwise_stride == 1) else False

        # Layers
        self.sequential_block = Sequential()

        if self.apply_expansion:
            self.sequential_block.add(ConvLayer(num_filters=self.expansion_channels, kernel_size=1, strides=1, use_activation=True, use_bn=True))

        self.sequential_block.add(DepthwiseConv2D(kernel_size=3, strides=self.depthwise_stride, padding="same", use_bias=False))
        self.sequential_block.add(BatchNormalization())
        self.sequential_block.add(Activation("swish"))

        self.sequential_block.add(ConvLayer(num_filters=self.num_out_channels, kernel_size=1, strides=1, use_activation=False, use_bn=True))

    def call(self, data, **kwargs):

        out = self.sequential_block(data)

        if self.residual_connection:
            out = out + data

        return out
