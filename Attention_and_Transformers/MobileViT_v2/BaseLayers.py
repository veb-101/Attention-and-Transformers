from typing import Union
from .utils import make_divisible

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, DepthwiseConv2D


class ConvLayer(Layer):
    def __init__(
        self,
        num_filters: int = 16,
        kernel_size: int = 3,
        strides: int = 2,
        use_activation: bool = True,
        use_bn: bool = True,
        use_bias: bool = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.num_filters = num_filters
        self.use_bn = use_bn
        self.use_activation = use_activation
        self.use_bias = use_bias if use_bias is not None else (False if self.use_bn else True)

        self.conv_layer = Sequential(name="Conv_layer")

        self.conv_layer.add(
            Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, strides=self.strides, padding="same", use_bias=self.use_bias)
        )

        if self.use_bn:
            self.conv_layer.add(BatchNormalization())

        if self.use_activation:
            self.conv_layer.add(Activation("swish"))

    def call(self, x, **kwargs):
        return self.conv_layer(x, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "num_filters": self.num_filters,
                "use_bias": self.use_bias,
                "use_activation": self.use_activation,
                "use_bn": self.use_bn,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Code taken from: https://github.com/veb-101/Training-Mobilenets-From-Scratch/blob/main/mobilenet_v2.py
class InvertedResidualBlock(Layer):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        depthwise_stride: int = 1,
        expansion_factor: Union[int, float] = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Input Parameters

        self.num_in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_stride = depthwise_stride
        self.expansion_factor = expansion_factor

        self.num_out_channels = int(make_divisible(self.out_channels, divisor=8))
        self.expansion_channels = int(make_divisible(self.expansion_factor * self.num_in_channels))

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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.num_in_channels,
                "out_channels": self.out_channels,
                "depthwise_stride": self.depthwise_stride,
                "expansion_factor": self.expansion_factor,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
