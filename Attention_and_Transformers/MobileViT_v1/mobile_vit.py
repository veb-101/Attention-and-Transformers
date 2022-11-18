from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

from .BaseLayers import ConvLayer, InvertedResidualBlock
from .mobile_vit_block import MobileViTBlock


def MobileViT(
    out_channels: list,
    expansion_factor: int,
    tf_repeats: list,
    tf_embedding_dims: list,
    linear_drop: float = 0.0,
    attention_drop: float = 0.2,
    num_classes: int = 1000,
    input_shape: tuple = (256, 256, 3),
    model_type: str = "S",
):

    """
    Arguments
    --------
        out_channel: (list)  Output channels of each layer

        expansion_factor: (int)   Inverted residual block -> Bottelneck expansion size

        tf_repeats: (list)  Number of time to repeat each transformer block

        tf_embedding_dims: (list)  Embedding dimension used in each transformer block

        num_classes: (int)   Number of output classes

        input_shape: (tuple) Input shape -> H, W, C

        model_type: (str)   Model to create

        linear_drop: (float) Dropout rate for Dense layers

        attention_drop: (float) Dropout rate for the attention matrix

    """

    input_layer = Input(shape=input_shape)

    # Block 1
    out = ConvLayer(num_filters=out_channels[0], kernel_size=3, strides=2)(input_layer)

    out = InvertedResidualBlock(
        in_channels=out_channels[0],
        out_channels=out_channels[1],
        depthwise_stride=1,
        expansion_factor=expansion_factor,
        name="block-1-IR1",
    )(out)

    # Block 2
    out = InvertedResidualBlock(
        in_channels=out_channels[1],
        out_channels=out_channels[2],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-2-IR1",
    )(out)

    out = InvertedResidualBlock(
        in_channels=out_channels[2],
        out_channels=out_channels[3],
        depthwise_stride=1,
        expansion_factor=expansion_factor,
        name="block-2-IR2",
    )(out)

    out = InvertedResidualBlock(
        in_channels=out_channels[2],
        out_channels=out_channels[3],
        depthwise_stride=1,
        expansion_factor=expansion_factor,
        name="block-2-IR3",
    )(out)

    # Block 3
    out = InvertedResidualBlock(
        in_channels=out_channels[3],
        out_channels=out_channels[4],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-3-IR1",
    )(out)

    out = MobileViTBlock(
        out_filters=out_channels[5],
        embedding_dim=tf_embedding_dims[0],
        transformer_repeats=tf_repeats[0],
        name="MobileViTBlock-1",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 4
    out = InvertedResidualBlock(
        in_channels=out_channels[5],
        out_channels=out_channels[6],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-4-IR1",
    )(out)

    out = MobileViTBlock(
        out_filters=out_channels[7],
        embedding_dim=tf_embedding_dims[1],
        transformer_repeats=tf_repeats[1],
        name="MobileViTBlock-2",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 5
    out = InvertedResidualBlock(
        in_channels=out_channels[7],
        out_channels=out_channels[8],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-5-IR1",
    )(out)

    out = MobileViTBlock(
        out_filters=out_channels[9],
        embedding_dim=tf_embedding_dims[2],
        transformer_repeats=tf_repeats[2],
        name="MobileViTBlock-3",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    out = ConvLayer(num_filters=out_channels[10], kernel_size=1, strides=1)(out)

    # Output layer
    out = GlobalAveragePooling2D()(out)

    if linear_drop > 0.0:
        out = Dropout(rate=linear_drop)(out)

    out = Dense(units=num_classes)(out)

    model = Model(inputs=input_layer, outputs=out, name=f"MobileViT-{model_type}")

    return model


@dataclass(frozen=True)
class config_MobileViT_S:
    out_channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    expansion_factor = 4
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [144, 192, 240]


@dataclass(frozen=True)
class config_MobileViT_XS:
    out_channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    expansion_factor = 4
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [144, 192, 240]


@dataclass(frozen=True)
class config_MobileViT_XXS:
    out_channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    expansion_factor = 4
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [144, 192, 240]


def build_MobileVit_V1(model_type: str = "S", num_classes: int = 1000, input_shape: tuple = (None, None, 3), **kwargs):

    """
    Create MobileVit-V1 Classification models

    Arguments
    --------
        model_type: (str)   MobileVit version to create. Options: S, XS, XSS

        num_classes: (int)   Number of output classes

        input_shape: (tuple) Input shape -> H, W, C

    Additional arguments:
    ---------------------

        linear_drop: (float) Dropout rate for Dense layers

        attention_drop: (float) Dropout rate for the attention matrix

    """

    if model_type not in ["S", "XS", "XSS"]:
        raise ValueError("Bad Input. 'model_type' should one of ['S', 'XS', 'XXS']")

    if model_type == "S":
        config = config_MobileViT_S()
    elif model_type == "S":
        config = config_MobileViT_XS()
    else:
        config = config_MobileViT_XXS()

    model = MobileViT(
        out_channels=config.out_channels,  # (list)  Output channels of each layer
        expansion_factor=config.expansion_factor,  # (int)   Inverted residual block -> Bottelneck expansion size
        tf_repeats=config.tf_repeats,  # (list)  Number of time to repeat each transformer block
        tf_embedding_dims=config.tf_embedding_dims,  # (list)  Embedding dimension used in each transformer block
        num_classes=num_classes,  # (int)   Number of output classes
        input_shape=input_shape,  # (tuple) Input shape -> H, W, C
        model_type=model_type,  # (str)   Model to create
        **kwargs,
    )

    return model


if __name__ == "__main__":

    model = build_MobileVit_V1(
        model_type="S",  # "XS", "XXS"
        input_shape=(256, 256, 3),  # (None, None, 3)
        num_classes=1000,
        # linear_drop=0.2,
        # attention_drop=0.2,
    )

    model.summary()
