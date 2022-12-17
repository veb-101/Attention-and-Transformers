from dataclasses import dataclass

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

from .BaseLayers import ConvLayer, InvertedResidualBlock
from .mobile_vit_v1_block import MobileViT_v1_Block


def MobileViT_v1(
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
    out_b1_1 = ConvLayer(num_filters=out_channels[0], kernel_size=3, strides=2)(input_layer)

    out_b1_2 = InvertedResidualBlock(
        in_channels=out_channels[0],
        out_channels=out_channels[1],
        depthwise_stride=1,
        expansion_factor=expansion_factor,
        name="block-1-IR1",
    )(out_b1_1)

    if out_b1_1.shape[-1] == out_b1_2.shape[-1]:
        out = out_b1_1 + out_b1_2
    else:
        out = out_b1_2

    # Block 2
    out_b2_1 = InvertedResidualBlock(
        in_channels=out_channels[1],
        out_channels=out_channels[2],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-2-IR1",
    )(out)

    out_b2_2 = InvertedResidualBlock(
        in_channels=out_channels[2],
        out_channels=out_channels[3],
        depthwise_stride=1,
        expansion_factor=expansion_factor,
        name="block-2-IR2",
    )(out_b2_1)

    out = out_b2_1 + out_b2_2

    out_b2_3 = InvertedResidualBlock(
        in_channels=out_channels[3],
        out_channels=out_channels[4],
        depthwise_stride=1,
        expansion_factor=expansion_factor,
        name="block-2-IR3",
    )(out)

    out = out + out_b2_3

    # Block 3
    out = InvertedResidualBlock(
        in_channels=out_channels[4],
        out_channels=out_channels[5],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-3-IR1",
    )(out)

    out = MobileViT_v1_Block(
        out_filters=out_channels[6],
        embedding_dim=tf_embedding_dims[0],
        transformer_repeats=tf_repeats[0],
        name="MobileViTBlock-1",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 4
    out = InvertedResidualBlock(
        in_channels=out_channels[6],
        out_channels=out_channels[7],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-4-IR1",
    )(out)

    out = MobileViT_v1_Block(
        out_filters=out_channels[8],
        embedding_dim=tf_embedding_dims[1],
        transformer_repeats=tf_repeats[1],
        name="MobileViTBlock-2",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 5
    out = InvertedResidualBlock(
        in_channels=out_channels[8],
        out_channels=out_channels[9],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-5-IR1",
    )(out)

    out = MobileViT_v1_Block(
        out_filters=out_channels[10],
        embedding_dim=tf_embedding_dims[2],
        transformer_repeats=tf_repeats[2],
        name="MobileViTBlock-3",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    out = ConvLayer(num_filters=out_channels[11], kernel_size=1, strides=1)(out)

    # Output layer
    out = GlobalAveragePooling2D()(out)

    if linear_drop > 0.0:
        out = Dropout(rate=linear_drop)(out)

    out = Dense(units=num_classes)(out)

    model = Model(inputs=input_layer, outputs=out, name=f"MobileViT_v1-{model_type}")

    return model


@dataclass(frozen=True)
class config_MobileViT_v1_S:
    out_channels = [16, 32, 64, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    depthwise_expansion_factor = 4
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [144, 192, 240]


@dataclass(frozen=True)
class config_MobileViT_v1_XS:
    out_channels = [16, 32, 48, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    depthwise_expansion_factor = 4
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [96, 120, 144]


@dataclass(frozen=True)
class config_MobileViT_v1_XXS:
    out_channels = [16, 16, 24, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    depthwise_expansion_factor = 2
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [64, 80, 96]


def build_MobileViT_v1(model_type: str = "S", num_classes: int = 1000, input_shape: tuple = (None, None, 3), **kwargs):

    """
    Create MobileViT-v1 Classification models

    Arguments
    --------
        model_type: (str)   MobileViT version to create. Options: S, XS, XSS

        num_classes: (int)   Number of output classes

        input_shape: (tuple) Input shape -> H, W, C

    Additional arguments:
    ---------------------

        linear_drop: (float) Dropout rate for Dense layers

        attention_drop: (float) Dropout rate for the attention matrix

    """

    if model_type == "S":
        config = config_MobileViT_v1_S()
    elif model_type == "XS":
        config = config_MobileViT_v1_XS()
    elif model_type == "XXS":
        config = config_MobileViT_v1_XXS()
    else:
        raise ValueError("Bad Input. 'model_type' should one of ['S', 'XS', 'XXS']")

    model = MobileViT_v1(
        out_channels=config.out_channels,
        expansion_factor=config.depthwise_expansion_factor, # Inverted residual block -> Bottelneck expansion size
        tf_repeats=config.tf_repeats, 
        tf_embedding_dims=config.tf_embedding_dims,
        num_classes=num_classes,
        input_shape=input_shape,
        model_type=model_type,
        **kwargs,
    )

    return model


if __name__ == "__main__":

    model = build_MobileViT_v1(
        model_type=r"S",  # "XS", "XXS"
        input_shape=(256, 256, 3),  # (None, None, 3)
        num_classes=1000,
        linear_drop=0.0,
        attention_drop=0.0,
    )

    model.summary(positions=[0.33, 0.64, 0.75, 1.0])
