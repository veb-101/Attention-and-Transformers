from dataclasses import dataclass
from typing import Optional, Union

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

from .utils import bound_fn, make_divisible
from .BaseLayers import ConvLayer, InvertedResidualBlock
from .mobile_vit_v3_block import MobileViT_v3_Block


def MobileViT_v3(
    out_channels: list,
    expansion_factor: int,
    tf_repeats: list,
    tf_embedding_dims: list,
    linear_drop: Optional[float] = 0.0,
    attention_drop: Optional[float] = 0.2,
    num_classes: Optional[int] = 1000,
    input_shape: Optional[tuple] = (256, 256, 3),
    model_name: str = "MobileViT-v3-1.0",
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
    out = ConvLayer(num_filters=out_channels["block_1_1_dim"], kernel_size=3, strides=2, name="block-1-Conv")(input_layer)

    out = InvertedResidualBlock(
        in_channels=out_channels["block_1_1_dim"],
        out_channels=out_channels["block_1_2_dim"],
        depthwise_stride=1,
        expansion_factor=expansion_factor,
        name="block-1-IR2",
    )(out)

    # Block 2
    out = InvertedResidualBlock(
        in_channels=out_channels["block_1_2_dim"],
        out_channels=out_channels["block_2_dim"],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-2-IR1",
    )(out)

    out_b2_2 = InvertedResidualBlock(
        in_channels=out_channels["block_2_dim"],
        out_channels=out_channels["block_2_dim"],
        depthwise_stride=1,
        expansion_factor=expansion_factor,
        name="block-2-IR2",
    )(out)

    out = out + out_b2_2

    # # ========================================================
    # # According to paper, one more repeat should be present, but not present in the final code.

    # out_b2_3 = InvertedResidualBlock(
    #     in_channels=out_channels["block_2_dim"],
    #     out_channels=out_channels["block_2_dim"],
    #     depthwise_stride=1,
    #     expansion_factor=expansion_factor,
    #     name="block-2-IR3",
    # )(out)

    # out = out + out_b2_3
    # # ========================================================

    # Block 3
    out = InvertedResidualBlock(
        in_channels=out_channels["block_2_dim"],
        out_channels=out_channels["block_3_dim"],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-3-IR1",
    )(out)

    out = MobileViT_v3_Block(
        out_filters=out_channels["block_3_dim"],
        embedding_dim=tf_embedding_dims["block_3_attn_dim"],
        transformer_repeats=tf_repeats[0],
        name="MobileViTBlock-1",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 4
    out = InvertedResidualBlock(
        in_channels=out_channels["block_3_dim"],
        out_channels=out_channels["block_4_dim"],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-4-IR1",
    )(out)

    out = MobileViT_v3_Block(
        out_filters=out_channels["block_4_dim"],
        embedding_dim=tf_embedding_dims["block_4_attn_dim"],
        transformer_repeats=tf_repeats[1],
        name="MobileViTBlock-2",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 5
    out = InvertedResidualBlock(
        in_channels=out_channels["block_4_dim"],
        out_channels=out_channels["block_5_dim"],
        depthwise_stride=2,
        expansion_factor=expansion_factor,
        name="block-5-IR1",
    )(out)

    out = MobileViT_v3_Block(
        out_filters=out_channels["block_5_dim"],
        embedding_dim=tf_embedding_dims["block_5_attn_dim"],
        transformer_repeats=tf_repeats[2],
        name="MobileViTBlock-3",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Output layer
    out = GlobalAveragePooling2D()(out)

    if linear_drop > 0.0:
        out = Dropout(rate=linear_drop)(out)

    out = Dense(units=num_classes)(out)

    model = Model(inputs=input_layer, outputs=out, name=model_name)

    return model


def update_dimensions(width_multiplier: Union[int, float]):
    out_channels = config_MobileViT_v3.out_channels
    tf_embedding_dims = config_MobileViT_v3.tf_embedding_dims

    layer_0_dim = bound_fn(min_val=16, max_val=64, value=out_channels[0] * width_multiplier)
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))

    layer_1_dim = int(make_divisible(out_channels[1] * width_multiplier, divisor=16))

    layer_2_dim = int(make_divisible(out_channels[2] * width_multiplier, divisor=8))

    layer_3_dim = int(make_divisible(out_channels[3] * width_multiplier, divisor=8))
    layer_3_attn_dim = int(make_divisible(tf_embedding_dims[0] * width_multiplier, divisor=8))

    layer_4_dim = int(make_divisible(out_channels[4] * width_multiplier, divisor=8))
    layer_4_attn_dim = int(make_divisible(tf_embedding_dims[1] * width_multiplier, divisor=8))

    layer_5_dim = int(make_divisible(out_channels[5] * width_multiplier, divisor=8))
    layer_5_attn_dim = int(make_divisible(tf_embedding_dims[2] * width_multiplier, divisor=8))

    return {
        "out_channels": {
            "block_1_1_dim": layer_0_dim,
            "block_1_2_dim": layer_1_dim,
            "block_2_dim": layer_2_dim,
            "block_3_dim": layer_3_dim,
            "block_4_dim": layer_4_dim,
            "block_5_dim": layer_5_dim,
        },
        "tf_embedding_dims": {
            "block_3_attn_dim": layer_3_attn_dim,
            "block_4_attn_dim": layer_4_attn_dim,
            "block_5_attn_dim": layer_5_attn_dim,
        },
    }


@dataclass(frozen=True)
class config_MobileViT_v3:
    out_channels = [32, 64, 128, 256, 384, 512]
    depthwise_expansion_factor = 2
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [128, 192, 256]


def build_MobileViT_v3(
    width_multiplier: float = 1.0,
    num_classes: int = 1000,
    input_shape: tuple = (None, None, 3),
    **kwargs,
):
    """
    Create MobileViT-v3 Classification models

    Arguments
    --------
        width_multiplier: (int, float) manipulate number of channels.
                            Default: 1.0 --> Refers to the base model.

        num_classes: (int)   Number of output classes

        input_shape: (tuple) Input shape -> H, W, C

    Additional arguments:
    ---------------------

        linear_drop: (float) Dropout rate for Dense layers

        attention_drop: (float) Dropout rate for the attention matrix

    """
    updated_dims = update_dimensions(width_multiplier)

    out_channels = updated_dims["out_channels"]
    tf_embedding_dims = updated_dims["tf_embedding_dims"]

    model = MobileViT_v3(
        out_channels=out_channels,
        expansion_factor=config_MobileViT_v3.depthwise_expansion_factor,
        tf_repeats=config_MobileViT_v3.tf_repeats,
        tf_embedding_dims=tf_embedding_dims,
        num_classes=num_classes,
        input_shape=input_shape,
        model_name=f"MobileViT-v3-{width_multiplier}",
        **kwargs,
    )

    return model


if __name__ == "__main__":

    model = build_MobileViT_v3(
        width_multiplier=0.75,
        input_shape=(256, 256, 3),
        num_classes=1000,
        linear_drop=0.0,
        attention_drop=0.0,
    )

    model.summary(positions=[0.33, 0.64, 0.75, 1.0])
