from dataclasses import dataclass
from typing import Optional, Union

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

from .BaseLayers import ConvLayer, InvertedResidualBlock
from .mobile_vit_v3_block import MobileViT_v3_Block
from .configs import model_ref_v1_info, model_ref_v2_info


def MobileViT_v3(
    ref_version: str,
    attention_type: str,
    out_channels: list,
    expansion_factor: int,
    tf_repeats: list,
    tf_embedding_dims: list,
    linear_drop: Optional[float] = 0.0,
    attention_drop: Optional[float] = 0.2,
    num_classes: Optional[int] = 1000,
    input_shape: Optional[tuple] = (256, 256, 3),
    model_name: Optional[str] = "MobileViT-v3",
):

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

    if ref_version.lower() == "v1":
        out_b2_3 = InvertedResidualBlock(
            in_channels=out_channels["block_2_dim"],
            out_channels=out_channels["block_2_dim"],
            depthwise_stride=1,
            expansion_factor=expansion_factor,
            name="block-2-IR3",
        )(out)

        out = out + out_b2_3

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
        attention_type=attention_type,
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
        attention_type=attention_type,
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
        attention_type=attention_type,
        name="MobileViTBlock-3",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    if out_channels.get("conv_final", False):
        # final conv layer
        out = ConvLayer(num_filters=out_channels["conv_final"], kernel_size=1, strides=1)(out)

    out = GlobalAveragePooling2D()(out)

    if linear_drop > 0.0:
        out = Dropout(rate=linear_drop)(out)

    out = Dense(units=num_classes)(out)

    model = Model(inputs=input_layer, outputs=out, name=model_name)

    return model


def build_MobileViT_v3(
    ref_version: str = "v1",  # v1, v2
    indentifier: Union[float, str] = "S",  # "S", "XS", "XXS", 1.0, 0.5, 2.0
    num_classes: int = 1000,
    input_shape: tuple = (None, None, 3),
    **kwargs,
):

    if ref_version.lower() == "v1":
        if not isinstance(indentifier, str):
            raise ValueError("Identifier for 'ref_version=v1' should be one of ('S', 'XS', 'XXS')")
        elif indentifier.upper() not in ("S", "XS", "XXS"):
            raise ValueError("Bad Input. 'identifier' for v1 should be one of ('S', 'XS', 'XXS')")

        config = model_ref_v1_info(model_type=indentifier.upper())

    elif ref_version.lower() == "v2":
        if not isinstance(indentifier, int) and not isinstance(indentifier, float):
            raise ValueError("Identifier for 'ref_version=v2' should be either be an integer or float eg. (0.5, 0.75, 1.0, 1.5, 2.0)")

        config = model_ref_v2_info(width_multiplier=indentifier)

    else:
        raise ValueError("Bad Input. 'ref_version' should be one of ('v1', 'v2')")

    attention_type = config["attention_type"].upper()
    if attention_type not in ("MHSA", "LSA"):
        raise ValueError("Bad Input. 'attention_type' should be one of ('MHSA, 'LSA')")

    model = MobileViT_v3(
        ref_version=ref_version,
        attention_type=config["attention_type"],
        out_channels=config["out_channels"],
        expansion_factor=config["depthwise_expansion_factor"],
        tf_repeats=config["tf_repeats"],
        tf_embedding_dims=config["tf_embedding_dims"],
        num_classes=num_classes,
        input_shape=input_shape,
        model_name=f"MobileViT-v3-({ref_version}-{indentifier})",
        **kwargs,
    )

    return model


if __name__ == "__main__":

    model = build_MobileViT_v3(
        ref_version="v1",  # v1, v2
        indentifier="XXS",  # "S", "XS", "XXS", 1.0, 0.5, 2.0
        input_shape=(256, 256, 3),
        num_classes=1000,
        linear_drop=0.0,
        attention_drop=0.0,
    )

    print(model.name, model.count_params())

    model = build_MobileViT_v3(
        ref_version="v2",  # v1, v2
        indentifier=0.5,  # "S", "XS", "XXS", 1.0, 0.5, 2.0
        input_shape=(256, 256, 3),
        num_classes=1000,
        linear_drop=0.0,
        attention_drop=0.0,
    )

    print(model.name, model.count_params())
