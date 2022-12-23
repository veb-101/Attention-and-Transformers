from typing import Optional, Union

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

from .configs import get_mobile_vit_v3_configs
from .BaseLayers import ConvLayer, InvertedResidualBlock
from .mobile_vit_v3_block import MobileViT_v3_Block


def MobileViT_v3(
    configs,
    ref_version: str,
    linear_drop: Optional[float] = 0.0,
    attention_drop: Optional[float] = 0.2,
    num_classes: Optional[int] = 1000,
    input_shape: Optional[tuple] = (256, 256, 3),
    model_name: Optional[str] = "MobileViT-v3",
):
    """
    Arguments
    --------

        configs: A dataclass instance with model information such as per layer output channels, transformer embedding dimensions, transformer repeats, IR expansion factor

        ref_version: (str) MobileViT version reference (v1 or v2)

        num_classes: (int)   Number of output classes

        input_shape: (tuple) Input shape -> H, W, C

        model_type: (str)   Model to create

        linear_drop: (float) Dropout rate for Dense layers

        attention_drop: (float) Dropout rate for the attention matrix

    """

    input_layer = Input(shape=input_shape)

    # Block 1
    out = ConvLayer(
        num_filters=configs.block_1_1_dims,
        kernel_size=3,
        strides=2,
        name="block-1-Conv",
    )(input_layer)

    out_1_2 = InvertedResidualBlock(
        in_channels=configs.block_1_1_dims,
        out_channels=configs.block_1_2_dims,
        depthwise_stride=1,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-1-IR2",
    )(out)

    if out.shape[-1] == out_1_2.shape[-1]:
        out = out + out_1_2
    else:
        out = out_1_2

    # Block 2
    out = InvertedResidualBlock(
        in_channels=configs.block_1_2_dims,
        out_channels=configs.block_2_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-2-IR1",
    )(out)

    out_b2_2 = InvertedResidualBlock(
        in_channels=configs.block_2_1_dims,
        out_channels=configs.block_2_2_dims,
        depthwise_stride=1,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-2-IR2",
    )(out)

    out = out + out_b2_2

    if ref_version == "v1":
        out_b2_3 = InvertedResidualBlock(
            in_channels=configs.block_2_2_dims,
            out_channels=configs.block_2_3_dims,
            depthwise_stride=1,
            expansion_factor=configs.depthwise_expansion_factor,
            name="block-2-IR3",
        )(out)

        out = out + out_b2_3

    # Block 3
    out = InvertedResidualBlock(
        in_channels=configs.block_2_3_dims if ref_version == "v1" else configs.block_2_2_dims,
        out_channels=configs.block_3_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-3-IR1",
    )(out)

    out = MobileViT_v3_Block(
        ref_version=ref_version,
        out_filters=configs.block_3_2_dims,
        embedding_dim=configs.tf_block_3_dims,
        transformer_repeats=configs.tf_block_3_repeats,
        name="MobileViTBlock-1",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 4
    out = InvertedResidualBlock(
        in_channels=configs.block_3_2_dims,
        out_channels=configs.block_4_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-4-IR1",
    )(out)

    out = MobileViT_v3_Block(
        ref_version=ref_version,
        out_filters=configs.block_4_2_dims,
        embedding_dim=configs.tf_block_4_dims,
        transformer_repeats=configs.tf_block_4_repeats,
        name="MobileViTBlock-2",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 5
    out = InvertedResidualBlock(
        in_channels=configs.block_4_2_dims,
        out_channels=configs.block_5_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-5-IR1",
    )(out)

    out = MobileViT_v3_Block(
        ref_version=ref_version,
        out_filters=configs.block_5_2_dims,
        embedding_dim=configs.tf_block_5_dims,
        transformer_repeats=configs.tf_block_5_repeats,
        name="MobileViTBlock-3",
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    if ref_version == "v1":
        # final conv layer
        out = ConvLayer(num_filters=configs.final_conv_dims, kernel_size=1, strides=1)(out)

    # Output layer
    out = GlobalAveragePooling2D()(out)

    if linear_drop > 0.0:
        out = Dropout(rate=linear_drop)(out)

    out = Dense(units=num_classes)(out)

    model = Model(inputs=input_layer, outputs=out, name=model_name)

    return model


def build_MobileViT_v3(
    ref_version: str = "v1",  # v1, v2
    indentifier: Union[float, str] = "S",  # "S", "XS", "XXS", 1.0, 0.5, 2.0
    fast_version: bool = False,
    num_classes: int = 1000,
    input_shape: tuple = (256, 256, 3),
    updates: Optional[dict] = None,
    **kwargs,
):
    ref_version = ref_version.lower()

    v1_model_type = None
    v2_width_multiplier = None

    if ref_version == "v1":

        indentifier = indentifier.upper()

        if not isinstance(indentifier, str) or indentifier not in ("S", "XS", "XXS"):
            raise ValueError(
                f"input={indentifier} Identifier for 'ref_version=v1' should be a 'str' with one of the following value: ('S', 'XS', 'XXS')"
            )
        v1_model_type = indentifier

    elif ref_version == "v2":
        try:
            _ = float(indentifier)
        except ValueError:
            raise ValueError(
                f"value:{indentifier}, type:{type(indentifier)} Identifier for 'ref_version=v2' should be either be an integer or float eg. (0.5, 0.75, 1.0, 1.5, 2.0)"
            )

        v2_width_multiplier = float(indentifier)

    else:
        raise ValueError(f"Bad value. input={ref_version} 'ref_version' should be one of ('v1', 'v2')")

    configs = get_mobile_vit_v3_configs(
        ref_version=ref_version,
        v1_model_type=v1_model_type,
        fast_version=fast_version,
        v2_width_multiplier=v2_width_multiplier,
        ref_version_updates=updates,
    )

    model = MobileViT_v3(
        configs=configs,
        ref_version=ref_version,
        num_classes=num_classes,
        input_shape=input_shape,
        model_name=f"MobileViT-v3_{ref_version}_{indentifier}{'_fast' if fast_version else ''}",
        **kwargs,
    )

    return model


if __name__ == "__main__":

    model = build_MobileViT_v3(
        ref_version="v2",  # v1, v2
        indentifier=1.0,  # "S", "XS", "XXS", "S-FAST", "XS-FAST", "XXS-FAST", 1.0, 0.5, 2.0
        input_shape=(256, 256, 3),
        num_classes=1000,
        linear_drop=0.0,
        attention_drop=0.0,
    )

    print(model.name, model.count_params())

    # model = build_MobileViT_v3(
    #     ref_version="v1",  # v1, v2
    #     indentifier="XXS",  # "S", "XS", "XXS", "S-FAST", "XS-FAST", "XXS-FAST", 1.0, 0.5, 2.0
    #     input_shape=(256, 256, 3),
    #     num_classes=1000,
    #     linear_drop=0.0,
    #     attention_drop=0.0,
    # )

    # print(model.name, model.count_params())

    # model = build_MobileViT_v3(
    #     ref_version="v1",  # v1, v2
    #     indentifier="XXS-FAST",  # "S", "XS", "XXS", "S-FAST", "XS-FAST", "XXS-FAST", 1.0, 0.5, 2.0
    #     input_shape=(256, 256, 3),
    #     num_classes=1000,
    #     linear_drop=0.0,
    #     attention_drop=0.0,
    # )

    # print(model.name, model.count_params())
