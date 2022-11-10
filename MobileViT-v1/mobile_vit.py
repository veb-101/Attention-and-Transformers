import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense

from BaseLayers import ConvLayer, InvertedResidualBlock
from mobile_vit_block import MobileViTBlock


def MobileViT(
    out_channels,
    expansion_factor,
    tf_repeats,
    tf_embedding_dims,
    linear_drop=0.0,
    attention_drop=0.2,
    num_classes=1000,
    input_shape=(256, 256, 3),
    model_type="S",
):

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


def build_mobileViT_S():

    out_channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    expansion_factor = 4
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [144, 192, 240]

    model = MobileViT(out_channels, expansion_factor, tf_repeats, tf_embedding_dims, num_classes=1000, input_shape=(256, 256, 3), model_type="S")
    return model


def build_mobileViT_XS():

    out_channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    expansion_factor = 4
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [96, 120, 144]

    model = MobileViT(out_channels, expansion_factor, tf_repeats, tf_embedding_dims, num_classes=1000, input_shape=(256, 256, 3), model_type="XS")
    return model


def build_mobileViT_XXS():

    out_channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    expansion_factor = 2
    tf_repeats = [2, 4, 3]
    tf_embedding_dims = [64, 80, 96]

    model = MobileViT(out_channels, expansion_factor, tf_repeats, tf_embedding_dims, num_classes=1000, input_shape=(256, 256, 3), model_type="XXS")
    return model


if __name__ == "__main__":

    model = build_mobileViT_XXS()

    model.summary()
