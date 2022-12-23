from typing import Union
from .utils import bound_fn, make_divisible


def model_ref_v1_info(model_type: str = "S"):
    out_channels_str = ["block_1_1_dim", "block_1_2_dim", "block_2_dim", "block_3_dim", "block_4_dim", "block_5_dim", "conv_final"]
    tf_embedding_str = ["block_3_attn_dim", "block_4_attn_dim", "block_5_attn_dim"]

    # Base
    if model_type == "S":
        # https://github.com/micronDLA/MobileViTv3/blob/d381beb017ae3c244686afaa48064f95865df7b9/MobileViTv3-v1/cvnets/models/classification/mobilevit.py#L86
        # exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        # exp_channels = min(320 * 4, 960)
        exp_channels = 960
        out_channels = [16, 32, 64, 128, 256, 320, exp_channels]
        tf_embedding_dims = [144, 192, 240]
        depthwise_expansion_factor = 4

    elif model_type == "XS":
        # exp_channels = min(160 * 4, 960)
        out_channels = [16, 32, 48, 96, 160, 160, 640]
        tf_embedding_dims = [96, 120, 144]
        depthwise_expansion_factor = 4

    else:
        # exp_channels = min(128 * 4, 960)
        out_channels = [16, 16, 24, 64, 80, 128, 512]
        tf_embedding_dims = [64, 80, 96]
        depthwise_expansion_factor = 2

    out_channels = {i: j for i, j in zip(out_channels_str, out_channels)}
    tf_embedding_dims = {i: j for i, j in zip(tf_embedding_str, tf_embedding_dims)}
    tf_repeats = [2, 4, 3]

    info_dict = {
        "out_channels": out_channels,
        "tf_embedding_dims": tf_embedding_dims,
        "depthwise_expansion_factor": depthwise_expansion_factor,
        "tf_repeats": tf_repeats,
    }
    return info_dict


def model_ref_v2_info(width_multiplier: Union[int, float]):
    # Base
    out_channels = [32, 64, 128, 256, 384, 512]
    tf_embedding_dims = [128, 192, 256]
    tf_repeats = [2, 4, 3]
    depthwise_expansion_factor = 2

    # Update using width multiplier
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

    info_dict = {
        "tf_repeats": tf_repeats,
        "depthwise_expansion_factor": depthwise_expansion_factor,
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

    return info_dict
