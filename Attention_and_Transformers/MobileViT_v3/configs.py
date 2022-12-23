from typing import Optional
from Attention_and_Transformers.MobileViT_v1 import get_mobile_vit_v1_configs
from Attention_and_Transformers.MobileViT_v2 import get_mobile_vit_v2_configs


def get_mobile_vit_v3_configs(
    ref_version: str = "v1",
    v1_model_type: Optional[str] = None,
    fast_version: Optional[bool] = False,
    v2_width_multiplier: Optional[float] = None,
    ref_version_updates: Optional[dict] = None,
):

    if ref_version == "v1":
        if v1_model_type == "S":
            # https://github.com/micronDLA/MobileViTv3/blob/d381beb017ae3c244686afaa48064f95865df7b9/MobileViTv3-v1/cvnets/models/classification/mobilevit.py#L86
            # exp_channels = min(mobilevit_config["last_layer_exp_factor"] * in_channels, 960) = min(320 * 4, 960) = 960

            exp_channels = 960
            base_model_updates = {
                    "block_3_1_dims": 128,
                    "block_3_2_dims": 128,
                    "block_4_1_dims": 256,
                    "block_4_2_dims": 256,
                    "block_5_1_dims": 320,
                    "block_5_2_dims": 320,
                    "final_conv_dims": exp_channels,
                }

        elif v1_model_type == "XS":
            # exp_channels = min(160 * 4, 960) = 640
            exp_channels = 640
            base_model_updates = {
                "block_3_1_dims": 96,
                "block_3_2_dims": 96,
                "block_4_1_dims": 160,
                "block_4_2_dims": 160,
                "block_5_1_dims": 160,
                "block_5_2_dims": 160,
                "final_conv_dims": exp_channels,
            }

        else:
            # exp_channels = min(128 * 4, 960) = 512
            exp_channels = 512
            base_model_updates = {
                "block_3_1_dims": 64,
                "block_3_2_dims": 64,
                "block_4_1_dims": 80,
                "block_4_2_dims": 80,
                "block_5_1_dims": 128,
                "block_5_2_dims": 128,
                "final_conv_dims": exp_channels,
            }

        if fast_version:
            base_model_updates["tf_block_4_repeats"] = 2

        if ref_version_updates:
            base_model_updates.update(ref_version_updates)
            # updates = {**base_model_updates, **ref_version_updates}

        configs = get_mobile_vit_v1_configs(
            model_type=v1_model_type,
            updates=base_model_updates,
        )

    else:
        configs = get_mobile_vit_v2_configs(width_multiplier=v2_width_multiplier, updates=ref_version_updates)

    return configs


# def update_mobile_vit_v2_configs(width_multiplier: Union[int, float]):
#     # Base
#     out_channels = [32, 64, 128, 256, 384, 512]
#     tf_embedding_dims = [128, 192, 256]
#     tf_repeats = [2, 4, 3]
#     depthwise_expansion_factor = 2

#     # Update using width multiplier
#     layer_0_dim = bound_fn(min_val=16, max_val=64, value=out_channels[0] * width_multiplier)
#     layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))

#     layer_1_dim = int(make_divisible(out_channels[1] * width_multiplier, divisor=16))

#     layer_2_dim = int(make_divisible(out_channels[2] * width_multiplier, divisor=8))

#     layer_3_dim = int(make_divisible(out_channels[3] * width_multiplier, divisor=8))
#     layer_3_attn_dim = int(make_divisible(tf_embedding_dims[0] * width_multiplier, divisor=8))

#     layer_4_dim = int(make_divisible(out_channels[4] * width_multiplier, divisor=8))
#     layer_4_attn_dim = int(make_divisible(tf_embedding_dims[1] * width_multiplier, divisor=8))

#     layer_5_dim = int(make_divisible(out_channels[5] * width_multiplier, divisor=8))
#     layer_5_attn_dim = int(make_divisible(tf_embedding_dims[2] * width_multiplier, divisor=8))

#     info_dict = {
#         "tf_repeats": tf_repeats,
#         "depthwise_expansion_factor": depthwise_expansion_factor,
#         "out_channels": {
#             "block_1_1_dim": layer_0_dim,
#             "block_1_2_dim": layer_1_dim,
#             "block_2_dim": layer_2_dim,
#             "block_3_dim": layer_3_dim,
#             "block_4_dim": layer_4_dim,
#             "block_5_dim": layer_5_dim,
#         },
#         "tf_embedding_dims": {
#             "block_3_attn_dim": layer_3_attn_dim,
#             "block_4_attn_dim": layer_4_attn_dim,
#             "block_5_attn_dim": layer_5_attn_dim,
#         },
#     }

#     return info_dict

# base_config = get_mobile_vit_v1_configs(
#     model_type=v1_model_type,
#     updates=base_model_updates,
# )

# if ref_version_updates is not None:
#     base_config.__dict__.update(ref_version_updates)
