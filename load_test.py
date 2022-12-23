import tensorflow as tf

# from ATF import ViT
# from ATF import MobileViT_v1

from Attention_and_Transformers.ViT import VisionTransformer
from Attention_and_Transformers.MobileViT_v1 import build_MobileViT_v1
from Attention_and_Transformers.MobileViT_v2 import build_MobileViT_v2
from Attention_and_Transformers.MobileViT_v3 import build_MobileViT_v3


# Load ViT
model = VisionTransformer(
    img_size=32,
    patch_size=4,
    n_classes=10,
    embedding_dim=64,
    depth=2,
    num_heads=2,
    mlp_ratio=2.0,
    linear_drop=0.2,
    attention_drop=0.2,
)

model.build((None, None, None, 3))
print("ViT test Num. Parameteres:", model.count_params())

# Load MobileViT-V1
model = build_MobileViT_v1(model_type="XXS")
print("MobileViT_v1 XXS Num. Parameteres:", model.count_params())

model = build_MobileViT_v1(model_type="XS")
print("MobileViT_v1 XS Num. Parameteres:", model.count_params())

model = build_MobileViT_v1(model_type="S")
print("MobileViT_v1 S Num. Parameteres:", model.count_params())

# Load MobileViT-V1
model = build_MobileViT_v2(width_multiplier=0.5)
# model = build_MobileViT_v2(width_multiplier=0.75)
# model = build_MobileViT_v2(width_multiplier=1.0)
# model = build_MobileViT_v2(width_multiplier=1.5)
# model = build_MobileViT_v2(width_multiplier=2.0)

model.save(f"{model.name}")
print(f"{model.name} Num. Parameteres:", model.count_params())

updates = {
    "block_3_1_dims": 256,
    "block_3_2_dims": 384,
    "tf_block_3_dims": 164,
    "tf_block_3_repeats": 3,
}

model = build_MobileViT_v2(width_multiplier=0.5, updates=updates)
print(f"{model.name} Num. Parameteres:", model.count_params())

# ================================================================================


# =================================-MobileViT-v3-=================================
model = build_MobileViT_v3(ref_version="v1", indentifier="XXS")
# model = build_MobileViT_v3(ref_version="v1", indentifier="XS")
# model = build_MobileViT_v3(ref_version="v1", indentifier="S")
print(f"{model.name} Num. Parameteres:", model.count_params())

model = build_MobileViT_v3(ref_version="v1", indentifier="XXS-fast")
# model = build_MobileViT_v3(ref_version="v1", indentifier="XS-fast")
# model = build_MobileViT_v3(ref_version="v1", indentifier="S-fast")
print(f"{model.name} Num. Parameteres:", model.count_params())


model = build_MobileViT_v3(ref_version="v2", indentifier=0.5)
# model = build_MobileViT_v3(ref_version="v2", indentifier=0.75)
# model = build_MobileViT_v3(ref_version="v2", indentifier=1.0)
print(f"{model.name} Num. Parameteres:", model.count_params())
# ================================================================================
