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

# Load MobileViT-V2
model = build_MobileViT_v2(width_multiplier=0.5)
print("MobileViT_v2 0.5 Num. Parameteres:", model.count_params())

# Load MobileViT-V3
model = build_MobileViT_v3(width_multiplier=0.5)
print("MobileViT_v3 0.5 Num. Parameteres:", model.count_params())
