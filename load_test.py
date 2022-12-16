import tensorflow as tf

# from ATF import ViT
# from ATF import MobileViT_v1

from Attention_and_Transformers.ViT import VisionTransformer
from Attention_and_Transformers.MobileViT_v1 import build_MobileViT_v1


# Test ViT
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
model.summary()


# Test MobileViT-V1
model = build_MobileViT_v1(model_type="S")
model.summary()
