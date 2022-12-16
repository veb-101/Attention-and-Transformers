import tensorflow as tf

from ATF import ViT
from ATF import MobileViT_v1


# Test ViT
model = ViT.VisionTransformer(
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
model = MobileViT_v1.build_MobileViT_V1(model_type="S")
model.summary()
