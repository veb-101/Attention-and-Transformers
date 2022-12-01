import tensorflow as tf
from ATF import VisionTransformers
from ATF import MobileViT_v1


# Test ViT
model = VisionTransformers.VisionTransformer(
    img_size=32, patch_size=4, n_classes=10, embedding_dim=256, depth=6, num_heads=8, mlp_ratio=2.0, linear_drop=0.2, attention_drop=0.2
)


a = tf.random.normal((1, 32, 32, 3))
_ = model(a)
model.summary()


# Test MobileViT-V1
model = MobileViT_v1.build_MobileViT_V1(model_type="S")
model.summary()
