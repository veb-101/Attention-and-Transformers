import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dropout, Dense, LayerNormalization, Concatenate

from .BaseLayers import ConvLayer
from .multihead_self_attention_2D import MultiHeadSelfAttentionEinSum2D as MHSA

tf.random.set_seed(1)
tf.keras.utils.set_random_seed(1)


class Transformer(Layer):
    def __init__(
        self,
        num_heads: int = 4,
        embedding_dim: int = 90,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        linear_drop: float = 0.2,
        attention_drop: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.norm_1 = LayerNormalization(epsilon=1e-6)
        self.norm_2 = LayerNormalization(epsilon=1e-6)

        self.attn = MHSA(
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            qkv_bias=qkv_bias,
            attention_drop=attention_drop,
            linear_drop=linear_drop,
        )

        hidden_features = int(embedding_dim * mlp_ratio)

        self.mlp = Sequential(
            layers=[
                Dense(hidden_features, activation="swish"),
                Dropout(linear_drop),
                Dense(embedding_dim),
                Dropout(linear_drop),
            ]
        )

    def call(self, x):

        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))

        return x


# https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/modules/mobilevit_block.py#L186
def unfolding(nn, patch_h=2, patch_w=2):
    """
    H, W
    [                            [
        [1, 2, 3, 4],     Goal      [1, 3, 9, 11],
        [5, 6, 7, 8],     ====>     [2, 4, 10, 12],
        [9, 10, 11, 12],            [5, 7, 13, 15],
        [13, 14, 15, 16],           [6, 8, 14, 16]
    ]                            ]
    """
    B, H, W, C = tf.shape(nn)[0], tf.shape(nn)[1], tf.shape(nn)[2], tf.shape(nn)[3]
    patch_area = int(patch_h * patch_w)

    num_patch_h, num_patch_w = int(tf.math.ceil(H / patch_h)), int(tf.math.ceil(W / patch_w))
    num_patches = num_patch_h * num_patch_w

    interpolate = False

    if ((num_patch_h * patch_h) != H) or ((num_patch_w * patch_w) != W):
        nn = tf.image.resize(nn, [num_patch_h * patch_h, num_patch_w * patch_w], method="bilinear")
        interpolate = True

    reshaped_fm = tf.reshape(nn, (B * num_patch_h, patch_h, num_patch_w, patch_w * C))
    transposed_fm = tf.transpose(reshaped_fm, perm=[0, 2, 1, 3])
    reshaped_fm = tf.reshape(transposed_fm, (B, num_patches, patch_area, C))
    transposed_fm = tf.transpose(reshaped_fm, perm=[0, 2, 1, 3])

    info_dict = {
        "orig_size": (H, W),
        "interpolate": interpolate,
        "num_patches_w": num_patch_w,
        "num_patches_h": num_patch_h,
    }

    return transposed_fm, info_dict


# https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/modules/mobilevit_block.py#L233
def folding(nn, info_dict: dict, patch_h=2, patch_w=2):

    B, C = tf.shape(nn)[0], tf.shape(nn)[3]

    num_patch_h = info_dict["num_patches_h"]
    num_patch_w = info_dict["num_patches_w"]

    # [B, P, N C] --> [B, N, P, C]
    nn = tf.transpose(nn, perm=(0, 2, 1, 3))

    # [B, N, P, C] --> [B*nh, nw, ph, pw*C]
    nn = tf.reshape(nn, (B * num_patch_h, num_patch_w, patch_h, patch_w * C))

    # [B*nh, nw, ph, pw*C] --> [B*nh, ph, nw, pw*C]
    nn = tf.transpose(nn, perm=(0, 2, 1, 3))

    # [B*nh, ph, nw, pw*C] --> [B, nh*ph, nw,pw, C]
    nn = tf.reshape(nn, (B, num_patch_h * patch_h, num_patch_w * patch_w, C))

    if info_dict["interpolate"]:
        nn = tf.image.resize(nn, size=info_dict["orig_size"])
    return nn


class MobileViTBlock(Layer):
    def __init__(
        self,
        out_filters=64,
        embedding_dim=90,
        patch_size=4,
        transformer_repeats=2,
        num_heads=4,
        attention_drop=0.0,
        linear_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.out_filters = out_filters
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.transformer_repeats = transformer_repeats
        self.num_heads = num_heads

        # local_feature_extractor 1 and 2
        self.local_features_1 = ConvLayer(num_filters=self.out_filters, kernel_size=3, strides=1, use_bn=True, use_activation=True)
        self.local_features_2 = ConvLayer(
            num_filters=self.embedding_dim, kernel_size=1, strides=1, use_bn=False, use_activation=False, use_bias=False
        )

        layers = [
            Transformer(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                linear_drop=linear_drop,
                attention_drop=attention_drop,
            )
            for _ in range(self.transformer_repeats)
        ]

        layers.append(LayerNormalization(epsilon=1e-6))

        # Repeated transformer blocks
        self.transformer_blocks = Sequential(layers=layers)

        # Fusion blocks
        self.local_features_3 = ConvLayer(num_filters=self.out_filters, kernel_size=1, strides=1, use_bn=True, use_activation=True)
        self.concat = Concatenate(axis=-1)
        self.fuse_local_global = ConvLayer(num_filters=self.out_filters, kernel_size=3, strides=1, use_bn=True, use_activation=True)

    def call(self, x):
        local_representation = self.local_features_1(x)
        local_representation = self.local_features_2(local_representation)

        # Transformer as Convolution Steps
        # --------------------------------
        # # Unfolding
        unfolded, info_dict = unfolding(local_representation)

        # # Infomation sharing/mixing --> global representation
        global_representation = self.transformer_blocks(unfolded)

        # # Folding
        folded = folding(global_representation, info_dict=info_dict)
        # # --------------------------------

        # Fusion
        local_mix = self.local_features_3(folded)
        fusion = self.concat([local_mix, x])
        fusion = self.fuse_local_global(fusion)

        return fusion


if __name__ == "__main__":
    batch = 2
    H = W = 32
    C = 96
    P = 2 * 2
    L = 4
    embedding_dim = 144

    mvitblk = MobileViTBlock(
        out_filters=C,
        embedding_dim=embedding_dim,
        patch_size=P,
        transformer_repeats=L,
        attention_drop=0.0,
        linear_drop=0.0,
    )

    inputs = tf.random.normal((batch, H, W, C))

    out = mvitblk(inputs)
    print("inputs.shape", inputs.shape)
    print("out.shape", out.shape)
