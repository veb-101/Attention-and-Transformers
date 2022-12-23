from typing import Union, Optional

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, DepthwiseConv2D, Dropout, Dense, BatchNormalization, LayerNormalization, Activation, Concatenate

# from Attention_and_Transformers.MobileViT_v1.multihead_self_attention_2D import MultiHeadSelfAttentionEinSum2D as MHSA
# from Attention_and_Transformers.MobileViT_v2.linear_attention import LinearSelfAttention as LSA

from .BaseLayers import ConvLayer
from .attention_blocks import MultiHeadSelfAttentionEinSum2D as MHSA
from .attention_blocks import LinearSelfAttention as LSA


class Transformer(Layer):
    def __init__(
        self,
        ref_version: str = "v1",
        num_heads: int = 4,
        embedding_dim: int = 90,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        linear_drop: float = 0.2,
        attention_drop: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ref_version = ref_version
        self.embedding_dim = embedding_dim
        self.mlp_ratio = mlp_ratio
        self.linear_drop = linear_drop

        self.norm_1 = LayerNormalization(epsilon=1e-6)
        self.norm_2 = LayerNormalization(epsilon=1e-6)

        if self.ref_version == "v1":
            self.attn = MHSA(
                num_heads=num_heads,
                embedding_dim=embedding_dim,
                qkv_bias=qkv_bias,
                attention_drop=attention_drop,
                linear_drop=linear_drop,
            )
        elif self.ref_version == "v2":
            self.attn = LSA(
                embedding_dim=embedding_dim,
                qkv_bias=qkv_bias,
                attention_drop=attention_drop,
                linear_drop=linear_drop,
            )

        hidden_features = int(self.embedding_dim * self.mlp_ratio)

        self.mlp = Sequential(
            layers=[
                Dense(hidden_features, activation="swish"),
                Dropout(self.linear_drop),
                Dense(embedding_dim),
                Dropout(self.linear_drop),
            ]
        )

    def call(self, x):

        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "ref_version": self.ref_version,
                "embedding_dim": self.embedding_dim,
                "mlp_ratio": self.mlp_ratio,
                "linear_drop": self.linear_drop,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/modules/mobilevit_block.py#L186
def unfolding(
    x,
    B: int = 1,
    D: int = 144,
    patch_h: int = 2,
    patch_w: int = 2,
    num_patches_h: int = 10,
    num_patches_w: int = 10,
):
    """
    ### Notations (wrt paper) ###
        B/b = batch
        P/p = patch_size
        N/n = number of patches
        D/d = embedding_dim

    H, W
    [                            [
        [1, 2, 3, 4],     Goal      [1, 3, 9, 11],
        [5, 6, 7, 8],     ====>     [2, 4, 10, 12],
        [9, 10, 11, 12],            [5, 7, 13, 15],
        [13, 14, 15, 16],           [6, 8, 14, 16]
    ]                            ]
    """
    # [B, H, W, D] --> [B*nh, ph, nw, pw*D]
    reshaped_fm = tf.reshape(x, (B * num_patches_h, patch_h, num_patches_w, patch_w * D))

    # [B*nh, ph, nw, pw*D] --> [B*nh, nw, ph, pw*D]
    transposed_fm = tf.transpose(reshaped_fm, perm=[0, 2, 1, 3])

    # [B*nh, nw, ph, pw*D] --> [B, N, P, D]
    reshaped_fm = tf.reshape(transposed_fm, (B, num_patches_h * num_patches_w, patch_h * patch_w, D))

    # [B, N, P, D] --> [B, P, N, D]
    transposed_fm = tf.transpose(reshaped_fm, perm=[0, 2, 1, 3])

    return transposed_fm


# https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/modules/mobilevit_block.py#L233
def folding(
    x,
    B: int = 1,
    D: int = 144,
    patch_h: int = 2,
    patch_w: int = 2,
    num_patches_h: int = 10,
    num_patches_w: int = 10,
):
    """
    ### Notations (wrt paper) ###
        B/b = batch
        P/p = patch_size
        N/n = number of patches
        D/d = embedding_dim
    """
    # [B, P, N D] --> [B, N, P, D]
    x = tf.transpose(x, perm=(0, 2, 1, 3))

    # [B, N, P, D] --> [B*nh, nw, ph, pw*D]
    x = tf.reshape(x, (B * num_patches_h, num_patches_w, patch_h, patch_w * D))

    # [B*nh, nw, ph, pw*D] --> [B*nh, ph, nw, pw*D]
    x = tf.transpose(x, perm=(0, 2, 1, 3))

    # [B*nh, ph, nw, pw*D] --> [B, nh*ph, nw, pw, D] --> [B, H, W, C]
    x = tf.reshape(x, (B, num_patches_h * patch_h, num_patches_w * patch_w, D))

    return x


class MobileViT_v3_Block(Layer):
    def __init__(
        self,
        ref_version: str = "v1",
        out_filters: int = 64,
        embedding_dim: int = 90,
        transformer_repeats: int = 2,
        num_heads: Optional[int] = 4,
        patch_size: Optional[Union[int, tuple]] = (2, 2),
        attention_drop: float = 0.0,
        linear_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ref_version = ref_version
        self.out_filters = out_filters
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.transformer_repeats = transformer_repeats

        self.patch_size_h, self.patch_size_w = patch_size if isinstance(self.patch_size, tuple) else (self.patch_size // 2, self.patch_size // 2)
        self.patch_size_h, self.patch_size_w = tf.cast(self.patch_size_h, tf.int32), tf.cast(self.patch_size_w, tf.int32)

        # local_feature_extractor 1 and 2
        local_rep_layers = [
            DepthwiseConv2D(kernel_size=3, strides=1, padding="same", use_bias=False),
            BatchNormalization(),
            Activation("swish"),
            ConvLayer(num_filters=self.embedding_dim, kernel_size=1, strides=1, use_bn=False, use_activation=False, use_bias=False),
        ]
        self.local_rep = Sequential(layers=local_rep_layers)

        transformer_layers = [
            Transformer(
                ref_version=self.ref_version,
                num_heads=num_heads,
                embedding_dim=self.embedding_dim,
                linear_drop=linear_drop,
                attention_drop=attention_drop,
            )
            for _ in range(self.transformer_repeats)
        ]

        transformer_layers.append(LayerNormalization(epsilon=1e-6))

        # Repeated transformer blocks
        self.transformer_blocks = Sequential(layers=transformer_layers)

        self.concat = Concatenate(axis=-1)

        # Fusion blocks
        self.project = False

        if self.ref_version == "v1":
            self.conv_proj = ConvLayer(num_filters=self.out_filters, kernel_size=1, strides=1, use_bn=True, use_activation=True)
            self.project = True

        self.fusion = ConvLayer(
            num_filters=self.out_filters, kernel_size=1, strides=1, use_bn=True, use_activation=True if self.ref_version == "v1" else False
        )

    def call(self, x):
        local_representation = self.local_rep(x)

        # Transformer as Convolution Steps
        # --------------------------------
        # # Unfolding

        batch_size, fmH, fmW = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        num_patches_h = tf.math.floordiv(fmH, self.patch_size_h)
        num_patches_w = tf.math.floordiv(fmW, self.patch_size_w)

        unfolded = unfolding(
            local_representation,
            B=batch_size,
            D=self.embedding_dim,
            patch_h=self.patch_size_h,
            patch_w=self.patch_size_w,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )

        # # Infomation sharing/mixing --> global representation
        global_representation = self.transformer_blocks(unfolded)

        # # Folding
        folded = folding(
            global_representation,
            B=batch_size,
            D=self.embedding_dim,
            patch_h=self.patch_size_h,
            patch_w=self.patch_size_w,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
        )
        # # --------------------------------

        # New Fustion Block
        if self.project:
            folded = self.conv_proj(folded)

        fused = self.fusion(self.concat((folded, local_representation)))

        # For MobileViTv3: Skip connection
        final = x + fused

        return final

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_filters": self.out_filters,
                "embedding_dim": self.embedding_dim,
                "patch_size": self.patch_size,
                "transformer_repeats": self.transformer_repeats,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    batch = 2
    H = W = 32
    C = 96
    P = 2 * 2
    L = 4
    embedding_dim = 144

    mvitblk = MobileViT_v3_Block(
        ref_version="v1",
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
