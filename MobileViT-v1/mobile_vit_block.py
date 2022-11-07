import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dropout, Reshape, Dense, LayerNormalization, Concatenate

from BaseLayers import ConvLayer
from multihead_self_attention_2D import MultiHeadSelfAttentionEinSum2D as MHSA

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
        self.local_features_2 = ConvLayer(num_filters=self.embedding_dim, kernel_size=1, strides=1, use_bn=False, use_activation=False)

        # Repeated transformer blocks
        self.transformer_blocks = Sequential(
            layers=[
                Transformer(
                    embedding_dim=self.embedding_dim,
                    num_heads=self.num_heads,
                    linear_drop=linear_drop,
                    attention_drop=attention_drop,
                )
                for _ in range(self.transformer_repeats)
            ]
        )

        # Fusion blocks
        self.local_features_3 = ConvLayer(num_filters=self.out_filters, kernel_size=1, strides=1, use_bn=True, use_activation=True)
        self.concat = Concatenate(axis=-1)
        self.fuse_local_global = ConvLayer(num_filters=self.out_filters, kernel_size=3, strides=1, use_bn=True, use_activation=True)

    def call(self, x):
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        local_representation = self.local_features_1(x)
        local_representation = self.local_features_2(local_representation)

        # num_patches = tf.cast(tf.math.divide_no_nan(tf.shape(x)[1] * tf.shape(x)[2], self.patch_size), tf.int32)

        # Transformer as Convolution Steps
        # --------------------------------
        # # Unfolding
        unfolded = Reshape((self.patch_size, -1, self.embedding_dim))(local_representation)

        # # Infomation sharing/mixing --> global representation
        global_representation = self.transformer_blocks(unfolded)

        # # Folding
        folded = Reshape((H, W, self.embedding_dim))(global_representation)
        # --------------------------------

        # Fusion
        local_mix = self.local_features_3(folded)
        fusion = self.concat([local_mix, x])
        fusion = self.fuse_local_global(fusion)

        return fusion


if __name__ == "__main__":
    batch = 2
    H = W = 16
    C = 64
    P = 2 * 2
    L = 4
    embedding_dim = 80

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
