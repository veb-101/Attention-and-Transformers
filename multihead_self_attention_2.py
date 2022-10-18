import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, EinsumDense


np.random.seed(1)
tf.random.set_seed(1)
tf.keras.utils.set_random_seed(1)


class MultiHeadSelfAttentionEinSum(Layer):
    """
    Apply Multi-Head Self Attention.

    Parameters
    ----------
    num_heads: int = 2
        Number of heads

    embedding_dim: int = 64
        Size of embedding dimension i.e. D or Dmodel

    projection_dim: int = None
        Dimension for each head

    qkv_bias: bool = True
        Use bias in query, keys and values projection layer

    attention_drop: float = 0.2
        Dropout rate for the attention matrix

    linear_drop: float = 0.2
        Dropout rate for the final linear projection

    """

    def __init__(
        self,
        num_heads: int = 2,
        embedding_dim: int = 64,
        projection_dim: int = None,
        qkv_bias: bool = True,
        attention_drop: float = 0.2,
        linear_drop: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim if projection_dim else self.embedding_dim // self.num_heads
        self.scale = self.projection_dim**0.5
        self.qkv_bias = qkv_bias

        self.attn_dropout = Dropout(attention_drop)
        self.linear_dropout = Dropout(linear_drop)

        ##### Notations #####
        # B/b: batch
        # T/t: number of tokens
        # E/e: embedding dimension
        # N/n: number of heads
        # P/p: projection dimension

        # (B, T, E) * (E, N, P * 3) --> (B, N, P * 3)
        self.W_QKV = EinsumDense(
            "bte,enp->btnp",
            output_shape=[None, self.num_heads, 3 * self.projection_dim],
            bias_axes="np" if self.qkv_bias else None,
        )

        # (B, N, T, P) * (P, N, E) --> (B, T, E)
        self.Wo = EinsumDense("bntp,pne->bte", output_shape=[None, self.embedding_dim], bias_axes="e" if self.qkv_bias else None)

    def call(self, inputs):
        # inputs --> Shape: (B, T, E)

        output_tensor = self.W_QKV(inputs)  # Shape: (B, T, N, 3*P)
        output_tensor = tf.einsum("btnp->bntp", output_tensor)  # Shape: (B, N, T, 3*P)

        # Shape: (B, N, T, P) * 3
        q, k, v = tf.split(output_tensor, num_or_size_splits=3, axis=-1)

        # Shape: (B, N, T, P) * (B, N, T, P) --> (B, N, T, T)
        attention_matrix = tf.einsum("...ij, ...kj-> ...ik", q, k)

        # Shape: (B, N, T, T)
        attention_matrix = tf.nn.softmax(tf.math.divide(attention_matrix, self.scale), axis=-1)
        attention_matrix = self.attn_dropout(attention_matrix)

        # Shape: (B, N, T, T) * (B, N, T, P) --> (B, N, T, P)
        weighted_values = tf.einsum("...ij,...jk->...ik", attention_matrix, v)

        final = self.Wo(weighted_values)  # Shape: (B, N, T, P) * (P, N, E) --> (B, T, E)
        final = self.linear_dropout(final)

        return final

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embedding_dim": self.embedding_dim,
                "projection_dim": self.projection_dim,
                "qkv_bias": self.qkv_bias,
            }
        )
        return config


if __name__ == "__main__":

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=5292)])

    batch_dims = 2
    input_dims = 4
    embedding_dim = 64
    num_heads = 8
    use_bias = True

    proj_dim = embedding_dim // num_heads

    lal = MultiHeadSelfAttentionEinSum(
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        projection_dim=proj_dim,
        qkv_bias=use_bias,
        name="mhsa",
    )

    inputs = tf.random.normal((batch_dims, input_dims, embedding_dim))
    _ = lal(inputs)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(None, embedding_dim)))
    model.add(lal)
    # model.summary()

    print(model.count_params())

    layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=proj_dim, use_bias=use_bias)
    inputs = tf.keras.Input(shape=(None, embedding_dim))
    output_tensor = layer(inputs, inputs)
    model = tf.keras.Model(inputs, output_tensor)
    print(model.count_params())
    # model.summary()
