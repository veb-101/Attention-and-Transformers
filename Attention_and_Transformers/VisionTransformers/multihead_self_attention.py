import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Dense
from tensorflow.keras.layers import EinsumDense


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

        self.use_attention_drop = attention_drop > 0.0
        self.use_linear_drop = linear_drop > 0.0

        if self.use_attention_drop:
            self.attn_dropout = Dropout(attention_drop)

        if self.use_linear_drop:
            self.linear_dropout = Dropout(linear_drop)

        ##### Notations #####
        # B/b: batch
        # T/t: number of tokens
        # E/e: embedding dimension
        # N/n: number of heads
        # P/p: projection dimension

        # Shape:(B, T, E) * (E, N, P * 3) --> (B, T, N, P * 3)
        # self.W_QKV = EinsumDense(
        #     "bte,enp->btnp",
        #     output_shape=[None, self.num_heads, 3 * self.projection_dim],
        #     bias_axes="np" if self.qkv_bias else None,
        # )

        # New Shape: (B, T, E) * (E, N, P * 3) --> (B, N, T, P * 3)
        self.W_QKV = EinsumDense(
            "bte,enp->bntp",
            output_shape=[self.num_heads, None, 3 * self.projection_dim],
            bias_axes="np" if self.qkv_bias else None,
        )

        # Shape: (B, N, T, P) * (P, N, E) --> (B, T, E)
        self.Wo = EinsumDense("bntp,pne->bte", output_shape=[None, self.embedding_dim], bias_axes="e" if self.qkv_bias else None)

    def call(self, inputs):
        # inputs --> Shape: (B, T, E)

        # Old -- Transpose dimensions after operation
        # output_tensor = self.W_QKV(inputs)  # Shape: (B, T, N, P * 3)
        # output_tensor = tf.einsum("btnp->bntp", output_tensor)  # Shape: (B, N, T, P * 3)

        # New -- Simulatenously transpose the dimension
        output_tensor = self.W_QKV(inputs)  # Shape: (B, N, T, P * 3)

        # Shape: (B, N, T, P) * 3
        q, k, v = tf.split(output_tensor, num_or_size_splits=3, axis=-1)

        # Shape: (B, N, T, P) * (B, N, T, P) --> (B, N, T, T)
        attention_matrix = tf.einsum("...ij, ...kj-> ...ik", q, k)

        # Shape: (B, N, T, T)
        attention_matrix = tf.nn.softmax(tf.math.divide(attention_matrix, self.scale), axis=-1)
        if self.use_attention_drop:
            attention_matrix = self.attn_dropout(attention_matrix)

        # Shape: (B, N, T, T) * (B, N, T, P) --> (B, N, T, P)
        weighted_values = tf.einsum("...ij,...jk->...ik", attention_matrix, v)

        final = self.Wo(weighted_values)  # Shape: (B, N, T, P) * (P, N, E) --> (B, T, E)
        if self.use_linear_drop:
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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiHeadSelfAttention_basic(Layer):
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
        self.heads_merge_dimension = self.projection_dim * self.num_heads
        self.qkv_bias = qkv_bias

        self.qkv_W = Dense(units=3 * self.num_heads * self.projection_dim, name="W_expand_project", use_bias=self.qkv_bias)
        self.final_linear_project = Dense(units=self.embedding_dim, name="final_project", use_bias=self.qkv_bias)

        self.use_attention_drop = attention_drop > 0.0
        self.use_linear_drop = linear_drop > 0.0

        if self.use_attention_drop:
            self.attn_dropout = Dropout(attention_drop)

        if self.use_linear_drop:
            self.linear_dropout = Dropout(linear_drop)

    def call(self, input_mat):

        batch_dims = tf.shape(input_mat)[0]
        input_dims = tf.shape(input_mat)[1]

        Q_K_V = self.qkv_W(input_mat)  # Shape: (#B, #tokens, #projection_dim *3)
        # tf.print(tf.shape(Q_K_V))
        # Shape: (#B, #tokens, #heads, #projection_dim *3)
        Q_K_V_reshape = tf.reshape(Q_K_V, (batch_dims, input_dims, self.num_heads, 3 * self.projection_dim))
        # tf.print(tf.shape(Q_K_V_reshape))

        Q_K_V_transpose = tf.transpose(Q_K_V_reshape, perm=(0, 2, 1, 3))  # Shape: (#B, #heads, #tokens, #projection_dim * 3)
        # tf.print(tf.shape(Q_K_V_transpose))

        q, k, v = tf.split(Q_K_V_transpose, num_or_size_splits=3, axis=-1)
        # print("q:", q.shape, "k:", k.shape, "v:", v.shape)

        # Shape: (#B, #heads, #tokens, #tokens)
        attention_matrix = tf.nn.softmax(q @ tf.transpose(k, perm=(0, 1, 3, 2)) / self.scale, axis=-1)
        # tf.print("attention_matrix:", tf.shape(attention_matrix))
        if self.use_attention_drop:
            attention_matrix = self.attn_dropout(attention_matrix)

        weighted_values = attention_matrix @ v  # Shape: (#B, #heads, #tokens, #projection_dim)
        # tf.print("Reweighting inputs:", tf.shape(weighted_values))

        weighted_values = tf.transpose(weighted_values, perm=(0, 2, 1, 3))  # Shape: (#B, #tokens, #heads, #projection_dim)
        # tf.print("bring head and dim together", tf.shape(final_out))

        weighted_values = tf.reshape(weighted_values, (batch_dims, input_dims, self.heads_merge_dimension))
        # tf.print("Concatenating values from all heads:", tf.shape(weighted_values))

        mhsa_output = self.final_linear_project(weighted_values)
        if self.use_linear_drop:
            final = self.linear_dropout(final)
        # tf.print("mhsa_output:", tf.shape(mhsa_output))

        return mhsa_output

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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    batch_dims = 2
    input_dims = 4
    embedding_dim = 64
    num_heads = 8
    use_bias = False

    proj_dim = embedding_dim // num_heads

    print("Normal")
    lal = MultiHeadSelfAttention_basic(num_heads=num_heads, embedding_dim=embedding_dim, projection_dim=proj_dim, qkv_bias=use_bias, name="mhsa")

    inputs = tf.random.normal((batch_dims, input_dims, embedding_dim))
    _ = lal(inputs)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(None, embedding_dim)))
    model.add(lal)
    # model.summary()

    print(model.count_params())

    print("Einsum")
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

    print("Tensorflow default")
    layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=proj_dim, use_bias=use_bias)
    inputs = tf.keras.Input(shape=(None, embedding_dim))
    output_tensor = layer(inputs, inputs)
    model = tf.keras.Model(inputs, output_tensor)
    print(model.count_params())
    # model.summary()
