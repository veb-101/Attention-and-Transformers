import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout, Dense

np.random.seed(1)
tf.random.set_seed(1)
tf.keras.utils.set_random_seed(1)


class MultiHeadSelfAttention(Layer):
    """
    Apply Multi-Head Self Attention.

    Parameters
    ----------
    num_heads: int = 2,
    embedding_dim: int = 64,
    projection_dim: int = None,
    qkv_bias: bool = True,
    attention_drop: float = 0.2,
    linear_drop: float = 0.2,

    num_heads: int
        Number of heads

    embedding_dim: int
        Size of embedding dimension i.e. D or Dmodel

    projection_dim: int
        Dimension for each head

    qkv_bias: bool
        Use bias in query, keys and values projection layer

    attention_drop: float
        Dropout rate for the attention matrix

    linear_drop: float
        Dropout rate for the final linear projection

    Attributes
    ----------

    scale: float
        Square root of projection_dim


    heads_merge_dimension: int
        Number of total Dimension after merging feature vector output from each head

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

        self.attn_dropout = Dropout(attention_drop)
        self.linear_dropout = Dropout(linear_drop)

        self.scale = self.projection_dim**0.5

        self.qkv_W = Dense(units=3 * self.num_heads * self.projection_dim, name="W_expand_project", use_bias=qkv_bias)

        self.heads_merge_dimension = self.projection_dim * self.num_heads
        self.final_linear_project = Dense(units=self.embedding_dim, name="final_project")

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

        attention_matrix = tf.nn.softmax(q @ tf.transpose(k, perm=(0, 1, 3, 2)) / self.scale, axis=-1)
        # tf.print("attention_matrix:", tf.shape(attention_matrix))
        attention_matrix = self.attn_dropout(attention_matrix)

        weighted_values = attention_matrix @ v  # Shape: (#B, #heads, #tokens, #projection_dim)
        # tf.print("Reweighting inputs:", tf.shape(weighted_values))

        weighted_values = tf.transpose(weighted_values, perm=(0, 2, 1, 3))  # Shape: (#B, #tokens, #heads, #projection_dim)
        # tf.print("bring head and dim together", tf.shape(final_out))

        weighted_values = tf.reshape(weighted_values, (batch_dims, input_dims, self.heads_merge_dimension))
        # tf.print("Concatenating values from all heads:", tf.shape(weighted_values))

        mhsa_output = self.final_linear_project(weighted_values)
        mhsa_output = self.linear_dropout(mhsa_output)
        # tf.print("mhsa_output:", tf.shape(mhsa_output))

        return mhsa_output


if __name__ == "__main__":
    batch_dims = 2
    input_dims = 4
    embedding_dim = 64
    num_heads = 8

    proj_dim = embedding_dim // num_heads

    lal = MultiHeadSelfAttention(num_heads=num_heads, embedding_dim=embedding_dim, projection_dim=proj_dim, name="mhsa")

    inputs = tf.random.normal((batch_dims, input_dims, embedding_dim))
    _ = lal(inputs)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(None, embedding_dim)))
    model.add(lal)
    # model.summary()

    print(model.count_params())

    # layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=proj_dim, use_bias=False)
    # target = tf.keras.Input(shape=[4, embedding_dim])
    # # source = tf.keras.Input(shape=[4, 16])
    # output_tensor, weights = layer(target, target, return_attention_scores=True)

    # # print(len())

    layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=proj_dim)
    inputs = tf.keras.Input(shape=(None, embedding_dim))
    output_tensor = layer(inputs, inputs)
    model = tf.keras.Model(inputs, output_tensor)
    print(model.count_params())
    # model.summary()
