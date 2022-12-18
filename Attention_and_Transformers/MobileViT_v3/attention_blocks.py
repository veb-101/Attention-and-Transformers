import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from tensorflow.keras.layers import EinsumDense


class LinearSelfAttention(Layer):
    def __init__(
        self,
        embedding_dim: int = 64,
        qkv_bias: bool = True,
        attention_drop: float = 0.2,
        linear_drop: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        ##### Notations (wrt paper) #####
        # B/b = batch
        # P/p = patch_size
        # N/n = number of patches
        # D/d = embedding_dim

        # self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.qkv_bias = qkv_bias

        self.use_attention_drop = attention_drop > 0.0
        self.use_linear_drop = linear_drop > 0.0

        if self.use_attention_drop:
            self.attn_dropout = Dropout(attention_drop)

        if self.use_linear_drop:
            self.linear_dropout = Dropout(linear_drop)

        # Shape: (B, P, N, D) * (D, 1 + (2 * D)) --> (B, P, N, 1 + (2 * D))
        self.W_QKV = EinsumDense(
            "bpnd,de->bpne",
            output_shape=[None, None, 1 + (2 * self.embedding_dim)],
            bias_axes="e" if self.qkv_bias else None,
        )

        # Shape: (B, P, N, D) * (D, D) --> (B, P, N, D)
        self.Wo = EinsumDense("bpnd,de->bpne", output_shape=[None, None, self.embedding_dim], bias_axes="e" if self.qkv_bias else None)

    def call(self, inputs):
        """
        From paper repo:

        "For MobileViTv3, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer."

        In TensorFlow as channel C is already the last dimension, I'm using the EinsumDense layer.
        """

        # Inputs Shape --> (B, P, N, D)

        qkv = self.W_QKV(inputs)  # Shape: (B, P, N, (1 + 2*D))

        # Shape: (B, P, N, 1), (B, P, N, D), (B, P, N, D)
        q, k, v = tf.split(qkv, [1, self.embedding_dim, self.embedding_dim], axis=-1)
        v = tf.nn.relu(v)

        # Apply softmax along N dimension
        context_scores = tf.nn.softmax(q, axis=2)

        if self.use_attention_drop:
            context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, P, N, d] x [B, P, N, 1] -> [B, P, N, d]
        context_vector = k * context_scores

        # [B, P, N, d] --> [B, P, 1, d]
        context_vector = tf.math.reduce_sum(context_vector, axis=2, keepdims=True)

        # Combine context vector with values
        # [B, P, N, d] * [B, P, 1, d] --> [B, P, N, d]
        # [B, P, 1, d] ---expand---> [B, P, N, d]
        updated_values = tf.einsum("...nd, ...kd->...nd", v, context_vector)

        # Shape: (B, P, N, D) * (D, ) --> (B, P, N, D)
        final = self.Wo(updated_values)

        if self.use_linear_drop:
            final = self.linear_dropout(final)

        return final

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "projection_dim": self.projection_dim,
                "qkv_bias": self.qkv_bias,
                "attention_drop": self.attention_drop,
                "linear_drop": self.linear_drop,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiHeadSelfAttentionEinSum2D(Layer):
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

        ##### Notations (wrt paper) #####
        # B/b = batch
        # P/p = patch_size
        # N/n = number of patches
        # D/d = embedding_dim
        # H/h = num_heads
        # E/e = projection_dim

        # New Shape: (B, P, N, D) * (D, H, E * 3) --> (B, P, H, N, E * 3)
        self.W_QKV = EinsumDense(
            "bpnd,dhe->bphne",
            output_shape=[None, self.num_heads, None, 3 * self.projection_dim],
            bias_axes="he" if self.qkv_bias else None,
        )

        # Shape: (B, P, H, N, E) * (E, H, D) --> (B, P, N, D)
        self.Wo = EinsumDense("bphne,ehd->bpnd", output_shape=[None, None, embedding_dim], bias_axes="d" if self.qkv_bias else None)

    def call(self, inputs):
        """
        In the attention matrix, dot product among patches only occur among pixels at similar positions.
        So the first pixel in each patch, only attends to the pixel at similar position in other patches.
        For visual reference, check the diagram on page 5 and 17 in the paper.
        """

        # Inputs Shape --> (B, P, N, D)

        output_tensor = self.W_QKV(inputs)  # Shape: (B, P, H, N, E * 3)

        # Shape: (B, P, H, N, E) * 3
        q, k, v = tf.split(output_tensor, num_or_size_splits=3, axis=-1)

        # Shape: (B, P, H, N, E) * (B, P, H, N, E) --> (B, P, H, N, N)
        attention_matrix = tf.einsum("...ij, ...kj-> ...ik", q, k)

        # Shape: (B, P, H, N, N)
        attention_matrix = tf.nn.softmax(tf.math.divide(attention_matrix, self.scale), axis=-1)

        if self.use_attention_drop:
            attention_matrix = self.attn_dropout(attention_matrix)

        # Shape: (B, P, H, N, N) * (B, P, H, N, E) --> (B, P, H, N, E)
        weighted_values = tf.einsum("...ij,...jk->...ik", attention_matrix, v)

        final = self.Wo(weighted_values)  # Shape: (B, P, H, N, E) * (E, H, D) --> (B, P, N, D)

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
                "attention_drop": self.attention_drop,
                "linear_drop": self.linear_drop,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":

    batch_dims = 1
    P = 4
    N = 256
    embedding_dim = 64
    use_bias = True

    lal = LinearSelfAttention(
        embedding_dim=embedding_dim,
        qkv_bias=use_bias,
        name="LSA",
    )

    inputs = tf.random.normal((batch_dims, P, N, embedding_dim))
    _ = lal(inputs)

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=(None, None, embedding_dim)))
    model.add(lal)

    # model.summary()
    print(model.count_params())
