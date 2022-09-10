import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dropout, Dense, LayerNormalization
from tensorflow.keras import Model, Sequential

np.random.seed(1)
tf.random.set_seed(1)
tf.keras.utils.set_random_seed(1)


class PatchEmbedding(Layer):
    """
    Split image into patches and then embed them.

    Parameters
    ----------
    img_size: int
        Size of input image (square)

    patch_size: int
        Size of teh patch (square)

    embedding_dim: int
        Dimension of Patch embedding

    Attributes
    ----------

    n_patces: int
        Number of patches inside of our image

    proj: layers.Conv2D
        Convolutional layer that does both splitting image into patches
        and their embedding

    """

    def __init__(self, *, img_size, patch_size, embedding_dim=512, **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.n_patches = (img_size // self.patch_size) ** 2

        self.proj = Conv2D(filters=embedding_dim, kernel_size=self.patch_size, strides=self.patch_size)

    def call(self, x):
        """
        Run forward pass

        Parameters
        ----------
        x: tf.Tensor
            SHAPE: (B, img_size, img_size, in_channels)

        Returns
        -------
        tf.Tensor
            Shape: (B, n_patches, embedding_dim)
        """

        x = self.proj(x)  # Shape: (#B, #n_patches **0.5, n_patches **0.5, embedding_dim)
        x = tf.reshape(x, (-1, self.n_patches, self.embedding_dim))  # Shape: (#B, n_patches, embedding_dim)

        return x


class MultiHeadSelfAttention(Layer):
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

        weighted_values = tf.reshape(weighted_values, (batch_dims, input_dims, -1))
        # tf.print("Concatenating values from all heads:", tf.shape(weighted_values))

        mhsa_output = self.final_linear_project(weighted_values)
        mhsa_output = self.linear_dropout(mhsa_output)
        # tf.print("mhsa_output:", tf.shape(mhsa_output))

        return mhsa_output


class Block(Layer):
    """

    Parameters
    ----------

    num_heads: int
        Number of attention heads.

    embedding_dim: int
        Size of embedding dimension.

    qkv_bias: bool
        If True, then use bias in query, key and value projections

    mlp_ratio: float
        Determines the size of hidden dimension in MLP w.r.t embedding_dim

    linear_drop, attention_drop: float
        Dropout layers probability


    Atrributes
    ----------

    norm_1, norm_2: LayerNormalization
        The LayerNormalization layers.

    attn: MultiHeadSelfAttention
        MultiHeadSelfAttention block

    mlp: MLP
        multilayer perceptron block
    """

    def __init__(
        self,
        num_heads: int = 2,
        embedding_dim: int = 512,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        linear_drop: float = 0.2,
        attention_drop: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.norm_1 = LayerNormalization(epsilon=1e-6)
        self.norm_2 = LayerNormalization(epsilon=1e-6)

        self.attn = MultiHeadSelfAttention(
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            qkv_bias=qkv_bias,
            attention_drop=attention_drop,
            linear_drop=linear_drop,
        )

        hidden_features = int(embedding_dim * mlp_ratio)

        self.mlp = Sequential(
            layers=[
                Dense(hidden_features, activation="gelu"),
                Dropout(linear_drop),
                Dense(embedding_dim),
                Dropout(linear_drop),
            ]
        )

    def call(self, x):

        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))

        return x


class VisionTransformer(Model):
    """
    Parameters
    ----------

    img_size: int
        Height and width of image (square).

    patch_size: int
        Height and with of each patch (square)

    embedding_dim: int
        Dimensionality of token/patch embeddings.

    depth: int
        Number of transformer blocks.

    num_heads: int
        Number of attention heads.

    mlp_ratio: float
        Determines the hidden dimension of the 'MLP' module.

    qkv_bias: bool
        If True, include bias in query, key and value weight projection.

    linear_drop, attention_drop: float
        Dropout layers probability

    Attributes
    ----------

    patch_embed: PatchEmbedding
        Instance of PatchEmbedding layer.

    cls_token: tf.Variable
        Learnable parameter that will represetn the first token in the sequence.
        It has `embedding_dim` elements.

    pos_embed: tf.Variable
        Positional embedding of the cls token + all the patches
        It has '(n_patches + 1) * embedding_dim' elements.

    pos_drop: Dropout layer

    blocks: list
        List of 'Block' layers.

    norm: LayerNormalization
        The LayerNormalization layer.

    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        n_classes=10,
        embedding_dim=768,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        linear_drop=0,
        attention_drop=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_dim = embedding_dim

        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, embedding_dim=self.embedding_dim)

        self.pos_drop = Dropout(rate=linear_drop)

        self.blocks = []

        for idx in range(depth):
            self.blocks.append(
                Block(
                    num_heads=num_heads,
                    embedding_dim=self.embedding_dim,
                    qkv_bias=qkv_bias,
                    mlp_ratio=mlp_ratio,
                    linear_drop=linear_drop,
                    attention_drop=attention_drop,
                    name=f"AttentionBlock_{idx+1:>02}",
                )
            )
        zeros_init = tf.zeros_initializer()
        self.cls_token = self.add_weight(shape=(1, 1, self.embedding_dim), initializer=zeros_init, trainable=True)
        self.pos_embed = self.add_weight(shape=(1, 1 + self.patch_embedding.n_patches, self.embedding_dim), initializer=zeros_init, trainable=True)

        self.mlp_head = Sequential(layers=[LayerNormalization(epsilon=1e-6), Dense(n_classes)])

    def call(self, x):

        B = tf.shape(x)[0]
        x = self.patch_embedding(x)  # Shape: (B, n_patches, embedding_dim)

        cls_token = tf.broadcast_to(self.cls_token, (B, 1, self.embedding_dim))  # Shape: (B, 1, embedding_dim)

        x = tf.concat((cls_token, x), axis=1)  # Shape: (B, 1 + n_patches, embedding_dim)

        x = x + self.pos_embed  # Shape: (B, 1 + n_patches, embedding_dim)

        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        cls_token_final = tf.gather(x, indices=0, axis=1)  # Take only the class token

        tf.print(tf.shape(cls_token_final))

        x = self.mlp_head(cls_token_final)

        return x


if __name__ == "__main__":
    a = tf.random.normal((1, 224, 224, 3))

    model = VisionTransformer()
    model(a)

    model.summary()
