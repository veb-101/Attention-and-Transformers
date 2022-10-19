import numpy as np
import tensorflow as tf

# from multihead_self_attention import MultiHeadSelfAttention
from multihead_self_attention_2 import MultiHeadSelfAttentionEinSum as MultiHeadSelfAttention
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

    n_classes: int
        Number of Classes.

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
        self.cls_token = self.add_weight(shape=(1, 1, self.embedding_dim), initializer=zeros_init, trainable=True, name="cls_token")
        self.pos_embed = self.add_weight(
            shape=(1, 1 + self.patch_embedding.n_patches, self.embedding_dim), initializer=zeros_init, trainable=True, name="position_emebedding"
        )

        # self.mlp_head = Sequential(layers=[LayerNormalization(epsilon=1e-6), Dense(n_classes)])

        hidden_features = int(embedding_dim * mlp_ratio)

        self.mlp_head = Sequential(
            layers=[
                Dense(hidden_features, activation="gelu"),
                Dropout(linear_drop),
                Dense(n_classes),
                # Dropout(linear_drop),
            ]
        )

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

        x = self.mlp_head(cls_token_final)

        return x


if __name__ == "__main__":

    # gpus = tf.config.list_physical_devices("GPU")
    # if gpus:
    #     tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=5292)])

    class Config:
        IMAGE_SIZE = 32

        EMBEDDING_DIM = 256
        MLP_RATIO = 2.0
        NUM_HEADS = 8
        DEPTH = 6
        PATCH_SIZE = 4

        N_CLASSES = 10
        LINEAR_DROP = 0.2
        ATTENTION_DROP = 0.0
        LEARNING_RATE = 3e-4

        BATCH_SIZE = 256
        NUM_EPOCHS = 100
        WEIGHT_DECAY = 0.01

    model = VisionTransformer(
        img_size=Config.IMAGE_SIZE,
        patch_size=Config.PATCH_SIZE,
        n_classes=Config.N_CLASSES,
        embedding_dim=Config.EMBEDDING_DIM,
        depth=Config.DEPTH,
        num_heads=Config.NUM_HEADS,
        mlp_ratio=Config.MLP_RATIO,
        linear_drop=Config.LINEAR_DROP,
        attention_drop=Config.ATTENTION_DROP,
    )

    a = tf.random.normal((1, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3))

    print(model(a).shape)
    model.summary()
