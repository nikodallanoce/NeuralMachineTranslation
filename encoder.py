import tensorflow as tf
from tensorflow.keras import layers
from positional_embedding import PositionalEmbedding


class EncoderLayer(layers.Layer):

    def __init__(self, layers_size: int, dense_size: int, num_heads: int, dropout=0.1, **kwargs) -> None:
        super(EncoderLayer, self).__init__(**kwargs)

        self.layers_size = layers_size
        self.dense_size = dense_size
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads, layers_size, dropout=dropout)
        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_size, activation="elu"), layers.Dropout(dropout), layers.Dense(layers_size)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, mask=None) -> tf.Tensor:
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        else:
            print("Mask not built")
            assert False

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class EncoderTransformer(layers.Layer):

    def __init__(self,
                 num_layers: int,
                 layers_size: int,
                 dense_size: int,
                 num_heads: int,
                 max_length: int,
                 v_size_src: int,
                 dropout: float = 0.1) -> None:
        super(EncoderTransformer, self).__init__()

        self.layers_size = layers_size
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(max_length, v_size_src, layers_size)
        self.enc_layers = [EncoderLayer(layers_size, dense_size, num_heads) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, mask=None) -> tf.Tensor:
        src_embeddings = self.pos_embedding(inputs)
        enc_out = self.dropout(src_embeddings)
        for i in range(self.num_layers):
            enc_out = self.enc_layers[i](enc_out)

        return enc_out  # (batch_size, input_seq_len, layers_size)
