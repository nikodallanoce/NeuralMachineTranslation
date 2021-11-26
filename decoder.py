import tensorflow as tf
from tensorflow.keras import layers
from positional_embedding import PositionalEmbedding


class DecoderLayer(layers.Layer):

    def __init__(self, layers_size: int, dense_size: int, num_heads: int, dropout=0.1, **kwargs) -> None:
        super(DecoderLayer, self).__init__(**kwargs)

        self.layers_size = layers_size
        self.dense_size = dense_size
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads, layers_size, dropout=dropout)
        self.attention_2 = layers.MultiHeadAttention(num_heads, layers_size, dropout=dropout)
        self.dense_proj = tf.keras.Sequential(
            [layers.Dense(dense_size, activation="elu"), layers.Dropout(dropout), layers.Dense(layers_size)]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.dropout_1 = layers.Dropout(dropout)
        self.dropout_2 = layers.Dropout(dropout)
        self.dropout_3 = layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, encoder_outputs: tf.Tensor, mask=None) -> tf.Tensor:
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            assert False

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)


class DecoderTransformer(layers.Layer):

    def __init__(self,
                 num_layers: int,
                 layers_size: int,
                 dense_size: int,
                 num_heads: int,
                 max_length: int,
                 v_size_dst: int,
                 dropout=0.1) -> None:
        super(DecoderTransformer, self).__init__()

        self.layers_size = layers_size
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(max_length, v_size_dst, layers_size)
        self.dec_layers = [DecoderLayer(layers_size, dense_size, num_heads) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout)
        self.supports_masking = True

    def call(self, inputs: tf.Tensor, enc_output: tf.Tensor, mask=None) -> tf.Tensor:
        dst_embeddings = self.pos_embedding(inputs)
        dec_output = self.dropout(dst_embeddings)
        for i in range(self.num_layers):
            dec_output = self.dec_layers[i](dec_output, enc_output)

        return dec_output
