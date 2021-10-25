import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 layers_size: int,
                 num_heads: int,
                 dff: int,
                 dropout: float = 0.1) -> None:
        super(EncoderLayer, self).__init__()

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, layers_size)

        self.ffn = tf.keras.Sequential([
              tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
              tf.keras.layers.Dense(layers_size)  # (batch_size, seq_len, d_model)
            ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, src_tokens: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        attn_output, _ = self.mha.call(src_tokens, src_tokens, src_tokens, mask, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)  # (batch_size, input_seq_len, layers_size)
        out1 = self.layernorm1(src_tokens + attn_output)  # (batch_size, input_seq_len, layers_size)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, layers_size)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, layers_size)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self,
                 layers_size: int,
                 num_heads: int,
                 dff: int,
                 dropout=0.1) -> None:
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads, layers_size)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads, layers_size)

        self.ffn = tf.keras.Sequential([
              tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
              tf.keras.layers.Dense(layers_size)  # (batch_size, seq_len, d_model)
            ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self,
             dst_tokens: tf.Tensor,
             enc_output: tf.Tensor,
             training: bool,
             look_ahead_mask: tf.Tensor,
             padding_mask: tf.Tensor):
        # enc_output.shape == (batch_size, input_seq_len, layers_size)

        attn1, attn_weights_block1 = self.mha1(dst_tokens, dst_tokens, dst_tokens,
                                               look_ahead_mask,return_attention_scores=True)
        attn1 = self.dropout1(attn1, training=training)  # (batch_size, target_seq_len, layers_size)
        out1 = self.layernorm1(attn1 + dst_tokens)

        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask, return_attention_scores=True)
        attn2 = self.dropout2(attn2, training=training)  # (batch_size, target_seq_len, layers_size)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, layers_size)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, layers_size)

        return out3, attn_weights_block1, attn_weights_block2
