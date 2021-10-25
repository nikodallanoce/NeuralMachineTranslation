from utilities import positional_encoding
from layerTransformer import DecoderLayer
import tensorflow as tf


class DecoderRNN(tf.keras.layers.Layer):

    def __init__(self, dst_v_size: int, layers_size: int) -> None:
        super(DecoderRNN, self).__init__()
        self.dst_v_size = dst_v_size
        self.layers_size = layers_size

        self.embedding = tf.keras.layers.Embedding(dst_v_size, layers_size)

        self.gru = tf.keras.layers.GRU(layers_size, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.attention = tf.keras.layers.Attention(layers_size)

        self.Wc = tf.keras.layers.Dense(layers_size, activation=tf.math.tanh)

        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    def call(self,
             dst_tokens: tf.Tensor,
             enc_output: tf.Tensor,
             mask: tf.Tensor,
             state=None):

        embeddings = self.embedding(dst_tokens)
        rnn_output, state = self.gru(embeddings, initial_state=state)
        context_vector, attention_weights = self.attention(query=rnn_output, value=enc_output, mask=mask)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)
        attention_vector = self.Wc(context_and_rnn_output)
        logits = self.fc(attention_vector)
        return logits, attention_weights


class DecoderTransformer(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers: int,
                 layers_size: int,
                 num_heads: int,
                 dff: int,
                 target_vocab_size: int,
                 maximum_position_encoding: int, dropout=0.1) -> None:
        super(DecoderTransformer, self).__init__()

        self.layers_size = layers_size
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, layers_size)
        self.pos_encoding = positional_encoding(maximum_position_encoding, layers_size)

        self.dec_layers = [DecoderLayer(layers_size, num_heads, dff, dropout) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self,
             dst_tokens: tf.Tensor,
             enc_output: tf.Tensor,
             training: bool,
             look_ahead_mask: tf.Tensor,
             padding_mask: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        seq_len = tf.shape(dst_tokens)[1]
        attention_weights = {}

        x = self.embedding(dst_tokens)  # (batch_size, target_seq_len, layers_size)
        x *= tf.math.sqrt(tf.cast(self.layers_size, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        return x, attention_weights
