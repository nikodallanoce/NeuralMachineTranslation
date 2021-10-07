from utilities import positional_encoding
from attention import *
from layerTransformer import EncoderLayer


class EncoderRNN(tf.keras.layers.Layer):

    def __init__(self,
                 layers_size: int,
                 src_vocab_size: int,
                 dropout: float = 0.1):
        super(EncoderRNN, self).__init__()

        self.d_model = layers_size

        self.embedding = tf.keras.layers.Embedding(src_vocab_size, layers_size)
        self.rnn = tf.keras.layers.LSTM(layers_size, return_sequences=True, return_state=True,)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, src_tokens: tf.Tensor, training: bool, h_state: tf.Tensor):
        out = self.embedding(src_tokens, training=training)  # (batch_size, input_seq_len, layers_size)
        out, h_state, c_state = self.rnn(out, inintial_state=h_state)
        return out, h_state, c_state  # (batch_size, input_seq_len, layers_size)


class EncoderTransformer(tf.keras.layers.Layer):

    def __init__(self, num_layers: int,
                 layers_size: int,
                 num_heads: int,
                 dff: int,
                 src_vocab_size: int,
                 maximum_position_encoding: int,
                 dropout: float = 0.1):
        super(EncoderTransformer, self).__init__()

        self.layers_size = layers_size
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(src_vocab_size, layers_size)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.layers_size)

        self.enc_layers = [EncoderLayer(layers_size, num_heads, dff, dropout) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, src_tokens: tf.Tensor, training: bool, mask: tf.Tensor):
        seq_len = tf.shape(src_tokens)[1]

        # adding embedding and position encoding.
        x = self.embedding(src_tokens)  # (batch_size, input_seq_len, layers_size)
        x *= tf.math.sqrt(tf.cast(self.layers_size, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i].call(x, training, mask)

        return x  # (batch_size, input_seq_len, layers_size)
