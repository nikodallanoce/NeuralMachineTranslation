from utilities import positional_encoding
from attention import *
from layerTransformer import EncoderLayer


class EncoderRNN(tf.keras.layers.Layer):

    def __init__(self,
                 d_model: int,
                 src_vocab_size: int,
                 dropout: float = 0.1):
        super(EncoderRNN, self).__init__()

        self.d_model = d_model

        self.embedding = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.rnn = tf.keras.layers.LSTM(d_model, return_sequences=True, return_state=True,)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training, h_state):
        x = self.embedding(x, training=training)  # (batch_size, input_seq_len, d_model)
        out, h_state, c_state = self.rnn(x, inintial_state=h_state)
        return out, h_state, c_state  # (batch_size, input_seq_len, d_model)


class EncoderTransformer(tf.keras.layers.Layer):

    def __init__(self, num_layers: int,
                 d_model: int,
                 num_heads: int,
                 dff: int,
                 src_vocab_size: int,
                 maximum_position_encoding: int,
                 dropout: float = 0.1):
        super(EncoderTransformer, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(src_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool, mask: tf.Tensor):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i].call(x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
