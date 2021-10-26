from utilities import positional_encoding
from layerTransformer import EncoderLayer
import tensorflow as tf
from transformers import TFBertModel


class EncoderRNN(tf.keras.layers.Layer):

    def __init__(self, src_vocab_size: int, layers_size: int) -> None:
        super(EncoderRNN, self).__init__()
        self.layers_size = layers_size
        self.src_vocab_size = src_vocab_size

        self.embedding = tf.keras.layers.Embedding(self.src_vocab_size, layers_size)
        self.gru = tf.keras.layers.GRU(self.layers_size, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, src_tokens: tf.Tensor, state=None) -> (tf.Tensor, tf.Tensor):
        # The embedding layer looks up the embedding for each token.
        embeddings = self.embedding(src_tokens)

        # The GRU processes the embedding sequence.
        output, state = self.gru(embeddings, initial_state=state)
        return output, state  # output shape: (batch, input_seq_length, layers_size), state shape: (batch, layers_size)


class EncoderTransformer(tf.keras.layers.Layer):

    def __init__(self, num_layers: int,
                 layers_size: int,
                 num_heads: int,
                 dff: int,
                 src_vocab_size: int,
                 maximum_position_encoding: int,
                 dropout: float = 0.1) -> None:
        super(EncoderTransformer, self).__init__()

        self.layers_size = layers_size
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(src_vocab_size, layers_size)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.layers_size)

        self.enc_layers = [EncoderLayer(layers_size, num_heads, dff, dropout) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, src_tokens: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(src_tokens)[1]

        # adding embedding and position encoding.
        x = self.embedding(src_tokens)  # (batch_size, input_seq_len, layers_size)
        x *= tf.math.sqrt(tf.cast(self.layers_size, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i].call(x, training, mask)

        return x  # (batch_size, input_seq_len, layers_size)


class EncoderBERT(tf.keras.layers.Layer):

    def __init__(self, bert: TFBertModel) -> None:
        super(EncoderBERT, self).__init__()
        self.bert = bert

    def call(self, src_tokens: tf.Tensor, training: bool, mask: tf.Tensor) -> tf.Tensor:
        mask = tf.ones(src_tokens.shape) - tf.cast(tf.math.equal(src_tokens, 0), tf.float32)
        output = self.bert([src_tokens, mask], training=training)[0]  # last_hidden_state
        return output  # (batch_size, input_seq_len, layers_size)
