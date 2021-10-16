import tensorflow.keras.activations
from tensorflow.keras.layers import Dropout, LSTM, Dense, GRU, Embedding
from attention import *
from utilities import *
from layerTransformer import DecoderLayer


class DecoderRNN(tensorflow.keras.models.Model):

    def get_config(self):
        pass

    def __init__(self,
                 rnn_type: str = "lstm",
                 emb_size: int = 768,
                 layers_size: int = 768,
                 v_size: int = 0,
                 emb_dropout: float = 0.1,
                 layers_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 freeze_parameters: bool = False) -> None:
        """
        Setup the rnn decoder
        :param rnn_type: the type of RNN, LSTM or GRU
        :param layers_size: size of the rnn layers (number of units)
        :param v_size: size of the target language
        :param emb_dropout: dropout applied to the last encoder hidden state
        :param layers_dropout: dropout applied beetween each rnn layer
        :param att_dropout: dropout applied after the attention mechanism
        :param freeze_parameters: if the parameters should be frozen
        """
        super(DecoderRNN, self).__init__()

        self.layers_size = layers_size
        self.emb_layer = Embedding(v_size, emb_size)
        self.emb_dropout = Dropout(emb_dropout)

        if rnn_type == "lstm":
            decoder = LSTM
        elif rnn_type == "gru":
            decoder = GRU
        else:
            raise ValueError("The decoder must be LSTM or GRU")

        self.decoder = decoder(layers_size, dropout=layers_dropout, return_sequences=True, return_state=True)
        self.attention = Attention(layers_size, "general")
        self.att_dropout = Dropout(att_dropout)
        self.out_layer = Dense(v_size, activation="softmax")
        self.freeze_parameters = freeze_parameters

    def call(self,
             dst_tokens: tf.Tensor,
             encoder_out: tf.Tensor = None,
             src_mask: tf.Tensor = None,
             h_state: list = None) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """
        Performs the decoding of the encoder output
        :param dst_tokens: the tokens of the target sentence
        :param encoder_out: hidden states of the encoder
        :param src_mask: attention mask for the source sentence
        :param h_state: last hidden state of the decoder
        :return: tensor with softmax probabilities
        """
        # Build the embeddings
        emb_dst = self.emb_layer(dst_tokens)

        # Decoder step, token by token
        _, rnn_state, rnn_c = self.decoder(emb_dst, initial_state=h_state)
        context = self.attention(rnn_state, encoder_out, src_mask)
        context = self.att_dropout(context)

        # Concatenate decoder hidden state with attention scores
        out = tf.concat([context, rnn_state], 1)

        # Generate the token probabilities
        out = tf.keras.activations.relu(out)
        out = self.out_layer(out)
        return out, rnn_state, rnn_c


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
