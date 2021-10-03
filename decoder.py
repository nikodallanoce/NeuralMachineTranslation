import keras.activations
from keras.layers import Dropout, LSTM, Dense, GRU, Embedding
from attention import *


class DecoderRNN(keras.models.Model):

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
        out = keras.activations.relu(out)
        out = self.out_layer(out)
        return out, rnn_state, rnn_c


class DecoderTransformer(keras.models.Model):

    def get_config(self):
        pass

    def __init__(self,) -> None:
        super(DecoderTransformer, self).__init__()

    def call(self,
             dst_tokens: tf.Tensor,
             enc_out: tf.Tensor = None,
             src_mask: tf.Tensor = None,
             out_encoder: tf.Tensor = None):
        return None
