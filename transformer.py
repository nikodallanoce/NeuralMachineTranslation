from utilities import *


class TransformerNMT(tf.keras.Model):

    def __init__(self,
                 encoder: tf.keras.layers.Layer,
                 decoder: tf.keras.layers.Layer,
                 dst_v_size: int,
                 lan_src: str = "english",
                 lan_dst: str = "italian") -> None:
        super(TransformerNMT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lan_src = lan_src
        self.lan_dst = lan_dst
        self.final_layer = tf.keras.layers.Dense(dst_v_size)

    @staticmethod
    def __create_masks(src: tf.Tensor, dst: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(src)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(src)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(dst)[1])
        dec_target_padding_mask = create_padding_mask(dst)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask

    def call(self, inputs: list, training: bool) -> (tf.Tensor, tf.Tensor):
        # Keras models prefer if you pass all your inputs in the first argument
        src, dst = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.__create_masks(src, dst)

        enc_output = self.encoder(src, training, enc_padding_mask)  # (batch_size, inp_seq_len, layers_size)

        dec_output, attention_weights = self.decoder(dst, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights
