import tensorflow as tf
import keras


class Attention(keras.layers.Layer):

    def __init__(self, dec_layers_size: int, att_type: str):
        super(Attention, self).__init__()
        self.size = dec_layers_size
        self.att_type = att_type

    def call(self, dec_h_state: tf.Tensor, encoder_out: tf.Tensor = None, src_mask: tf.Tensor = None) -> tf.Tensor:
        att_score = tf.matmul(dec_h_state, encoder_out, transpose_b=True)
        att_score = tf.multiply(att_score, tf.cast(src_mask, tf.float32))
        context = keras.activations.softmax(att_score, axis=2)
        context = tf.matmul(context, encoder_out)
        return context
