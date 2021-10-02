import keras
import tensorflow as tf
from tokenizer import TokenizerNMT


class TransformerNMT:

    def __init__(self,
                 encoder,
                 decoder: keras.models.Model,
                 tokenizer_src,
                 tokenizer_dst: TokenizerNMT,
                 lan_src: str = "english",
                 lan_dst: str = "italian"):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        self.lan_src = lan_src
        self.lan_dst = lan_dst

    def translate(self, src_sentence):
        src_tokens = self.tokenizer_src(src_sentence, add_special_tokens=True, truncation=True, padding="max_length",
                                        return_attention_mask=True, return_token_type_ids=True, return_tensors="tf",
                                        max_length=20)
        encoder_out = self.encoder(**src_tokens)
        encoder_last_h = tf.convert_to_tensor(encoder_out[0].numpy()[:, -1, :])
        encoder_out = encoder_out[0]
        att_mask = src_tokens.data["attention_mask"]
        decoder_input = tf.reshape(tf.constant(self.tokenizer_dst.word_index["start"]), (1, 1))
        translation = list()
        while True:
            decoder_out, decoder_h, decoder_c = self.decoder(tf.convert_to_tensor(decoder_input),
                                                             encoder_out, att_mask, [encoder_last_h, encoder_last_h])
            translation_token = int(tf.argmax(decoder_out, axis=1).numpy())
            translation_word = self.tokenizer_dst.index_word[translation_token]
            if translation_word == "end" or len(translation) > 10:
                break
            else:
                translation.append(translation_word)
                decoder_input = tf.reshape(tf.constant(translation_token), (1, 1))
                encoder_last_h = decoder_h

        out_translation = " ".join(translation)
        print("Source sentence: {0}\nTranslated as: {1}".format(src_sentence, out_translation))
        return out_translation
