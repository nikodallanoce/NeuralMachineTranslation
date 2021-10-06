import keras
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer, AutoTokenizer
from tqdm import tqdm


class TransformerNMT:

    def __init__(self,
                 encoder: TFBertModel,
                 decoder: keras.models.Model,
                 tokenizer_src: BertTokenizer,
                 tokenizer_dst: AutoTokenizer,
                 lan_src: str = "english",
                 lan_dst: str = "italian"):
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        self.lan_src = lan_src
        self.lan_dst = lan_dst

    def __train_step(self, encoded_src, dst, att_mask):
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        with tf.GradientTape() as tape:
            h_state = tf.expand_dims(encoded_src[-1], 0)
            c_state = tf.expand_dims(encoded_src[-1], 0)
            dec_tokens = list()
            for i, token in enumerate(dst):
                token = tf.reshape(token, (1, 1))
                dec_out, h_state, c_state = self.decoder(token, encoded_src, att_mask, [h_state, c_state])
                dec_tokens.append(dec_out)

            dec_tokens = tf.convert_to_tensor(dec_tokens)
            loss = crossentropy(dst, dec_tokens, sample_weight=att_mask)

        tr_p = self.decoder.trainable_variables
        gradients = tape.gradient(loss, tr_p)
        optimizer = keras.optimizers.Adam()
        optimizer.apply_gradients(zip(gradients, tr_p))
        return loss/len(dst)

    def train(self, tr_set_src, tr_set_dst, epochs: int):
        losses = list()
        # Tokenize the dataset
        src_dataset_tokenized = self.tokenizer_src(tr_set_src, add_special_tokens=True,
                                                   truncation=True, padding="max_length", return_attention_mask=True,
                                                   return_tensors="tf", max_length=30)
        dst_dataset_tokenized = self.tokenizer_src(tr_set_dst, add_special_tokens=True,
                                                   truncation=True, padding="max_length", return_attention_mask=True,
                                                   return_tensors="tf", max_length=30)

        att_masks: tf.Tensor = src_dataset_tokenized.data["attention_mask"]  # Source sentences attention masks
        tokens_dst: tf.Tensor = dst_dataset_tokenized.data["input_ids"]  # Target sentences tokens
        enc_out = self.encoder(**src_dataset_tokenized)[0]  # Encode source sentences
        for epoch in tqdm(range(epochs)):
            loss = 0
            for i in range(len(enc_out)):
                loss += self.__train_step(enc_out[i], tokens_dst[i], tf.expand_dims(att_masks[i], 0))

            loss /= len(enc_out)
            losses.append(loss)
        return losses

    def translate(self, src_sentences):
        # Encode the source sentences
        src_tokens = self.tokenizer_src(src_sentences, add_special_tokens=True,
                                        truncation=True, padding="max_length", return_attention_mask=True,
                                        return_tensors="tf", max_length=30)
        att_masks = src_tokens.data["attention_mask"]
        enc_out = self.encoder(**src_tokens)[0]
        decoder_input = self.tokenizer_dst.decode["[CLS]"]
        translations = list()
        for i in range(len(src_tokens)):
            att_mask = tf.expand_dims(att_masks[i], 0)
            h_state = tf.expand_dims(enc_out[i, -1], 0)
            c_state = tf.expand_dims(enc_out[i, -1], 0)
            translation = list()
            while True:
                decoder_out, decoder_h, decoder_c = self.decoder(tf.convert_to_tensor(decoder_input),
                                                                 enc_out, att_mask, [h_state, c_state])
                translation_token = int(tf.argmax(decoder_out, axis=1).numpy())
                translation_word = self.tokenizer_dst.index_word[translation_token]
                if translation_word == "[SEP]" or len(translation) > 30:
                    break
                else:
                    translation.append(translation_word)
                    decoder_input = tf.reshape(tf.constant(translation_token), (1, 1))
                    h_state = decoder_h
                    c_state = decoder_c

            out_translation = " ".join(translation)
            translations.append(out_translation)
            print("Source sentence: {0}\nTranslated as: {1}\n".format(src_sentences[i], out_translation))

        return translations
