import numpy as np
import tensorflow as tf


class Translator:

    def __init__(self, tokenizer_src, tokenizer_dst, max_length: int, transformer: tf.keras.Model) -> None:
        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        self.max_length = max_length
        self.transformer = transformer

    def translate(self, input_sentence: str) -> (list, str):
        tokenized_input_sentence = self.tokenizer_src(input_sentence, return_tensors='tf', add_special_tokens=True,
                                                      max_length=self.max_length, padding='max_length',
                                                      truncation=True).data["input_ids"]
        decoded_sentence = "[CLS]"
        list_tokens = [decoded_sentence]
        for i in range(self.max_length):
            decoded_sentence = self.tokenizer_dst.convert_tokens_to_string(list_tokens)
            tokenized_dst_sentence = self.tokenizer_dst(decoded_sentence, return_tensors='tf', add_special_tokens=False,
                                                        max_length=self.max_length,
                                                        padding='max_length').data['input_ids']
            predictions = self.transformer([tokenized_input_sentence, tokenized_dst_sentence])
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = self.tokenizer_dst.ids_to_tokens[sampled_token_index]

            if sampled_token == "[SEP]":
                decoded_sentence = self.tokenizer_dst.convert_tokens_to_string(list_tokens[1:])
                break

            list_tokens.append(sampled_token)

        return list_tokens, decoded_sentence
