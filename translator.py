import numpy as np
from queue import SimpleQueue
import tensorflow as tf


class Translator(tf.Module):

    def __init__(self, tokenizer_src, tokenizer_dst, max_length: int, transformer: tf.keras.Model) -> None:
        super(Translator, self).__init__()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        self.max_length = max_length
        self.transformer = transformer

    def __translate(self, input_sentence: str) -> (list, str):
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

    def __translate_beam_search(self, input_sentence: str, k=2):
        tokenized_input_sentence = self.tokenizer_src(input_sentence, return_tensors='tf', add_special_tokens=True,
                                                      max_length=self.max_length, padding='max_length',
                                                      truncation=True).data["input_ids"]
        decoded_sentence = (["[CLS]"], 1)
        beam_queue = SimpleQueue()
        beam_queue.put(decoded_sentence)
        translated = list()
        i = 0
        while not beam_queue.empty() and i < self.max_length:
            tokenized_sentence, prb = beam_queue.get()
            decoded_sentence = self.tokenizer_dst.convert_tokens_to_string(tokenized_sentence)
            tokenized_dst_sentence = self.tokenizer_dst(decoded_sentence, return_tensors='tf', add_special_tokens=False,
                                                        max_length=self.max_length,
                                                        padding='max_length').data['input_ids']
            predictions = self.transformer([tokenized_input_sentence, tokenized_dst_sentence])
            i = len(tokenized_sentence) - 1
            sampled_token_indexes = np.argsort(predictions[0, i, :])[-k:]
            probabilities = [float(predictions[0, i, j]) for j in sampled_token_indexes]
            for samp_index, p in zip(sampled_token_indexes, probabilities):
                sampled_token = self.tokenizer_dst.ids_to_tokens[samp_index]
                tok_sent_with_new_samp = tokenized_sentence.copy()
                tok_sent_with_new_samp.append(sampled_token)
                next_sent = (tok_sent_with_new_samp, p * prb)
                if sampled_token == "[SEP]":  # and next_sent[1] > 0.02:
                    translated.append(next_sent)
                elif next_sent[1] > 0.001:
                    beam_queue.put(next_sent)

        translated.sort(key=lambda x: x[1], reverse=True)
        return translated

    def __call__(self, input_sentence: str, k=0):
        if k != 0:
            out_translation = self.__translate(input_sentence)
        else:
            out_translation = self.__translate_beam_search(input_sentence, k)

        return out_translation
