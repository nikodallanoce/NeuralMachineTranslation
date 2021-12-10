import numpy as np
from queue import Queue
import tensorflow as tf


class Translator(tf.Module):

    def __init__(self, tokenizer_src, tokenizer_dst, max_length: int, transformer: tf.keras.Model) -> None:
        super(Translator, self).__init__()
        self.tokenizer_src = tokenizer_src
        self.tokenizer_dst = tokenizer_dst
        self.max_length = max_length
        self.transformer = transformer

    def translate(self, input_sentence: str, k: int = 1) -> (list, str):
        tokenized_input_sentence = self.tokenizer_src(input_sentence, return_tensors='tf', add_special_tokens=True,
                                                      max_length=self.max_length, padding='max_length',
                                                      truncation=True).data["input_ids"]
        decoded_sentence = "[CLS]"
        list_tokens = [decoded_sentence]
        for i in range(self.max_length):
            decoded_sentence = self.tokenizer_dst.convert_tokens_to_string(list_tokens)
            tokenized_dst_sentence = self.tokenizer_dst(decoded_sentence, return_tensors='tf', add_special_tokens=False,
                                                        max_length=self.max_length,
                                                        padding='max_length', truncation=True).data['input_ids']
            predictions = self.transformer([tokenized_input_sentence, tokenized_dst_sentence])
            sampled_token_indexes = np.argsort(predictions[0, i, :])[-k:]
            p = [float(predictions[0, i, j]) for j in sampled_token_indexes]
            p = np.array(p)
            p /= np.sum(p)
            sampled_token_index = np.random.choice(sampled_token_indexes, 1, p=p)
            sampled_token = self.tokenizer_dst.ids_to_tokens[sampled_token_index[0]]
            if sampled_token == "[SEP]":
                decoded_sentence = self.tokenizer_dst.convert_tokens_to_string(list_tokens[1:])
                break

            list_tokens.append(sampled_token)

        return list_tokens, decoded_sentence

    def translate_beam_pruning(self, input_sentence, k=2, sequence_length=90):
        tokenized_input_sentence = self.tokenizer_src(input_sentence, return_tensors='tf', add_special_tokens=True,
                                                      max_length=sequence_length,
                                                      padding='max_length', truncation=True).data["input_ids"]
        decoded_sentence = (["[CLS]"], 1)
        beam_queue = Queue(2 ** 8)
        beam_queue.put(decoded_sentence)
        translated = []
        i = 0

        while not beam_queue.empty() and i < sequence_length:
            max_prob = 0
            tokenized_sentence, prb = beam_queue.get()
            decoded_sentence = self.tokenizer_dst.convert_tokens_to_string(tokenized_sentence)
            tokenized_dst_sentence = self.tokenizer_dst(decoded_sentence, return_tensors='tf', add_special_tokens=False,
                                                        max_length=sequence_length, truncation=True,
                                                        padding='max_length').data['input_ids']
            predictions = self.transformer([tokenized_input_sentence, tokenized_dst_sentence])
            i = len(tokenized_sentence) - 1
            sampled_token_indexes = np.flip(np.argsort(predictions[0, i, :])[-k:])
            probabilities = [float(predictions[0, i, j]) for j in sampled_token_indexes]

            for samp_index, p in zip(sampled_token_indexes, probabilities):
                sampled_token = self.tokenizer_dst.ids_to_tokens[samp_index]
                tok_sent_with_new_samp = tokenized_sentence.copy()
                tok_sent_with_new_samp.append(sampled_token)
                next_sent = (tok_sent_with_new_samp, p * prb)
                if next_sent[1] > max_prob:
                    max_prob = next_sent[1]

                if sampled_token == "[SEP]":
                    translated.append(next_sent)
                    print([self.tokenizer_dst.convert_tokens_to_string(next_sent[0][1:-1]), next_sent[1]])

                elif next_sent[1] > max_prob / 1.5:
                    beam_queue.put(next_sent)

            if len(translated) > 2:
                break

        translated.sort(key=lambda x: x[1], reverse=True)
        return translated[0], self.tokenizer_dst.convert_tokens_to_string(translated[0][0][1:-1])
