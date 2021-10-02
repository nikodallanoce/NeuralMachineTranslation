from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class TokenizerNMT(Tokenizer):

    def __init__(self, dataset, num_words=None, reverse=True):
        super(TokenizerNMT, self).__init__(num_words=num_words)

        # the dataset passed to the tokenizer should be the development and test set together
        preprocessed_dataset = ["[Start] " + pattern + " [End]" for pattern in dataset]
        self.fit_on_texts(preprocessed_dataset)  # builds the list of words from the dataset
        self.dataset_tokens = self.texts_to_sequences(preprocessed_dataset)  # tokenizes the dataset

        pad = "post"
        trun = "post"
        if reverse:
            self.dataset_tokens = [list(reversed(pattern)) for pattern in self.dataset_tokens]
            pad = "pre"
            trun = "pre"

        len_tokens = [len(pattern) for pattern in self.dataset_tokens]
        mean_tokens = np.mean(len_tokens)  # mean number of tokens for each tokens
        max_tokens = int(mean_tokens + 2 * np.std(len_tokens))  # max number of tokens allowed
        self.tokens = pad_sequences(self.dataset_tokens, max_tokens, padding=pad, truncating=trun)  # matrix of tokens
        self.tokens = pad_sequences(self.dataset_tokens, None, padding=pad, truncating=trun)
