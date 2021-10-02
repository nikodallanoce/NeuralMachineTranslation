from transformers import TFBertModel, BertTokenizer
from abc import ABC, abstractmethod
import keras


class Encoder(keras.models.Model):

    def __init__(self, model, tokenizer: BertTokenizer):
        super(Encoder, self).__init__()
        self.model = model
        self.tokenizer = tokenizer  # il tokenizer possiamo metterlo direttamente nel transformer

    @abstractmethod
    def encode(self, src: str):
        pass

    def call(self):
        return None


class BertEncoder(Encoder):

    def __init__(self, bert_model: TFBertModel, tokenizer: BertTokenizer):
        super(BertEncoder, self).__init__(bert_model, tokenizer)

    def encode(self, tokens):
        out = self.model.call(**tokens, output_hidden_states=True)
        return out
