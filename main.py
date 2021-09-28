import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, BartTokenizer, TFBartModel
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import plot_model
import tensorflow_addons as tfa
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model: TFBertModel = TFBertModel.from_pretrained("bert-base-uncased", trainable=True)
    tokens = tokenizer("good job guys", add_special_tokens=True, truncation=True, padding="max_length",
                        return_attention_mask=True, return_token_type_ids=True, return_tensors="tf", max_length=25)
    ret = model.call(**tokens, output_attentions=True, output_hidden_states=True)
    print()
