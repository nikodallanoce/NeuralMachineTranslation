from utilities import *
# from keras.preprocessing.text import Tokenizer
# from tokenizer import *
# import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from decoder import *

if __name__ == '__main__':
    # dataset_en = create_patterns("dataset/europarl-v7.it-en.en")
    # dataset_it = create_patterns("dataset/europarl-v7.it-en.it")

    en_set, it_set = create_dataset_anki("dataset/ita.txt")
    # tokenizer_en = TokenizerNMT(en_set)
    # tokenizer_it = TokenizerNMT(it_set, None, False)
    # print("Successfully loaded datasets")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model: TFBertModel = TFBertModel.from_pretrained("bert-base-uncased")
    en_set = list(en_set)
    tokens = tokenizer("good job guys, well done", add_special_tokens=True, truncation=True, padding="max_length",
                       return_attention_mask=True, return_token_type_ids=True, return_tensors="tf", max_length=20)
    out = model.call(**tokens)
    out_hidden = tf.convert_to_tensor(out[0].numpy()[:, -1, :])
    att_mask = tokens.data["attention_mask"]
    tokenizer_it = TokenizerNMT(it_set, None, False)
    decoder = DecoderRNN("lstm", v_size=len(tokenizer_it.word_index), freeze_parameters=False)
    tokens_it = tokenizer_it.tokens
    out_decoder = decoder.call(tf.convert_to_tensor(tokens_it[0, 0].reshape(1, 1)),
                               out[0], att_mask, [out_hidden, out_hidden])
    translation_token = int(tf.argmax(out_decoder[0], axis=1).numpy())
    dst_token_translated = tokenizer_it.index_word[translation_token]
    print()
