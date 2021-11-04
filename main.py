from transformers import BertTokenizer
import logging
from encoder import *
from decoder import *
# from translator import Translator
from utilities import *


if __name__ == '__main__':
    # dataset_en = create_patterns("dataset/europarl-v7.it-en.en")
    # dataset_it = create_patterns("dataset/europarl-v7.it-en.it")

    en_set, it_set = create_dataset_anki("dataset/ita.txt")

    # Create the tokenizers and get the number of tokens
    logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings for tensorflow
    logging.getLogger("transformers").setLevel(logging.ERROR)  # suppress warnings for transformers
    tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_it = BertTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
    v_size_en = tokenizer_en.vocab_size
    v_size_it = tokenizer_it.vocab_size

    # Tokenize the dataset
    tokens_en = tokenizer_en(en_set[:50000], add_special_tokens=True,
                             truncation=True, padding="max_length", return_attention_mask=True,
                             return_tensors="tf", max_length=30).data["input_ids"]
    tokens_it = tokenizer_it(it_set[:50000], add_special_tokens=True,
                             truncation=True, padding="max_length", return_attention_mask=True,
                             return_tensors="tf", max_length=30).data["input_ids"]

    # Build the dataset and split it in train, validation and test
    dataset = tf.data.Dataset.from_tensor_slices((tokens_en, tokens_it))  # build the tf dataset
    tr_set, val_set, ts_set = split_set(dataset, 0.8, 0.1, 0.1)  # split the tf dataset

    tr_batches = make_batches(tr_set, 128)  # create the train batches
    val_batches = make_batches(val_set, 128)  # create the validation batches

    # Build encoder and decoder
    # encoderBert: TFBertModel = TFBertModel.from_pretrained("bert-base-uncased", trainable=False)
    encoder_bert = EncoderBERT(TFBertModel.from_pretrained("bert-base-uncased"))
    encoder = EncoderTransformer(8, 512, 8, 2048, v_size_en, 10000)
    decoder = DecoderTransformer(8, 512, 8, 2048, v_size_it, 10000)
    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
    print()
