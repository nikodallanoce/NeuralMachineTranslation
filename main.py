from transformers import TFBertModel, BertTokenizer, logging
from encoder import *
from decoder import *
from transformer import TransformerNMT


if __name__ == '__main__':
    # dataset_en = create_patterns("dataset/europarl-v7.it-en.en")
    # dataset_it = create_patterns("dataset/europarl-v7.it-en.it")

    en_set, it_set = create_dataset_anki("dataset/ita.txt")

    # Create the tokenizers and get the number of tokens
    logging.set_verbosity_error()  # suppress warnings for transformers
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

    tr_batches = make_batches(tr_set, 128)
    val_batches = make_batches(val_set, 128)

    # Build encoder and decoder
    encoderBert: TFBertModel = TFBertModel.from_pretrained("bert-base-uncased")
    encoder = EncoderTransformer(8, 512, 8, 2048, v_size_en, 10000)
    decoder = DecoderTransformer(8, 512, 8, 2048, v_size_it, 10000)

    model = TransformerNMT(encoder, decoder, v_size_it)
    # out = model.train(en_set[:50], it_set[:50], 10)
    # decoder.save("decoder_NMT.h5")
    # print(out)
    print()