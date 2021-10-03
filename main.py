from utilities import *
# import tensorflow as tf
from transformers import TFBertModel, BertTokenizer, AutoTokenizer
from decoder import *
from transformer import TransformerNMT

if __name__ == '__main__':
    # dataset_en = create_patterns("dataset/europarl-v7.it-en.en")
    # dataset_it = create_patterns("dataset/europarl-v7.it-en.it")

    en_set, it_set = create_dataset_anki("dataset/ita.txt")

    # Create the tokenizers and get the number of tokens
    tokenizer_en = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_it = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-uncased")
    v_size_en = tokenizer_en.vocab_size
    v_size_it = tokenizer_it.vocab_size

    # Tokenize the dataset
    """dataset_tokens_en = tokenizer_en(list(en_set[:50]), add_special_tokens=True,
                                     truncation=True, padding="max_length", return_attention_mask=True,
                                     return_tensors="tf", max_length=30)
    dataset_tokens_it = tokenizer_it(list(it_set[:50]), add_special_tokens=True,
                                     truncation=True, padding="max_length", return_attention_mask=True,
                                     return_tensors="tf", max_length=30)

    # Get the tokens and attention masks
    tokens_en: tf.Tensor = dataset_tokens_en.data["input_ids"]
    att_mask_en: tf.Tensor = dataset_tokens_en.data["attention_mask"]
    tokens_it: tf.Tensor = dataset_tokens_it.data["input_ids"]
    att_mask_it: tf.Tensor = dataset_tokens_it.data["attention_mask"]"""

    # Build encoder and decoder
    encoder: TFBertModel = TFBertModel.from_pretrained("bert-base-uncased")
    decoder = DecoderRNN("lstm", v_size=v_size_it, emb_dropout=0, layers_dropout=0, att_dropout=0)

    # Encode the source sentences
    # enc_out = encoder(**dataset_tokens_en)[0]  # only takes last_hidden_state

    model = TransformerNMT(encoder, decoder, tokenizer_en, tokenizer_it)
    out = model.train(en_set[:50], it_set[:50], 10)
    print(out)
    """for i in range(len(enc_out)):
        att_mask = tf.expand_dims(att_mask_en[i], 0)
        h_state = tf.expand_dims(enc_out[i, -1], 0)
        c_state = tf.expand_dims(enc_out[i, -1], 0)
        translation = list()
        for token in tokens_it[i]:
            token = tf.reshape(token, (1, 1))
            dec_out, h_state, c_state = decoder(token, enc_out[i], att_mask, [h_state, c_state])
            translation.append(tokenizer_it.decode(tf.argmax(dec_out, axis=1)))
        break"""
    print()
