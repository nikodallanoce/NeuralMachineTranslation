from transformers import BertTokenizer
import logging
from encoder import *
from decoder import *
from translator import Translator
from utilities import *


def create_model(layers_size: int, num_layers: int, dense_size: int, num_heads: int, max_length: int, encoder=None) -> tf.keras.Model:
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="encoder_inputs")
    if encoder is not None:
        outputs = encoder(encoder_inputs)
        encoder_outputs = outputs.last_hidden_state
        layers_size = encoder_outputs.shape[-1]  # the size of the encoder and decoder layers must be the same
    else:
        encoder_outputs = EncoderTransformer(num_layers, layers_size, dense_size, num_heads, max_length, v_size_en)(encoder_inputs)

    # Decoder
    decoder_inputs = tf.keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    encoded_seq_inputs = tf.keras.Input(shape=(None, layers_size), name="decoder_state_inputs")
    decoder_outputs = DecoderTransformer(num_layers, layers_size, dense_size, num_heads, max_length, v_size_it)(decoder_inputs, encoded_seq_inputs)
    decoder_outputs = layers.Dense(v_size_it, activation="softmax")(decoder_outputs)
    decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs, name="decoder_transformer")

    # Final model
    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    transformer = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")
    return transformer


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

    # Build the model
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    transformer = create_model(512, 6, 2048, 8, 30)
    transformer.summary()
    transformer.compile(opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    translator = Translator(tokenizer_en, tokenizer_it, 30, transformer)
    print()
