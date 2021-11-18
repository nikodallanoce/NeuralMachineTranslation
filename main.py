from transformer import *
# from tqdm import tqdm
import logging
from translator import Translator
from utilities import *


if __name__ == '__main__':
    # dataset_en = create_patterns("dataset/europarl-v7.it-en.en")
    # dataset_it = create_patterns("dataset/europarl-v7.it-en.it")

    # en_set, it_set = create_dataset_anki("dataset/ita.txt")
    en_set, it_set = create_dataset_euparl("dataset/europarl-v7.it-en")

    # Create the tokenizers and get the number of tokens
    logging.getLogger("tensorflow").setLevel(logging.ERROR)  # suppress warnings for tensorflow
    logging.getLogger("transformers").setLevel(logging.ERROR)  # suppress warnings for transformers

    encoder_model = encoder_models["bert"]
    tokenizer_en = encoder_model["tokenizer"]
    tokenizer_it = BertTokenizerFast.from_pretrained("dbmdz/bert-base-italian-cased")
    v_size_en = tokenizer_en.vocab_size
    v_size_it = tokenizer_it.vocab_size

    # Choose the running strategy
    strategy = choose_strategy()

    # Tokenize the dataset
    max_length = np.max([set_max_tokens(en_set, "en"), set_max_tokens(it_set, "it")])
    with strategy.scope():
        tokens_en = tokenizer_en(en_set, add_special_tokens=True,
                                 truncation=True, padding="max_length", return_attention_mask=True,
                                 return_tensors="tf", max_length=max_length).data["input_ids"]
        tokens_it = tokenizer_it(it_set, add_special_tokens=True,
                                 truncation=True, padding="max_length", return_attention_mask=True,
                                 return_tensors="tf", max_length=max_length+1).data["input_ids"]

    # Build the dataset and split it in train, validation and test
    dataset = tf.data.Dataset.from_tensor_slices((tokens_en, tokens_it))  # build the tf dataset
    tr_set, val_set, ts_set = split_set(dataset, 0.8, 0.1, 0.1)  # split the tf dataset

    # Create the batches
    batch_size = 16 * strategy.num_replicas_in_sync
    tr_batches = make_batches(tr_set, batch_size)  # create the train batches
    val_batches = make_batches(val_set, batch_size)  # create the validation batches

    # Build the model
    with strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        transformer = create_model(512, 6, 2048, 8, 30, v_size_en, v_size_it)
        transformer.summary()
        transformer.compile(opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    with strategy.scope():
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./translator_bert.h5',
                                                                       save_weights_only=True, monitor='val_loss',
                                                                       mode='auto', save_best_only=True)

    # Train the model
    transformer.fit(tr_batches, epochs=10, validation_data=val_batches, callbacks=[model_checkpoint_callback])

    # Evaluate the model on the test set
    ts_loss, ts_accuracy = transformer.evaluate(make_batches(ts_set, batch_size))
    print("Test loss: {0}\nTest accuracy: {1}".format(ts_loss, ts_accuracy))

    # Build the translator
    with strategy.scope():
        translator = Translator(tokenizer_en, tokenizer_it, max_length, transformer)

    # Work with the translator
    en = "My house has several bedrooms."
    tokens, translated = translator(en)
    print([en, translated])
    print()
