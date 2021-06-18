from utilities import *
from keras.preprocessing.text import Tokenizer


if __name__ == '__main__':
    dataset_en = create_patterns("dataset/europarl-v7.it-en.en")
    dataset_it = create_patterns("dataset/europarl-v7.it-en.it")
    dataset_en = [sentence.replace("\n", "") for sentence in dataset_en]
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(dataset_en)
    tokens = tokenizer.texts_to_sequences(dataset_en)
    print("Successfully loaded datasets")
