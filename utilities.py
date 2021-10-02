from tokenizer import *
# import tensorflow as tf


def create_patterns(name: str):
    with open(name, encoding='UTF-8') as datafile:
        sentences = datafile.readlines()

    return sentences


def create_dataset_europarl(eng_ita=True):
    src, dst = 'en', 'it'
    if not eng_ita:
        src, dst = 'it', 'en'

    x = create_patterns('dataset/europarl-v7.it-en.{0}'.format(src))
    y = create_patterns('dataset/europarl-v7.it-en.{0}'.format(dst))
    ds = []
    for (xi, yi) in zip(x, y):
        ds.append((xi, yi))

    return ds


def split_set(tokenizer_src: TokenizerNMT, tokenizer_dst: TokenizerNMT, test: float):
    n = int(len(tokenizer_src.tokens) * (1-test))
    dev_set = list(zip(tokenizer_src.tokens[:n, :], tokenizer_dst.tokens[:n, :]))
    test_set = list(zip(tokenizer_src.tokens[n:, :], tokenizer_dst.tokens[n:, :]))
    return dev_set, test_set


def create_dataset_anki(name: str):
    with open(name, encoding="UTF-8") as datafile:
        src_set = list()
        dst_set = list()
        for sentence in datafile:
            src_set.append(sentence.split("\t")[0])
            dst_set.append(sentence.split("\t")[1])

    return np.array(src_set), np.array(dst_set)
