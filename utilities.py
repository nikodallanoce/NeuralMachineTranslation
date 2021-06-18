from tokenizer import *


def create_patterns(name: str):
    with open(name, encoding='UTF-8') as datafile:
        sentences = datafile.readlines()

    return sentences


def create_dataset(eng_ita=True):
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
    dataset = zip(tokenizer_src.tokens, tokenizer_dst.tokens)
    # np.random.shuffle(dataset)
    dev_set = dataset[:n]
    test_set = dataset[n:]
    return dev_set, test_set
