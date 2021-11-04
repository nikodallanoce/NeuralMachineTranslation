import tensorflow as tf
import numpy as np


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


def create_dataset_anki(name: str, preprocessed: bool = False) -> (list, list):
    with open(name, encoding="UTF-8") as datafile:
        src_set = list()
        dst_set = list()
        for sentence in datafile:
            sentence = sentence.split("\t")
            src_set.append(sentence[0])
            if preprocessed:
                dst_set.append(sentence[1].split("\n")[0])
            else:
                dst_set.append(sentence[1])

    return src_set, dst_set


def split_set(dataset: tf.data.Dataset,
              tr: float = 0.8,
              val: float = 0.1,
              ts: float = 0.1,
              shuffle: bool = True) -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    if tr+val+ts != 1:
        raise ValueError("Train, validation and test partition not allowed with such splits")

    dataset_size = dataset.cardinality().numpy()
    if shuffle:
        dataset = dataset.shuffle(dataset_size)

    tr_size = int(tr * dataset_size)
    val_size = int(val * dataset_size)

    tr_set = dataset.take(tr_size)
    val_set = dataset.skip(tr_size).take(val_size)
    ts_set = dataset.skip(tr_size).skip(val_size)
    return tr_set, val_set, ts_set


def __format_dataset(eng, ita):
    return {"encoder_inputs": eng, "decoder_inputs": ita[:, :-1],}, ita[:, 1:]


def make_batches(dataset_src_dst: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    dataset = dataset_src_dst.batch(batch_size)
    dataset = dataset.map(__format_dataset)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()
