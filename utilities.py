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


def make_batches(dataset_src_dst: tf.data.Dataset, batch_size: int):
    return dataset_src_dst.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def __get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = __get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
