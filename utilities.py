import tensorflow as tf
import numpy as np


def create_dataset_euparl(name: str, src: str = "en", dst: str = "it", size: float = 1) -> (list, list):
    with open(name+".{0}".format(src), encoding="UTF-8") as datafile:
        src_set = datafile.readlines()

    with open(name+".{0}".format(dst), encoding="UTF-8") as datafile:
        dst_set = datafile.readlines()

    if size != 1:
        datasets_to_shuffle = list((zip(src_set, dst_set)))
        np.random.shuffle(datasets_to_shuffle)
        src_set, dst_set = zip(*datasets_to_shuffle)
        src_set = list(src_set[:int(len(src_set) * size)])
        dst_set = list(dst_set[:int(len(dst_set) * size)])

    return src_set, dst_set


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


def merge_datasets(first_dataset, second_dataset) -> (list, list):
    first_src, first_dst = first_dataset
    second_src, second_dst = second_dataset
    src_set = first_src + second_src
    dst_set = first_dst + second_dst
    return src_set, dst_set


def set_max_tokens(dataset: list, language: str = "en", show_out: bool = False) -> int:
    len_sentences = [len(sentence.split()) for sentence in dataset]
    mean_len_sentences = np.mean(len_sentences)
    max_length = int(mean_len_sentences + 2 * np.std(len_sentences))
    if show_out:
        print("{0} dataset average sentence length: {1}".format(language, mean_len_sentences))
        print("{0} dataset max length allowed: {1}".format(language, max_length))

    return max_length


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
    return {"encoder_inputs": eng, "decoder_inputs": ita[:, :-1]}, ita[:, 1:]


def make_batches(dataset_src_dst: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
    dataset = dataset_src_dst.batch(batch_size)
    dataset = dataset.map(__format_dataset)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()
