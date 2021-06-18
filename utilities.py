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
