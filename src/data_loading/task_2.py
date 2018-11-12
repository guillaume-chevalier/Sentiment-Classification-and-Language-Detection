import sklearn.utils

from src.utils import identity


def load_train(corpus_train, sentence_tokenizer=identity):
    labels_filenames, labels_paths = list(zip(*sorted([(r.split("/")[-1], r) for r in corpus_train])))

    data = []
    labels = []
    for (path, label) in zip(labels_paths, labels_filenames):
        label = labels_filenames.index(label)

        loaded_data_as_a_string = _load_file(path)
        sentences = sentence_tokenizer(loaded_data_as_a_string)
        for sent in sentences:
            data.append(sent)
            labels.append(label)

    data, labels = sklearn.utils.shuffle(data, labels, random_state=42)

    labels_readable = [l.split("-")[0] for l in labels_filenames]
    return data, labels, labels_readable


def load_test(corpus_test, labels_readable):
    def index_of_label(filename, labels_suffix):
        for (i, suffix) in enumerate(labels_suffix):
            if suffix in filename:
                return i
        raise ValueError("Label not found. See data: " + filename + ", " + str(labels_suffix))

    labels_suffix = ["-" + l[:2] + ".txt" for l in labels_readable]
    if "-po.txt" in labels_suffix:
        # set po to pt instead...
        labels_suffix[labels_suffix.index("-po.txt")] = "-pt.txt"
    # e.g.: 'english-training.txt' ==> '-en.txt'
    # e.g.: 'portugese-training.txt' ==> '-pt.txt'

    data = []
    labels = []
    for file in corpus_test:
        loaded_data = _load_file(file).split("\n")
        label_index = index_of_label(file, labels_suffix)
        for document in loaded_data:
            data.append(document)
            labels.append(label_index)

    return data, labels


def _load_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as fml:
        crushed_content = fml.read().strip().replace("\n", " ").replace("\r", " ").replace("  ", " ")
    return crushed_content
