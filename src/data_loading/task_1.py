import sklearn.utils
from sklearn.model_selection import train_test_split


def load_all_data_task_1(pos_files_paths, neg_files_paths):
    pos = load_all_files(pos_files_paths)
    neg = load_all_files(neg_files_paths)

    X = pos + neg
    y = [1] * len(pos) + [0] * len(neg)

    # Have non-concentrated classes, but with seed for reproducibility:
    X, y = sklearn.utils.shuffle(X, y, random_state=42)

    # 20% for test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, y_train, X_test, y_test


def load_all_files(file_paths):
    return [_load_file(path) for path in file_paths]


def _load_file(file_path):
    with open(file_path, 'r', encoding="ISO-8859-1") as fml:
        crushed_content = fml.read().rstrip("\n").strip()
    return crushed_content
