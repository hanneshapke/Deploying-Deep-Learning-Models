import numpy as np

from constants import MAX_TOKENS, CHARS


def clean_data_onehot(sentence, max_tokens=MAX_TOKENS, sup_chars=CHARS):
    sentence = str(sentence).lower()
    x = np.zeros((max_tokens, len(sup_chars)), dtype=np.float32)
    for i, c in enumerate(sentence):
        if i >= max_tokens:
            break
        try:
            x[i][sup_chars.index(c)] = 1
        except ValueError:
            pass
    return x


def clean_data_encoded(sentence, max_tokens=MAX_TOKENS, sup_chars=CHARS):
    sentence = str(sentence).lower()
    x = np.zeros((max_tokens, ), dtype=np.float32)
    for i, c in enumerate(sentence):
        if i >= max_tokens:
            break
        try:
            x[i] = sup_chars.index(c)
        except ValueError:
            pass
    return x


def y_onehot(y_value):
    y = np.zeros((5), dtype=np.int8)
    y[y_value - 1] = 1
    return y
