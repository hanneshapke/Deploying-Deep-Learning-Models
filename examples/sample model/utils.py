import numpy as np

from constants import MAX_TOKENS, CHARS


def clean_data(sentence, max_tokens=MAX_TOKENS, sup_chars=CHARS):
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


def onehot_y(y_value):
    y = np.zeros((5), dtype=np.int8)
    y[y_value - 1] = 1
    return y
