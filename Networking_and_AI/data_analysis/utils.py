import numpy as np

RANDOM_STATE = 42


def create_sequences(x, y, seq_length):
    xs, ys = [], []
    for i in range(len(x) - seq_length):
        xs.append(x[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)
