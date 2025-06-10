import numpy as np

RANDOM_STATE = 42
N_ESTIMATORS = 100
SEQ_LENGTH = 10
N_EPOCHS = 10
LR = 0.001
CLASSIFICATION_INPUT_FEATURES = ['size', 'protocol_encoded', 'src_port', 'dst_port', 'flags_encoded']
TIME_SERIES_INPUT_FEATURES = ['size', 'protocol_encoded', 'flags_encoded']

def create_sequences(x, y, seq_length):
    xs, ys = [], []
    for i in range(len(x) - seq_length):
        xs.append(x[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)
