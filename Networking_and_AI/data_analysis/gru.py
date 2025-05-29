import numpy as np
from keras.src.layers import GRU, Dense, Dropout, Input
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from data_analysis.data_loader import load_packets_supervised_data
from data_analysis.graphs import draw_confusion_matrix
from data_analysis.utils import create_sequences, SEQ_LENGTH, N_EPOCHS, LR


def run_gru(df: DataFrame,
            seq_len: int = SEQ_LENGTH,
            n_epochs: int = N_EPOCHS,
            lr: float = LR):
    # encode categorical columns
    df = df[df['attack_type'] != 'unknown']

    le_protocol = LabelEncoder()
    df['protocol_encoded'] = le_protocol.fit_transform(df['protocol'])

    le_attack = LabelEncoder()
    df['attack_type_encoded'] = le_attack.fit_transform(df['attack_type'])

    # sort by time
    df = df.sort_values('time')

    # extract features and labels
    features = ['size', 'protocol_encoded']
    data_x = df[features].values
    data_y = df['attack_type_encoded'].values

    # create time series sequences
    seq_x, seq_y = create_sequences(data_x, data_y, seq_len)

    # one-hot encode labels
    num_classes = len(le_attack.classes_)
    y_seq_cat = to_categorical(seq_y, num_classes=num_classes)

    # train/test split
    split = int(0.8 * len(seq_x))
    train_x, test_x = seq_x[:split], seq_x[split:]
    train_y, test_y = y_seq_cat[:split], y_seq_cat[split:]
    test_y_raw = seq_y[split:]

    # GRU model
    model = Sequential([
        Input(shape=(seq_len, seq_x.shape[2])),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    # train
    model.fit(train_x, train_y, epochs=n_epochs, batch_size=32, validation_split=0.1)

    # evaluate
    y_pred_probs = model.predict(test_x)
    y_pred = y_pred_probs.argmax(axis=1)

    # report
    from sklearn.metrics import classification_report

    labels_in_test = np.unique(test_y_raw)
    target_names_in_test = le_attack.inverse_transform(labels_in_test)

    print(classification_report(
        test_y_raw,
        y_pred,
        labels=labels_in_test,
        target_names=target_names_in_test
    ))

    draw_confusion_matrix(test_y_raw, y_pred, labels=labels_in_test)


if __name__ == '__main__':
    dataframe = load_packets_supervised_data()
    run_gru(df=dataframe)
