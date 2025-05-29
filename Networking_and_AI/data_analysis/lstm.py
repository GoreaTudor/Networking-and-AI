import numpy as np
from keras.src import Input
from keras.src.layers import LSTM, Dense
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from data_analysis.data_loader import load_packets_supervised_data
from data_analysis.graphs import draw_confusion_matrix
from data_analysis.utils import create_sequences, SEQ_LENGTH, N_EPOCHS, LR


def run_lstm(df: DataFrame,
             seq_len: int = SEQ_LENGTH,
             n_epochs: int = N_EPOCHS,
             lr: float = LR):
    # filter and encode
    df = df[df['attack_type'] != 'unknown']

    le_protocol = LabelEncoder()
    df['protocol_encoded'] = le_protocol.fit_transform(df['protocol'])

    le_attack = LabelEncoder()
    df['attack_type_encoded'] = le_attack.fit_transform(df['attack_type'])

    # sort by time
    df = df.sort_values('time')

    # select features
    data_x = df[['size', 'protocol_encoded']].values
    data_y = df['attack_type_encoded'].values

    # create sequences
    seq_x, seq_y = create_sequences(data_x, data_y, seq_len)

    # one-hot encode labels
    num_classes = len(le_attack.classes_)
    seq_y_cat = to_categorical(seq_y, num_classes=num_classes)

    # train/test split
    split = int(0.8 * len(seq_x))
    train_x, test_x = seq_x[:split], seq_x[split:]
    train_y, test_y = seq_y_cat[:split], seq_y_cat[split:]
    test_y_raw = seq_y[split:]

    # build LSTM model
    model = Sequential()
    model.add(Input(shape=(seq_len, seq_x.shape[2])))
    model.add(LSTM(64))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=n_epochs, batch_size=32, validation_split=0.1)

    pred_y_prob = model.predict(test_x)
    pred_y = np.argmax(pred_y_prob, axis=1)

    labels_in_test = np.unique(test_y_raw)
    target_names_in_test = le_attack.inverse_transform(labels_in_test)

    print(classification_report(
        test_y_raw,
        pred_y,
        labels=labels_in_test,
        target_names=target_names_in_test
    ))

    draw_confusion_matrix(test_y_raw, pred_y, labels=labels_in_test)


if __name__ == '__main__':
    dataframe = load_packets_supervised_data()
    run_lstm(df=dataframe)
