import numpy as np
from keras.src import Input
from keras.src.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from data_analysis.data_loader import load_training_supervised_data, \
    load_testing_supervised_data
from data_analysis.graphs import draw_confusion_matrix
from data_analysis.utils import create_sequences, SEQ_LENGTH, N_EPOCHS, LR


def run_cnn(train_df: DataFrame,
            test_df: DataFrame,
            seq_len: int = SEQ_LENGTH,
            n_epochs: int = N_EPOCHS,
            lr: float = LR):
    # remove unknown labels
    train_df = train_df[train_df['attack_type'] != 'unknown']
    test_df = test_df[test_df['attack_type'] != 'unknown']

    # label encoding
    le_protocol = LabelEncoder()
    train_df['protocol_encoded'] = le_protocol.fit_transform(train_df['protocol'])
    test_df['protocol_encoded'] = le_protocol.transform(test_df['protocol'])

    le_attack = LabelEncoder()
    train_df['attack_type_encoded'] = le_attack.fit_transform(train_df['attack_type'])

    # check all test labels exist in train labels
    if not set(test_df['attack_type']).issubset(set(le_attack.classes_)):
        print("Test set contains labels not present in training set. Exiting.")
        return

    test_df['attack_type_encoded'] = le_attack.transform(test_df['attack_type'])

    # sort by time
    train_df = train_df.sort_values('time')
    test_df = test_df.sort_values('time')

    # feature selection
    features = ['size', 'protocol_encoded']
    train_x = train_df[features].values
    train_y = train_df['attack_type_encoded'].values

    test_x = test_df[features].values
    test_y = test_df['attack_type_encoded'].values

    # create sequences
    train_seq_x, train_seq_y = create_sequences(train_x, train_y, seq_len)
    test_seq_x, test_seq_y = create_sequences(test_x, test_y, seq_len)

    # one-hot encode labels
    num_classes = len(le_attack.classes_)
    train_seq_cat_y = to_categorical(train_seq_y, num_classes=num_classes)
    test_seq_cat_y = to_categorical(test_seq_y, num_classes=num_classes)

    # build model
    model = Sequential([
        Input(shape=(seq_len, train_seq_x.shape[2])),
        Conv1D(64, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # train
    model.fit(train_seq_x, train_seq_cat_y, epochs=n_epochs, batch_size=32, validation_split=0.1)

    # predict
    pred_y_prob = model.predict(test_seq_x)
    pred_y = np.argmax(pred_y_prob, axis=1)

    # convert numeric predictions and ground truth back to string labels
    test_labels_str = le_attack.inverse_transform(test_seq_y)
    pred_labels_str = le_attack.inverse_transform(pred_y)

    # get only label names in the test set
    labels_in_test = np.unique(test_seq_y)
    target_names_in_test = le_attack.inverse_transform(labels_in_test)

    # classification report
    print(classification_report(
        test_labels_str,
        pred_labels_str,
        labels=target_names_in_test,
        target_names=target_names_in_test
    ))

    # confusion matrix
    draw_confusion_matrix(test_labels_str, pred_labels_str, label_names=target_names_in_test)


if __name__ == '__main__':
    training_df = load_training_supervised_data()
    testing_df = load_testing_supervised_data()
    run_cnn(training_df, testing_df)
