import numpy as np
from keras.src import Input
from keras.src.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.src.models import Sequential
from keras.src.models.cloning import clone_model
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from data_analysis.data_loader import load_testing_supervised_data, load_training_supervised_data
from data_analysis.graphs import draw_confusion_matrix, draw_weights_graph
from data_analysis.utils import SEQ_LENGTH, N_EPOCHS, LR, create_sequences, TIME_SERIES_INPUT_FEATURES


### Uncomment next 2 lines if multiple models can't be trained at once:
# import tensorflow as tf
# tf.config.run_functions_eagerly(True)


def __encode_protocols_globally(train_dfs):
    le_protocol = LabelEncoder()
    protocols = np.concatenate([df['protocol'].values for df in train_dfs])
    le_protocol.fit(protocols)

    return protocols, le_protocol


def __encode_flags_globally(train_dfs):
    le_flags = LabelEncoder()
    flags = np.concatenate([df['flags'].fillna('-').values for df in train_dfs])
    print(f"flags {flags}")
    le_flags.fit(flags)

    return flags, le_flags


def __encode_attacks_globally(train_dfs):
    le_attack = LabelEncoder()
    attacks = np.concatenate([df[df['attack_type'] != 'unknown']['attack_type'].values for df in train_dfs])
    le_attack.fit(attacks)

    return attacks, le_attack


def __average_weights(weight_list):
    avg_weights = []
    for weights in zip(*weight_list):
        avg_weights.append(np.mean(weights, axis=0))
    return avg_weights


def run_federated(train_dfs: list[DataFrame],
                  test_df: DataFrame,
                  input_features: list[str] = TIME_SERIES_INPUT_FEATURES,
                  seq_len: int = SEQ_LENGTH,
                  n_epochs: int = N_EPOCHS,
                  lr: float = LR):
    # how many models?
    nr_clients = len(train_dfs)
    print(f"federated training using {nr_clients} clients")

    # filter unknowns
    test_df = test_df[test_df['attack_type'] != 'unknown']

    # encode labels globally
    protocols, le_protocol = __encode_protocols_globally(train_dfs)
    flags, le_flags = __encode_flags_globally(train_dfs)
    attacks, le_attack = __encode_attacks_globally(train_dfs)

    # encode test labels
    test_df['protocol_encoded'] = le_protocol.transform(test_df['protocol'])
    test_df['flags_encoded'] = le_flags.transform(test_df['flags'])
    test_df['attack_type_encoded'] = le_attack.transform(test_df['attack_type'])

    if not set(test_df['attack_type']).issubset(set(le_attack.classes_)):
        print("Test set contains labels not present in training set. Exiting.")
        return

    # sort test by time
    test_df = test_df.sort_values('time')

    # input and output features
    test_x = test_df[input_features].values
    test_y = test_df['attack_type_encoded'].values

    # create sequences & one-hot encode labels
    num_classes = len(le_attack.classes_)
    test_seq_x, test_seq_y = create_sequences(test_x, test_y, seq_len)
    test_seq_cat_y = to_categorical(test_seq_y, num_classes=len(le_attack.classes_))

    # create initial model
    initial_model = Sequential([
        Input(shape=(seq_len, test_seq_x.shape[2])),
        Conv1D(64, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    initial_model.compile(optimizer=Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    initial_weights = initial_model.get_weights()

    all_weights = []

    # train local clients
    for client_idx, client_df in enumerate(train_dfs):
        print(f"\n\nClient {client_idx + 1}:")

        # filter unknowns
        client_df = client_df[client_df['attack_type'] != 'unknown']

        # encode labels
        client_df['protocol_encoded'] = le_protocol.transform(client_df['protocol'])
        client_df['flags_encoded'] = le_flags.transform(client_df['flags'])
        client_df['attack_type_encoded'] = le_attack.transform(client_df['attack_type'])

        # sort by time
        client_df = client_df.sort_values('time')

        # input and output features
        client_x = client_df[input_features].values
        client_y = client_df['attack_type_encoded'].values

        # create sequences & one-hot encode labels
        client_seq_x, client_seq_y = create_sequences(client_x, client_y, seq_len)
        client_seq_cat_y = to_categorical(client_seq_y, num_classes=num_classes)

        # clone model
        local_model = clone_model(initial_model)
        local_model.set_weights(initial_weights)  # made sure I clone the weights
        local_model.compile(optimizer=Adam(learning_rate=lr),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        # train
        print(">> Training...")
        local_model.fit(client_seq_x, client_seq_cat_y, epochs=n_epochs, batch_size=32, verbose=0)
        all_weights.append(local_model.get_weights())

        # predict
        pred_y_prob = local_model.predict(test_seq_x)
        pred_y = np.argmax(pred_y_prob, axis=1)

        # decode
        test_labels_str = le_attack.inverse_transform(test_seq_y)
        pred_labels_str = le_attack.inverse_transform(pred_y)
        labels_in_test = np.unique(test_seq_y)
        target_names_in_test = le_attack.inverse_transform(labels_in_test)

        # report client
        print(">> Evaluation:")
        print(classification_report(
            test_labels_str,
            pred_labels_str,
            labels=target_names_in_test,
            target_names=target_names_in_test
        ))

        draw_confusion_matrix(test_labels_str, pred_labels_str, label_names=target_names_in_test)

    # aggregate weights -- by averaging
    print("\n\n >>> Global Model <<<")
    global_model = clone_model(initial_model)
    averaged_weights = __average_weights(all_weights)
    global_model.set_weights(averaged_weights)

    # predict
    pred_y_prob = global_model.predict(test_seq_x)
    pred_y = np.argmax(pred_y_prob, axis=1)

    # decode
    test_labels_str = le_attack.inverse_transform(test_seq_y)
    pred_labels_str = le_attack.inverse_transform(pred_y)
    labels_in_test = np.unique(test_seq_y)
    target_names_in_test = le_attack.inverse_transform(labels_in_test)

    # report
    print(">> Final Results:")
    print(classification_report(
        test_labels_str,
        pred_labels_str,
        labels=target_names_in_test,
        target_names=target_names_in_test
    ))

    draw_confusion_matrix(test_y=test_labels_str, pred_y=pred_labels_str, label_names=target_names_in_test)
    draw_weights_graph(all_weights=all_weights, averaged_weights=averaged_weights)


if __name__ == '__main__':
    testing_df = load_testing_supervised_data()

    run_federated(
        train_dfs=[
            load_training_supervised_data(file_name="all_floods_n"),
            load_training_supervised_data(file_name="all_ddos_n"),
            load_training_supervised_data(file_name="all_other_n"),
        ],
        test_df=testing_df
    )
