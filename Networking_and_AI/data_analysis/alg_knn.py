from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from data_analysis.data_loader import load_training_supervised_data, \
    load_testing_supervised_data
from data_analysis.graphs import draw_confusion_matrix
from data_analysis.utils import CLASSIFICATION_INPUT_FEATURES


def run_knn(train_df: DataFrame,
            test_df: DataFrame,
            input_features: list[str] = CLASSIFICATION_INPUT_FEATURES):
    # filter unknowns
    train_df = train_df[train_df["attack_type"] != "unknown"]
    test_df = test_df[test_df["attack_type"] != "unknown"]

    # replace nan ports & flags with -1
    train_df['src_port'] = train_df['src_port'].fillna(-1)
    test_df['src_port'] = test_df['src_port'].fillna(-1)
    train_df['dst_port'] = train_df['dst_port'].fillna(-1)
    test_df['dst_port'] = test_df['dst_port'].fillna(-1)
    train_df['flags'] = train_df['flags'].fillna("-")
    test_df['flags'] = test_df['flags'].fillna("-")

    # encode labels
    le_protocol = LabelEncoder()
    train_df["protocol_encoded"] = le_protocol.fit_transform(train_df["protocol"])
    test_df["protocol_encoded"] = le_protocol.transform(test_df["protocol"])

    le_flags = LabelEncoder()
    train_df["flags_encoded"] = le_flags.fit_transform(train_df["flags"])
    test_df["flags_encoded"] = le_flags.transform(test_df["flags"])

    le_attack = LabelEncoder()
    train_df["attack_type_encoded"] = le_attack.fit_transform(train_df["attack_type"])
    test_df["attack_type_encoded"] = le_attack.transform(test_df["attack_type"])

    if not set(test_df["attack_type"]).issubset(set(le_attack.classes_)):
        print("Test set contains unseen attack types.")
        return

    # input and output features
    train_x = train_df[input_features]
    test_x = test_df[input_features]
    train_y = train_df["attack_type_encoded"]
    test_y = test_df["attack_type_encoded"]

    # dynamic neighbor count based on number of classes
    n_neighbors = len(test_y.unique())
    print(f"Dynamic n_neighbors set to: {n_neighbors}")

    # train
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(train_x, train_y)

    # predict
    pred_y = model.predict(test_x)

    # decode
    test_labels_str = le_attack.inverse_transform(test_y)
    pred_labels_str = le_attack.inverse_transform(pred_y)
    labels_in_test = le_attack.inverse_transform(sorted(test_y.unique()))

    # report
    print(classification_report(
        test_labels_str,
        pred_labels_str,
        labels=labels_in_test,
        target_names=labels_in_test
    ))

    draw_confusion_matrix(test_labels_str, pred_labels_str, label_names=labels_in_test)


if __name__ == '__main__':
    training_df = load_training_supervised_data()
    testing_df = load_testing_supervised_data()
    run_knn(training_df, testing_df)
