from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from data_analysis.data_loader import load_training_supervised_data, \
    load_testing_supervised_data
from data_analysis.graphs import draw_confusion_matrix


def run_knn(train_df: DataFrame,
            test_df: DataFrame,
            n_neighbors: int = 5):
    # filter unknowns
    train_df = train_df[train_df["attack_type"] != "unknown"]
    test_df = test_df[test_df["attack_type"] != "unknown"]

    # encode protocol
    le_protocol = LabelEncoder()
    train_df["protocol_encoded"] = le_protocol.fit_transform(train_df["protocol"])
    test_df["protocol_encoded"] = le_protocol.transform(test_df["protocol"])

    # encode labels
    le_attack = LabelEncoder()
    train_df["attack_type_encoded"] = le_attack.fit_transform(train_df["attack_type"])

    if not set(test_df["attack_type"]).issubset(set(le_attack.classes_)):
        print("Test set contains unseen attack types.")
        return

    test_df["attack_type_encoded"] = le_attack.transform(test_df["attack_type"])

    # prepare data
    features = ["size", "protocol_encoded"]
    train_x = train_df[features]
    test_x = test_df[features]
    train_y = train_df["attack_type_encoded"]
    test_y = test_df["attack_type_encoded"]

    # train
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(train_x, train_y)

    # predict
    pred_y = model.predict(test_x)

    # convert back to strings for reporting
    test_labels_str = le_attack.inverse_transform(test_y)
    pred_labels_str = le_attack.inverse_transform(pred_y)
    labels_in_test = le_attack.inverse_transform(sorted(test_y.unique()))

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
