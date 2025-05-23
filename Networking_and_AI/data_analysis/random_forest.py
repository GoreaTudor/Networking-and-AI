from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from data_loader import load_packets_supervised_data
from graphs import draw_confusion_matrix


def run_random_forest(df: DataFrame,
                      test_size=0.2,
                      random_state=42):
    print("Random Forest Classifier")

    # drop unknown rows
    df = df[df["attack_type"] != "unknown"]

    # encode 'protocol' field
    le_protocol = LabelEncoder()
    df["protocol_encoded"] = le_protocol.fit_transform(df["protocol"])

    # split
    x = df[["size", "protocol_encoded"]]
    y = df["attack_type"]

    train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y)

    # train
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(train_x, train_y)

    # test
    pred_y = clf.predict(test_x)
    report = classification_report(test_y, pred_y)

    draw_confusion_matrix(test_y, pred_y, clf.classes_)

    print(f"Report: {report}")


if __name__ == '__main__':
    dataframe = load_packets_supervised_data()
    run_random_forest(df=dataframe)
