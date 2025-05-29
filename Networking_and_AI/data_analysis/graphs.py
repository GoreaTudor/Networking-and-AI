import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.metrics import confusion_matrix


##### GENERAL GRAPHS #####
def draw_attack_type_distribution(df: DataFrame):
    plt.figure(figsize=(8, 6))
    attack_counts = df['attack_type'].value_counts()
    plt.bar(attack_counts.index, attack_counts.values, color='skyblue')
    plt.title('Attack Type Distribution')
    plt.xlabel('Attack Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def draw_protocol_usage_by_attack_type(df: DataFrame):
    plt.figure(figsize=(10, 6))
    protocol_attack_counts = df.groupby(['protocol', 'attack_type']).size().unstack(fill_value=0)
    protocol_attack_counts.plot(kind='bar', stacked=False, figsize=(10, 6))
    plt.title('Protocol Usage by Attack Type')
    plt.xlabel('Protocol')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def draw_packet_size_distribution_by_attack_type(df: DataFrame):
    plt.figure(figsize=(10, 6))
    attack_types = df['attack_type'].unique()
    data = [df[df['attack_type'] == at]['size'] for at in attack_types]
    plt.boxplot(data, labels=attack_types)
    plt.title('Packet Size Distribution by Attack Type')
    plt.xlabel('Attack Type')
    plt.ylabel('Packet Size')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


##### PREDICTION GRAPHS #####
def draw_confusion_matrix(test_y, pred_y, label_names):
    cm = confusion_matrix(test_y, pred_y, labels=label_names)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)

    # Annotate numbers inside squares
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
