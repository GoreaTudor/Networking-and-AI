import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def draw_flag_distribution_by_attack_type(df: DataFrame):
    flag_attack_counts = df.groupby(['flags', 'attack_type']).size().unstack(fill_value=0)
    flag_attack_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
    plt.title('TCP Flag Distribution by Attack Type')
    plt.xlabel('TCP Flags')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def draw_src_port_distribution_by_attack_type(df: DataFrame, top_n=10):
    top_ports = df['src_port'].value_counts().nlargest(top_n).index
    filtered = df[df['src_port'].isin(top_ports)]
    port_attack_counts = filtered.groupby(['src_port', 'attack_type']).size().unstack(fill_value=0)
    port_attack_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set3')
    plt.title(f'Top {top_n} Source Ports by Attack Type')
    plt.xlabel('Source Port')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def draw_dst_port_distribution_by_attack_type(df: DataFrame, top_n=10):
    top_ports = df['dst_port'].value_counts().nlargest(top_n).index
    filtered = df[df['dst_port'].isin(top_ports)]
    port_attack_counts = filtered.groupby(['dst_port', 'attack_type']).size().unstack(fill_value=0)
    port_attack_counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set1')
    plt.title(f'Top {top_n} Destination Ports by Attack Type')
    plt.xlabel('Destination Port')
    plt.ylabel('Count')
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


def draw_weights_graph(all_weights, averaged_weights):
    plt.figure(figsize=(12, 6))
    for i, weights in enumerate(all_weights):
        flattened_weights = np.concatenate([w.flatten() for w in weights])
        plt.plot(range(len(flattened_weights)), flattened_weights, label=f"Model {i + 1}")
    aggregated_flattened_weights = np.concatenate([w.flatten() for w in averaged_weights])
    plt.plot(range(len(aggregated_flattened_weights)), aggregated_flattened_weights, label="Aggregated Model",
             linestyle="-", color="black")
    plt.title("Model Weights Visualization")
    plt.xlabel("Weight Index")
    plt.ylabel("Weight Value")
    plt.legend()
    plt.show()
