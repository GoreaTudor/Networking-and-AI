import pandas as pd
from pandas import DataFrame

__DEFAULT_FILE_NAME = "all_floods"
__CONVERTED_FILE_PREFIX = "..\\data\\converted\\"
__MERGED_FILE_PREFIX = "..\\data\\merged\\"
__UNKNOWN_LABEL = "unknown"


def load_packets_supervised_data(file_name: str = __DEFAULT_FILE_NAME,
                                 from_merged: bool = True) -> DataFrame:
    folder = __MERGED_FILE_PREFIX if from_merged else __CONVERTED_FILE_PREFIX

    file_path = folder + file_name + ".csv"
    df = pd.read_csv(file_path)

    # Drop rows where attack_type is "unknown"
    df_cleaned = df[df["attack_type"] != __UNKNOWN_LABEL].reset_index(drop=True)

    return df_cleaned


if __name__ == '__main__':
    df = load_packets_supervised_data("all_floods", from_merged=True)
    print(df.head())
