import pandas as pd

from data_analysis.graphs import draw_protocol_usage_by_attack_type, \
    draw_flag_distribution_by_attack_type, draw_src_port_distribution_by_attack_type, \
    draw_dst_port_distribution_by_attack_type
from graphs import draw_attack_type_distribution


def run_statistics(filename: str):
    filepath = "..\\data\\merged\\" + filename + ".csv"

    # load dataset
    df = pd.read_csv(filepath)
    print(f"\n>> Loaded dataset: {filepath}")
    print("=" * 50)

    # shape + columns
    print(f">> Rows: {df.shape[0]}")
    print(f">> Columns: {df.shape[1]}")
    print(f">> Column Names: {list(df.columns)}")

    # missing values
    print("\n>> Missing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")

    # Unique attack types
    print("\n>> Attack Types:")
    attack_counts = df['attack_type'].value_counts()
    for label, count in attack_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  - {label}: {count} ({percentage:.2f}%)")

    # Basic stats for size, ports
    print("\n>> Numerical Feature Summary:")
    numeric_cols = ['size', 'src_port', 'dst_port']
    print(df[numeric_cols].describe())

    # Correlation matrix
    print("\n>> Correlation between numerical features:")
    corr = df[numeric_cols].corr()
    print(corr)

    # Visuals
    print("\n>> Generating visualizations...")
    draw_attack_type_distribution(df)
    draw_protocol_usage_by_attack_type(df)
    draw_flag_distribution_by_attack_type(df)
    draw_src_port_distribution_by_attack_type(df)
    draw_dst_port_distribution_by_attack_type(df)


if __name__ == '__main__':
    # run_statistics(filename="full_tester")
    run_statistics(filename="full_tester_balanced")
