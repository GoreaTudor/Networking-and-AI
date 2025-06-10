import pandas as pd
import os

__CONVERTED_FILE_PREFIX = ".\\converted\\"
__MERGED_FILE_PREFIX = ".\\merged\\"


def merge_csv_files_continuous(file_names: list[str], output_file_name: str):
    merged_data = []
    time_offset = 0.0

    for file_name in file_names:
        file_path = os.path.join(__CONVERTED_FILE_PREFIX, file_name + ".csv")
        df = pd.read_csv(file_path)

        # Shift time
        df["time"] = df["time"] + time_offset

        # Update offset for next file
        if not df.empty:
            time_offset = df["time"].max()

        merged_data.append(df)

    if merged_data:
        final_df = pd.concat(merged_data, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    # Ensure output directory exists
    os.makedirs(__MERGED_FILE_PREFIX, exist_ok=True)

    # Save merged dataframe to CSV
    output_path = os.path.join(__MERGED_FILE_PREFIX, output_file_name + ".csv")
    final_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    merge_csv_files_continuous(
        file_names=["ping_flood_1", "syn_flood_1", "fin_flood_1", "rst_flood_1", "udp_flood_1"],
        output_file_name="all_floods_1"
    )
