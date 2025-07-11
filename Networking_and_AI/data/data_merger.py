import pandas as pd
import os

__CONVERTED_FILE_PREFIX = ".\\converted\\"
__MERGED_FILE_PREFIX = ".\\merged\\"


def merge_csv_files_continuous(file_names: list[str], output_file_name: str):
    print(f"Generating {output_file_name}...")
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


def generate_all_x():
    ### FLOODS ###
    merge_csv_files_continuous(
        file_names=[
            "ping_flood_0", "syn_flood_0", "fin_flood_0", "rst_flood_0", "udp_flood_0",
            "ping_flood_1", "syn_flood_1", "fin_flood_1", "rst_flood_1", "udp_flood_1",
        ],
        output_file_name="all_floods"
    )

    merge_csv_files_continuous(
        file_names=["normal_0",
                    "ping_flood_0", "normal_0",
                    "syn_flood_0", "normal_0",
                    "fin_flood_0", "normal_0",
                    "rst_flood_0", "normal_0",
                    "udp_flood_0", "normal_0",
                    "ping_flood_1", "normal_0",
                    "syn_flood_1", "normal_0",
                    "fin_flood_1", "normal_0",
                    "rst_flood_1", "normal_0",
                    "udp_flood_1", "normal_0",
                    ],
        output_file_name="all_floods_n"
    )

    ### DDOS ###
    merge_csv_files_continuous(
        file_names=[
            "ping_ddos_0", "syn_ddos_0", "fin_ddos_0", "rst_ddos_0", "udp_ddos_0",
            "ping_ddos_1", "syn_ddos_1", "fin_ddos_1", "rst_ddos_1", "udp_ddos_1",
        ],
        output_file_name="all_ddos"
    )

    merge_csv_files_continuous(
        file_names=["normal_0",
                    "ping_ddos_0", "normal_0",
                    "syn_ddos_0", "normal_0",
                    "fin_ddos_0", "normal_0",
                    "rst_ddos_0", "normal_0",
                    "udp_ddos_0", "normal_0",
                    "ping_ddos_1", "normal_0",
                    "syn_ddos_1", "normal_0",
                    "fin_ddos_1", "normal_0",
                    "rst_ddos_1", "normal_0",
                    "udp_ddos_1", "normal_0",
                    ],
        output_file_name="all_ddos_n"
    )

    ### PSCAN ###
    merge_csv_files_continuous(
        file_names=["pscan_test", "pscan_0", "pscan_1"],
        output_file_name="all_pscan"
    )

    merge_csv_files_continuous(
        file_names=["normal_0",
                    "pscan_test", "normal_0",
                    "pscan_0", "normal_0",
                    "pscan_1", "normal_0"
                    ],
        output_file_name="all_pscan_n"
    )

    ### R-DOS ###
    merge_csv_files_continuous(
        file_names=["reflected_dos_0", "reflected_dos_1"],
        output_file_name="all_reflected_dos"
    )

    merge_csv_files_continuous(
        file_names=["normal_0", "reflected_dos_0", "normal_0", "reflected_dos_1", "normal_0"],
        output_file_name="all_reflected_dos_n"
    )

    ### OTHER ###
    merge_csv_files_continuous(
        file_names=["pscan_test", "pscan_0", "pscan_1", "reflected_dos_0", "reflected_dos_1"],
        output_file_name="all_other"
    )

    merge_csv_files_continuous(
        file_names=["normal_0",
                    "pscan_test", "normal_0",
                    "pscan_0", "normal_0",
                    "pscan_1", "normal_0",
                    "reflected_dos_0", "normal_0",
                    "reflected_dos_1", "normal_0",
                    ],
        output_file_name="all_other_n"
    )


def generate_client_mixes():
    merge_csv_files_continuous(
        file_names=[
            "normal_0", "normal_0",
            "ping_flood_0", "syn_flood_0", "normal_0",
            "fin_flood_1", "rst_flood_0", "udp_flood_1", "normal_0",
            "ping_ddos_0", "syn_ddos_1", "normal_0",
            "fin_ddos_0", "rst_ddos_1", "udp_ddos_0",
            "reflected_dos_1", "pscan_0", "normal_0"
        ],
        output_file_name="client_mix_1"
    )

    merge_csv_files_continuous(
        file_names=[
            "normal_0", "normal_0", "normal_0",
            "ping_flood_1", "syn_flood_1", "fin_flood_0", "normal_0",
            "rst_flood_1", "udp_flood_0", "ping_ddos_1",
            "syn_ddos_0", "normal_0", "syn_ddos_0",
            "fin_ddos_1", "rst_ddos_0", "udp_ddos_1",
            "reflected_dos_0", "pscan_1", "normal_0"
        ],
        output_file_name="client_mix_2"
    )

    merge_csv_files_continuous(
        file_names=[
            "normal_0", "normal_0", "normal_0",
            "ping_flood_0", "syn_flood_1", "normal_0",
            "fin_flood_0", "rst_flood_1", "udp_flood_1",
            "ping_ddos_1", "syn_ddos_1", "normal_0",
            "fin_ddos_0", "rst_ddos_1", "udp_ddos_0",
            "reflected_dos_1", "pscan_test", "pscan_test",
            "normal_0"
        ],
        output_file_name="client_mix_3"
    )

    merge_csv_files_continuous(
        file_names=[
            "normal_0", "normal_0", "normal_0", "normal_0",
            "ping_flood_1", "syn_flood_0", "fin_flood_1",
            "rst_flood_1", "udp_flood_0", "normal_0",
            "ping_ddos_0", "ping_ddos_0",
            "syn_ddos_0", "fin_ddos_1", "rst_ddos_0", "udp_ddos_1",
            "reflected_dos_0", "pscan_0", "normal_0"
        ],
        output_file_name="client_mix_4"
    )

    merge_csv_files_continuous(
        file_names=[
            "normal_0", "normal_0", "normal_0", "normal_0", "normal_0",
            "ping_flood_0", "syn_flood_1", "fin_flood_0", "rst_flood_0", "udp_flood_1",
            "ping_ddos_1", "syn_ddos_1", "fin_ddos_0", "rst_ddos_1", "udp_ddos_0",
            "reflected_dos_1", "pscan_1", "normal_0"
        ],
        output_file_name="client_mix_5"
    )


def generate_testers():
    # tester
    merge_csv_files_continuous(
        file_names=[
            "normal_0",
            "ping_flood_0", "syn_flood_0", "fin_flood_0", "rst_flood_0", "udp_flood_0",
            "ping_ddos_0", "syn_ddos_0", "fin_ddos_0", "rst_ddos_0", "udp_ddos_0",
            "reflected_dos_0", "pscan_0",
            "normal_0",
            "ping_flood_1", "syn_flood_1", "fin_flood_1", "rst_flood_1", "udp_flood_1",
            "ping_ddos_1", "syn_ddos_1", "fin_ddos_1", "rst_ddos_1", "udp_ddos_1",
            "reflected_dos_1", "pscan_1",
            "normal_0"
        ],
        output_file_name="full_tester"
    )

    # shuffled
    merge_csv_files_continuous(
        file_names=[
            "normal_0", "reflected_dos_1", "syn_flood_0", "udp_ddos_0", "pscan_0",
            "rst_flood_1", "ping_ddos_1", "normal_0", "fin_flood_0", "ping_flood_1",
            "udp_flood_1", "syn_ddos_0", "reflected_dos_0", "fin_ddos_1", "normal_0",
            "rst_ddos_1", "pscan_1", "ping_flood_0", "syn_flood_1", "fin_ddos_0",
            "ping_ddos_0", "rst_ddos_0", "normal_0", "udp_flood_0", "syn_ddos_1",
            "fin_flood_1", "rst_flood_0", "udp_ddos_1"
        ],
        output_file_name="full_tester_shuffled"
    )

    # shuffled + repeats = balanced
    merge_csv_files_continuous(
        file_names=[
            "normal_0", "normal_0", "ping_flood_0", "normal_0", "syn_ddos_0",
            "reflected_dos_0", "normal_0", "pscan_1", "udp_flood_1", "normal_0",
            "ping_ddos_1", "syn_flood_1", "normal_0", "normal_0", "fin_flood_0",
            "rst_ddos_0", "reflected_dos_1", "ping_ddos_1", "normal_0", "pscan_0",
            "fin_ddos_0", "normal_0", "udp_ddos_1", "syn_ddos_1", "rst_flood_1",
            "normal_0", "ping_flood_1", "fin_ddos_1", "udp_flood_0", "normal_0",
            "syn_flood_0", "normal_0", "rst_flood_0"
        ],
        output_file_name="full_tester_balanced"
    )


if __name__ == '__main__':
    ### generate std cases
    generate_all_x()
    generate_client_mixes()
    generate_testers()

    # merge_csv_files_continuous(
    #     file_names=["ping_flood_1", "syn_flood_1", "fin_flood_1", "rst_flood_1", "udp_flood_1"],
    #     output_file_name="all_floods_1"
    # )
