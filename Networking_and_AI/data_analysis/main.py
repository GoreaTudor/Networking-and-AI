from data_analysis.data_loader import load_packets_supervised_data

if __name__ == '__main__':
    df = load_packets_supervised_data(debug_mode=True)

    # checks if the labels are added correctly
    # there also seems to be a delay issue with the timing generated in csv
    for i in range(0, 1100, 10):
        print(f"\n{i}: {df.iloc[i].attack_type}")