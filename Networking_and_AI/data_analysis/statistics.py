from data_analysis.graphs import draw_packet_size_distribution_by_attack_type, draw_protocol_usage_by_attack_type
from data_loader import load_packets_supervised_data
from graphs import draw_attack_type_distribution


def run_statistics():
    df = load_packets_supervised_data()

    draw_attack_type_distribution(df)
    draw_protocol_usage_by_attack_type(df)
    draw_packet_size_distribution_by_attack_type(df)


if __name__ == '__main__':
    run_statistics()
