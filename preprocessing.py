from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scapy.compat import raw
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse

from utils import should_omit_packet, read_pcap, PREFIX_TO_APP_ID, PREFIX_TO_TRAFFIC_ID


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet


def packet_to_sparse_array(packet, max_length=1500):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] / 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)

    arr = sparse.csr_matrix(arr)
    return arr


def transform_packet(packet):
    if should_omit_packet(packet):
        return None

    packet = remove_ether_header(packet)
    packet = pad_udp(packet)
    packet = mask_ip(packet)

    arr = packet_to_sparse_array(packet)

    return arr


def transform_pcap(path, output_path: Path = None, output_batch_size=10000):
    if Path(str(output_path.absolute()) + '_SUCCESS').exists():
        print(output_path, 'Done')
        return

    print('Processing', path)

    rows = []
    batch_index = 0
    for i, packet in enumerate(read_pcap(path)):
        arr = transform_packet(packet)
        if arr is not None:
            # get labels for app identification
            prefix = path.name.split('.')[0].lower()
            app_label = PREFIX_TO_APP_ID.get(prefix)
            traffic_label = PREFIX_TO_TRAFFIC_ID.get(prefix)
            row = {
                'app_label': app_label,
                'traffic_label': traffic_label,
                'feature': arr.todense().tolist()[0]
            }
            rows.append(row)

        # write every batch_size packets, by default 10000
        if rows and i > 0 and i % output_batch_size == 0:
            part_output_path = Path(str(output_path.absolute()) + f'_part_{batch_index:04d}.parquet')
            df = pd.DataFrame(rows)
            df.to_parquet(part_output_path)
            batch_index += 1
            rows.clear()

    # final write
    if rows:
        df = pd.DataFrame(rows)
        part_output_path = Path(str(output_path.absolute()) + f'_part_{batch_index:04d}.parquet')
        df.to_parquet(part_output_path)

    # write success file
    with Path(str(output_path.absolute()) + '_SUCCESS').open('w') as f:
        f.write('')

    print(output_path, 'Done')


@click.command()
@click.option('-s', '--source', help='path to the directory containing raw pcap files', required=True)
@click.option('-t', '--target', help='path to the directory for persisting preprocessed files', required=True)
@click.option('-n', '--njob', default=-1, help='num of executors', type=int)
def main(source, target, njob):
    data_dir_path = Path(source)
    target_dir_path = Path(target)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    if njob == 1:
        for pcap_path in sorted(data_dir_path.iterdir()):
            transform_pcap(pcap_path, target_dir_path / (pcap_path.name + '.transformed'))
    else:
        Parallel(n_jobs=njob)(
            delayed(transform_pcap)(pcap_path, target_dir_path / (pcap_path.name + '.transformed')) for pcap_path in
            sorted(data_dir_path.iterdir()))


if __name__ == '__main__':
    main()
