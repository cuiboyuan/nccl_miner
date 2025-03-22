import os
import argparse

from nccl_miner.visualizer import generate_chrome_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="The directory which contains NCCL logs to parse.")
    parser.add_argument("-o", "--output_file", default="nccl_trace.json", type=str, help="The name of the output trace JSON file.")
    args = parser.parse_args()

    nccl_log_files = []
    for log_file in os.listdir(args.log_dir):
        log_path = os.path.join(args.log_dir, log_file)
        if os.path.isfile(log_path):
            nccl_log_files.append(log_path)
    print(nccl_log_files)
    generate_chrome_trace(nccl_log_files, args.output_file)

