import os
import argparse

from nccl_miner.visualizer import generate_chrome_trace

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nccl_log_path", type=str, help="The directory which contains NCCL logs.")
    parser.add_argument("-t", "--torch_profiler_path", default=None, type=str, help="The directory which contains torch profiler output.")
    parser.add_argument("-o", "--output_path", default="nccl_trace.json", type=str, help="The path of the output trace JSON file.")
    args = parser.parse_args()

    nccl_log_files = []
    for log_file in os.listdir(args.nccl_log_path):
        log_path = os.path.join(args.nccl_log_path, log_file)
        if os.path.isfile(log_path):
            nccl_log_files.append(log_path)

    torch_files = []
    if args.torch_profiler_path is not None:
        for log_file in os.listdir(args.torch_profiler_path):
            log_path = os.path.join(args.torch_profiler_path, log_file)
            if os.path.isfile(log_path):
                torch_files.append(log_path)

    generate_chrome_trace(nccl_log_files, torch_files, args.output_path)

