import os
import argparse

from nccl_miner.misc.logger import *
from nccl_miner.miner_pipeline import mine_torch_nccl_pipeline
from nccl_miner.trace_generator import generate_perfetto_trace
from nccl_miner.flow_dumper import dump_data_flows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("nccl_log_path", type=str, help="The directory which contains NCCL logs.")
    parser.add_argument("torch_profiler_path", type=str, help="The directory which contains torch profiler output.")
    parser.add_argument("-o", "--trace_output", default="nccl_trace.json", type=str, help="The path of the output trace JSON file.")
    parser.add_argument("-d", "--dump_output", default="flows_dump.json", type=str, help="The path to dump data flows as a JSON file.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args()

    import logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        handlers=[logging.StreamHandler()]  # Output to terminal (stdout)
    )

    nccl_log_files = []
    for log_file in os.listdir(args.nccl_log_path):
        log_path = os.path.join(args.nccl_log_path, log_file)
        if os.path.isfile(log_path):
            nccl_log_files.append(log_path)

    torch_files = []
    for log_file in os.listdir(args.torch_profiler_path):
        log_path = os.path.join(args.torch_profiler_path, log_file)
        if os.path.isfile(log_path):
            torch_files.append(log_path)

    logi("Running Nccl Miner Pipeline...")
    cpu_ops, gpu_ops, data_flow_groups = mine_torch_nccl_pipeline(nccl_log_files, torch_files)

    logi("Generating trace visualizations...")
    generate_perfetto_trace(cpu_ops, gpu_ops, data_flow_groups, args.trace_output)

    logi("Dumping data flows...")
    dump_data_flows(data_flow_groups, args.dump_output)