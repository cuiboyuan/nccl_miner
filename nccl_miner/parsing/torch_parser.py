'''
Parse relevant information from torch profiler data.
'''
import json
import argparse
import os
import re

from nccl_miner.common.torch_event import CudaComm

def get_host_pid(filename):
    filename_pattern = r"(?P<host>[a-z0-9]+)_(?P<pid>\d+).(?P<garbage>\d+).pt.trace.json"
    match = re.match(filename_pattern, filename)
    if match:
        return match['host'], int(match['pid'])
    else:
        return None, None

def parse_torch_logs(log_files):
    torch_ops_per_device = {}
    for log_file in log_files:
        file_name = os.path.basename(log_file)
        print(f"Parsing log file {file_name}..")
        host, pid = get_host_pid(file_name)
        print(f"Host: {host}, PID: {pid}")

        cur_device = None
        with open(log_file, "r") as f:
            trace = json.load(f)
            events = trace['traceEvents']
            for event in events:
                if event['ph'] == "X":
                    if event['pid'] == pid:
                        # CPU ops
                        pass
                    elif isinstance(event['pid'], int):
                        # GPU ops
                        if cur_device is None:
                            cur_device = event['pid']
                            print(f"GPU {cur_device}")
                            torch_ops_per_device[cur_device] = {'host':host,
                                                                'pid':pid,
                                                                'operations':[]}
                        else:
                            assert cur_device == event['pid']
                        
                        if event['cat'] == "kernel":
                            cuda_comm = CudaComm.parse(event)
                            if cuda_comm is not None:
                                # TODO: add cpu info to cuda ops
                                torch_ops_per_device[cur_device]['operations'].append(cuda_comm)
    return {}, torch_ops_per_device

# Local testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="The directory which contains torch logs to parse.")
    parser.add_argument("-o", "--output_file", default="torch_trace.json", type=str, help="The name of the output trace JSON file.")
    args = parser.parse_args()

    torch_log_files = []
    for log_file in os.listdir(args.log_dir):
        log_path = os.path.join(args.log_dir, log_file)
        if os.path.isfile(log_path):
            torch_log_files.append(log_path)
    print(torch_log_files)
    parse_torch_logs(torch_log_files)