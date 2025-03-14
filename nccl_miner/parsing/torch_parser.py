'''
Parse relevant information from torch profiler data.
'''
import json
import os
import re

from nccl_miner.common.torch_event import *


def get_host_pid(filename):
    filename_pattern = r"(?P<host>[a-z0-9]+)_(?P<pid>\d+)\.(?P<garbage>\d+)\.pt\.trace\.json"
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
        cpu_events = []
        nccl_kernel_events = []
        with open(log_file, "r") as f:
            trace = json.load(f)
            events = trace['traceEvents']
            for event in events:
                if event['ph'] == "X":
                    if event['pid'] == pid:
                        # CPU events
                        cpu_events.append(event)
                    elif isinstance(event['pid'], int):
                        if cur_device is None:
                            cur_device = event['pid']
                            print(f"GPU {cur_device}")
                            torch_ops_per_device[cur_device] = {'host':host,
                                                                'pid':pid,
                                                                'operations':[]}
                        else:
                            assert cur_device == event['pid']
                        if event['cat'] == "kernel":
                            if re.match(r"ncclDevKernel_.*", event['name']):
                                nccl_kernel_events.append(event)
                            
        cpu_events.sort(key=lambda x: x['ts'] + x['dur'] if 'dur' in x else x['ts'])
        nccl_kernel_events.sort(key=lambda x: x['ts'] + x['dur'] if 'dur' in x else x['ts'])

        context_events = []
        torch_gpu_ops = []
        cur_kernel_id = 0
        for event in cpu_events:
            is_context = True
            if event['cat'] == "user_annotation":
                nccl_pattern = r"nccl:(\w+)"
                match = re.match(nccl_pattern, event['name'])
                if match:
                    func = match.groups()[0]
                    if func == "coalesced":
                        is_context = False
                        # TODO: use context events to deduce meaning of the data
                        # ...
                        kernel_exists = False
                        for context_event in reversed(context_events):
                            if context_event['name'] == "cudaLaunchKernelExC" and \
                                 context_event['cat'] == "cuda_runtime":
                                kernel_exists = True
                                break
                        if kernel_exists:
                            sendrecv_kernel_id = cur_kernel_id
                            cur_kernel_id += 1
                        for context_event in context_events:
                            if context_event['name'] in ['c10d::send', 'c10d::recv_']:
                                cuda_ptp = CudaPtp("SendRecv", device=cur_device)
                                cuda_ptp.associate_kernel_id(sendrecv_kernel_id)
                                torch_gpu_ops.append(cuda_ptp)
                        # clear context
                        context_events = []
                    else:
                        # print(event)
                        if func == "all_reduce":
                            func_name = "AllReduce"
                        elif func == "broadcast":
                            func_name = "Broadcast"
                        else:
                            func_name = func

                        is_context = False
                        # TODO: use context events to deduce meaning of the data
                        # ...
                        kernel_exists = False
                        for context_event in reversed(context_events):
                            if context_event['name'] == "cudaLaunchKernelExC" and \
                                 context_event['cat'] == "cuda_runtime":
                                kernel_exists = True
                                break
                        cuda_coll = CudaCollective(func_name, device=cur_device)
                        if kernel_exists:
                            cuda_coll.associate_kernel_id(cur_kernel_id)
                            cur_kernel_id += 1
                        torch_gpu_ops.append(cuda_coll)
                        # clear context
                        context_events = []

            if is_context:
                context_events.append(event)

        for op in torch_gpu_ops:
            if op.kernel_id is not None:
                kernel_op = nccl_kernel_events[op.kernel_id]
                op.associate_kernel(kernel_op)

        torch_ops_per_device[cur_device]['operations'] = torch_gpu_ops
        # print("Sorting operations..")
        # torch_ops_per_device[cur_device]['operations'].sort(key=lambda x: x.start_time)
    return {}, torch_ops_per_device


# Local testing
if __name__ == "__main__":
    import argparse

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