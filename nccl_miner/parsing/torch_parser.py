'''
Parse relevant information from torch profiler data.
'''
import json
import os
import re

from nccl_miner.common.torch_event import *

def create_cuda_local_op(context_events):
    """
    Create a CUDA local operation from a list of context events.
    """
    # TODO: Naive implementation for now, need to deduce the semantics of the data
    op_name = context_events[-1]['name']
    return CudaLocal(op_name)

def create_all_gpu_local_ops(cpu_events):
    """
    Get overarching events of every event in the list using a sweep line algorithm.
    """
    events = []
    for event in cpu_events:
        start = event['ts']
        end = event['ts'] + event['dur']
        events.append((start, 1, event))  # Event start
        events.append((end, -1, event))  # Event end

    # Sort events by time, breaking ties by type (-1 before 1 for same timestamp)
    events.sort(key=lambda x: (x[0], x[1]))

    all_gpu_local_ops = []
    active_events = []

    for time, event_type, event in events:
        if event_type == 1:  # Event start
            active_events.append(event)
        elif event_type == -1:  # Event end
            active_events.remove(event)
            # Only care about events with Cuda
            if event['name'] == "cudaLaunchKernel" or \
                event['name'] == "cuLaunchKernel":
                cuda_local = create_cuda_local_op(active_events)
                cuda_local.associate_cpu_event(event)
                all_gpu_local_ops.append(cuda_local)

    return all_gpu_local_ops

def get_host_pid(filename):
    filename_pattern = r"(?P<host>[a-z0-9]+)_(?P<pid>\d+)\.(?P<garbage>\d+)\.pt\.trace\.json"
    match = re.match(filename_pattern, filename)
    if match:
        return match['host'], int(match['pid'])
    else:
        return None, None

def parse_torch_logs(log_files):
    gpu_comm_ops_per_device = {}
    gpu_local_ops_per_device = {}
    for log_file in log_files:
        file_name = os.path.basename(log_file)
        print(f"Parsing log file {file_name}..")
        host, pid = get_host_pid(file_name)
        print(f"Host: {host}, PID: {pid}")

        cur_device = None
        cpu_events = []
        nccl_kernel_events = []
        local_kernel_events = []
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
                            gpu_comm_ops_per_device[cur_device] = {'host':host,
                                                                'pid':pid,
                                                                'operations':[]}
                        else:
                            assert cur_device == event['pid']
                        if event['cat'] == "kernel":
                            if re.match(r"ncclDevKernel_.*", event['name']):
                                nccl_kernel_events.append(event)
                            else:
                                local_kernel_events.append(event)

        cpu_events.sort(key=lambda x: x['ts'] + x['dur'] if 'dur' in x else x['ts'])
        nccl_kernel_events.sort(key=lambda x: x['ts'])

        # For NCCL kernel ops
        context_events = []
        device_gpu_comm_ops = []
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
                                device_gpu_comm_ops.append(cuda_ptp)
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
                        device_gpu_comm_ops.append(cuda_coll)
                        # clear context
                        context_events = []

            if is_context:
                context_events.append(event)

        for op in device_gpu_comm_ops:
            if op.kernel_id is not None:
                kernel_op = nccl_kernel_events[op.kernel_id]
                op.associate_kernel(kernel_op)

        gpu_comm_ops_per_device[cur_device]['operations'] = device_gpu_comm_ops

        # For CUDA local kernel ops
        gpu_local_ops_per_device[cur_device] = []

        events_per_tid = {}
        for event in cpu_events:
            if event['tid'] not in events_per_tid:
                events_per_tid[event['tid']] = []
            events_per_tid[event['tid']].append(event)
        
        device_gpu_local_ops = []
        for tid, events in events_per_tid.items():
            # CPU ops that can be used to deduce semantics of data
            device_gpu_local_ops.extend(create_all_gpu_local_ops(events))
        device_gpu_local_ops.sort(key=lambda x: x.cpu_start_time)
       
        local_kernel_events.sort(key=lambda x: x['ts'])

        assert len(local_kernel_events) == len(device_gpu_local_ops)
        for i in range(len(local_kernel_events)):
            local_op = device_gpu_local_ops[i]
            assert isinstance(local_op, CudaLocal)
            local_op.associate_kernel(local_kernel_events[i])
            gpu_local_ops_per_device[cur_device].append(local_op)

    return gpu_local_ops_per_device, gpu_comm_ops_per_device


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