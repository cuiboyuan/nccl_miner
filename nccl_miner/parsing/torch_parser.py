'''
Parse relevant information from torch profiler data.
'''
import json
import os
import re

from nccl_miner.common.torch_event import *
from ..common.torch_utils import *

def deduce_high_level_semantics(cuda_local_ops):
    sorted_cuda_local_ops = sorted(cuda_local_ops, key=lambda x: x.start_time)
    high_level_op = []
    for op in sorted_cuda_local_ops:
        assert isinstance(op, CudaLocal)
        if is_forward_pass(op.name):
            high_level_op.append(("Forward Pass", op.start_time, op.end_time))
        elif is_backward_pass(op.name):
            high_level_op.append(("Backward Pass", op.start_time, op.end_time))
        elif is_optimizer_step(op.name):
            high_level_op.append(("Optimizer Step", op.start_time, op.end_time))
        elif is_c10d_communication(op.name):
            high_level_op.append(("C10D Communication", op.start_time, op.end_time))
        else:
            high_level_op.append(("Misc Operations", op.start_time, op.end_time))

    combined_high_level_op = []
    cur_op_type, cur_start_ts, cur_end_ts = high_level_op[0]
    for op_type, start_time, end_time in high_level_op[1:]:
        if op_type == cur_op_type:
            cur_end_ts = end_time
        else:
            combined_high_level_op.append((cur_op_type, cur_start_ts, cur_end_ts))
            cur_op_type = op_type
            cur_start_ts = start_time
            cur_end_ts = end_time
    # complete last events
    combined_high_level_op.append((cur_op_type, cur_start_ts, cur_end_ts))

    semantical_ops = []
    for op_type, start_time, end_time in combined_high_level_op:
        high_level_sem = CudaLocal(op_type)
        high_level_sem.associate_timestamps(start_time, end_time)
        semantical_ops.append(high_level_sem)

    return semantical_ops

def get_device_gpu_ops(cur_device, cpu_events, nccl_kernel_events, local_kernel_events):

    ####
    # Step 1. Find all CUDA and NCCL ops on CPU
    # Based on CPU timestamps, find relevant contextual events that
    # provide insights on the content of data in CUDA and NCCL ops
    ####

    # Group events by threads
    events_per_tid = {}
    for event in cpu_events:
        if event['tid'] not in events_per_tid:
            events_per_tid[event['tid']] = []
        events_per_tid[event['tid']].append(event)

    device_gpu_comm_ops = []
    device_gpu_local_ops = []
    for tid, thread_events in events_per_tid.items():
        # Get overarching events of every event in the list using a sweep line algorithm.
        sorted_events = []
        for event in thread_events:
            start = event['ts']
            end = event['ts'] + event['dur']
            sorted_events.append((start, 1, event))  # Event start
            sorted_events.append((end, -1, event))  # Event end

        # Sort events by time, breaking ties by type (-1 before 1 for same timestamp)
        sorted_events.sort(key=lambda x: (x[0], x[1]))

        active_events = []
        past_relevant_events = []

        past_relevant_local_cuda_ops = []

        thread_gpu_local_ops = []
        thread_gpu_comm_ops = []
        for time, event_type, event in sorted_events:
            if event_type == 1:  # Event start
                active_events.append(event)
            elif event_type == -1:  # Event end
                active_events.remove(event)
                past_relevant_events.append(event)
                if event['name'] == "cudaLaunchKernel" or \
                    event['name'] == "cuLaunchKernel":
                    ## CUDA local kernel ops
                    cuda_local = CudaLocal(device=cur_device)
                    cuda_local.associate_context_events(active_events, [])
                    cuda_local.associate_cpu_event(event)
                    thread_gpu_local_ops.append(cuda_local)
                    past_relevant_local_cuda_ops.append(cuda_local)

                elif "nccl:" in event['name'] and event['cat'] == "user_annotation":
                    ## NCCL kernel ops
                    nccl_pattern = r"nccl:(\w+)"
                    match = re.match(nccl_pattern, event['name'])
                    if match:
                        # Check if this NCCL call has a corresponding CUDA kernel op.
                        kernel_exists = False
                        for context_event in reversed(past_relevant_events):
                            if context_event['name'] == "cudaLaunchKernelExC" and \
                                    context_event['cat'] == "cuda_runtime":
                                kernel_exists = True
                                break

                        func = match.groups()[0]
                        if func == "coalesced":
                            # TODO: use context events to deduce meaning of the data
                            # ...
                            cuda_coalesced = []
                            for idx, context_event in enumerate(past_relevant_events):
                                if context_event['name'] in ['c10d::send', 'c10d::recv_']:
                                    if context_event['name'] == 'c10d::send':
                                        cuda_ptp = CudaPtp("Send", device=cur_device)
                                        cuda_ptp.associate_context_events([], past_relevant_events[:idx],
                                                                          past_local_ops=past_relevant_local_cuda_ops)
                                    else: # c10d::recv_
                                        cuda_ptp = CudaPtp("Recv", device=cur_device)
                                    cuda_ptp.associate_cpu_event(context_event)
                                    if kernel_exists:
                                        cuda_ptp.has_kernel = True
                                    cuda_coalesced.append(cuda_ptp)
                            thread_gpu_comm_ops.append(cuda_coalesced)
                            # clear context
                            past_relevant_events = []
                            past_relevant_local_cuda_ops = []
                        else:
                            # print(event)
                            if func == "all_reduce":
                                func_name = "AllReduce"
                            elif func == "broadcast":
                                func_name = "Broadcast"
                            else:
                                func_name = func

                            # TODO: use context events to deduce meaning of the data
                            # ...
                            cuda_coll = CudaCollective(func_name, device=cur_device)
                            cuda_coll.associate_context_events(active_events, past_relevant_events,
                                                               past_local_ops=past_relevant_local_cuda_ops)
                            cuda_coll.associate_cpu_event(event)
                            if kernel_exists:
                                cuda_coll.has_kernel = True
                            thread_gpu_comm_ops.append([cuda_coll])
                            # clear context
                            past_relevant_events = []
                            past_relevant_local_cuda_ops = []

        device_gpu_local_ops.extend(thread_gpu_local_ops)
        device_gpu_comm_ops.extend(thread_gpu_comm_ops)

    ####
    # Step 2. Associate CUDA and NCCL ops with their corresponding kernel ops
    # Kernel provides more information, like actual time of execution.
    ####
    # sort NCCL kernels by start time, it shouldn't matter whether to sort by start or end time since there's no overlap btw NCCL kernels
    nccl_kernel_events.sort(key=lambda x: x['ts'])
    cur_nccl_kernel_id = 0
    # sort GPU comm ops by start time
    device_gpu_comm_ops.sort(key=lambda x: x[0].cpu_start_time)
    device_gpu_nccl_ops = []
    for op_list in device_gpu_comm_ops:
        kernel_exists = False
        for op in op_list:
            if op.has_kernel:
                kernel_exists = True
                op.associate_kernel(nccl_kernel_events[cur_nccl_kernel_id])
            device_gpu_nccl_ops.append(op)
        if kernel_exists:
            cur_nccl_kernel_id += 1
    device_gpu_nccl_ops.sort(key=lambda x: x.cpu_start_time)

    # sort local CUDA kernels by start time, it shouldn't matter whether to sort by start or end time since there's no overlap btw local CUDA kernels
    local_kernel_events.sort(key=lambda x: x['ts'])
    # sort GPU local ops by start time
    device_gpu_local_ops.sort(key=lambda x: x.cpu_start_time)
    assert len(local_kernel_events) == len(device_gpu_local_ops)

    for idx, local_op in enumerate(device_gpu_local_ops):
        assert isinstance(local_op, CudaLocal)
        local_op.associate_kernel(local_kernel_events[idx])

    ####
    # Step 3. Deduce high-level semantics information from Local CUDA and NCCL ops
    ####
    # deduce high-level semantics for local ops
    semantics_ops = deduce_high_level_semantics(device_gpu_local_ops)
    device_gpu_local_ops.extend(semantics_ops)

    return device_gpu_nccl_ops, device_gpu_local_ops

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

        device_gpu_comm_ops, device_gpu_cuda_ops = get_device_gpu_ops(cur_device, cpu_events, nccl_kernel_events, local_kernel_events)
        gpu_comm_ops_per_device[cur_device]['operations'] = device_gpu_comm_ops
        gpu_local_ops_per_device[cur_device] = device_gpu_cuda_ops

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