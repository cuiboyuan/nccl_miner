'''
Orchestrate parsing from different logs files and correlating information.
'''
from .nccl_parser import parse_nccl_logs
from .torch_parser import parse_torch_logs
from .combiner import group_nccl_colls, link_nccl_torch_calls

def parse_torch_nccl_pipeline(nccl_log_files, torch_log_files):
    # Parse torch profiler data
    # Get CPU operations, and GPU operations
    cpu_ops, torch_comms_per_device = parse_torch_logs(torch_log_files)
    # Parse NCCL logs
    # Get GPU operations, and identify communication groups (repr. as cliques)
    nccl_cliques, nccl_comms_per_device = parse_nccl_logs(nccl_log_files)
    # Correlate GPU operations from torch profiler and NCCL logs
    # This step completes the info of GPU operation
    gpu_ops = link_nccl_torch_calls(nccl_comms_per_device, torch_comms_per_device)
    # Identify groups of collective operations using cliques and GPU operations on each device
    coll_groups = group_nccl_colls(nccl_cliques, gpu_ops)
    return cpu_ops, gpu_ops, coll_groups
