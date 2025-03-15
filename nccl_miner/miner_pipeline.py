'''
Orchestrate the all the steps to mine data flow info from raw log files

1. Parsing: parsing from different logs files and correlating information.
2. Mining: extracting and deducing data flow information
'''

from .parsing.nccl_parser import parse_nccl_logs
from .parsing.torch_parser import parse_torch_logs
from .parsing.combiner import group_nccl_calls_across_devices, link_nccl_torch_calls
from .mining.dependency_deduction import deduce_flow_dependencies
from .mining.timestamp_deduction import deduce_flow_timestamps

def mine_torch_nccl_pipeline(nccl_log_files, torch_log_files):
    # Parsing:
    # Parse torch profiler data. Get CPU operations, and GPU operations
    cpu_ops, torch_comms_per_device = parse_torch_logs(torch_log_files)
    # Parse NCCL logs. Get GPU operations, and identify communication groups (repr. as cliques)
    nccl_cliques, nccl_comms_per_device = parse_nccl_logs(nccl_log_files)
    # Correlate GPU operations from torch profiler and NCCL logs
    # This step completes the info of GPU operation
    gpu_ops = link_nccl_torch_calls(nccl_comms_per_device, torch_comms_per_device)
    # Identify groups of collective operations using cliques and GPU operations on each device
    coll_groups = group_nccl_calls_across_devices(nccl_cliques, gpu_ops)
    
    # Extracting and deducing data flow information:
    data_flow_groups = {}
    for group_id, coll_op in coll_groups.items():
        flows, deps = deduce_flow_dependencies(coll_op)
        coll_op.associate_data_flows(flows, deps)
        deduce_flow_timestamps(coll_op)
    return cpu_ops, gpu_ops, coll_groups
