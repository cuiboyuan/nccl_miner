'''
Orchestrate the steps from parsing from raw log files to extracting data flow information
'''
from .parsing.parser_pipeline import parse_torch_nccl_pipeline
from .mining.core import probe_coll_op

def mine_torch_nccl_pipeline(nccl_log_files, torch_log_files):
    # Get CPU and GPU operations on each device, will just pass them along
    # Get collective groups, which will be used to extract data flows
    cpu_ops, gpu_ops, coll_groups = parse_torch_nccl_pipeline(nccl_log_files, torch_log_files)
    # Extract data flow
    data_flow_groups = {}
    for group_id, coll_op in coll_groups.items():
        flows, deps = probe_coll_op(coll_op)
        coll_op.associate_data_flows(flows, deps)
    return cpu_ops, gpu_ops, coll_groups
