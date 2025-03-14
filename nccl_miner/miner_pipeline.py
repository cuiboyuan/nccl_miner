'''
Orchestrate the steps from parsing from raw log files to extracting data flow information
'''
from .parsing.parser_pipeline import parse_torch_nccl_pipeline
from .mining.dependency_deduction import deduce_flow_dependencies
from .mining.timestamp_deduction import deduce_flow_timestamps

def mine_torch_nccl_pipeline(nccl_log_files, torch_log_files):
    # Get CPU and GPU operations on each device, will just pass them along
    # Get collective groups, which will be used to extract data flows
    cpu_ops, gpu_ops, coll_groups = parse_torch_nccl_pipeline(nccl_log_files, torch_log_files)
    # Extract data flow
    data_flow_groups = {}
    for group_id, coll_op in coll_groups.items():
        flows, deps = deduce_flow_dependencies(coll_op)
        coll_op.associate_data_flows(flows, deps)
        deduce_flow_timestamps(coll_op)
    return cpu_ops, gpu_ops, coll_groups
