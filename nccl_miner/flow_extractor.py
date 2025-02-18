
from .parsing.nccl_parser import parse_nccl_logs
from .parsing.torch_parser import parse_torch_logs, link_nccl_torch_calls
from .mining.miner import group_nccl_colls

def extract_flows_from_logs(nccl_log_files, torch_log_files=None):
    nccl_cliques, nccl_comms_per_device = parse_nccl_logs(nccl_log_files)
    if torch_log_files is not None:
        torch_comms_per_device = parse_torch_logs(torch_log_files)
        comms_per_device = link_nccl_torch_calls(nccl_comms_per_device, torch_comms_per_device)
    else:
        comms_per_device = nccl_comms_per_device
    all_comms = group_nccl_colls(nccl_cliques, comms_per_device)
    return all_comms
