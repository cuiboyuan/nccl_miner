'''
Extract data flow from the logs. User can then use the extracted information for their downstream tasks.
'''
import re
from .nccl_prober import *

def parse_coll_log(log_line):
    log_pattern = (
            r"(?P<host_name>[^:]+):"                     # Host name
            r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
            r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
            r"NCCL INFO\s+"
            r"(?P<coll_op>\w+):\s+"         # Collective operation
            r"opCount\s+(?P<op_cnt>\d+)\s+"             # Operation count
            r"sendbuff\s+(?P<src_buf_addr>0x[0-9a-f]+)\s+"  # Source buffer address
            r"recvbuff\s+(?P<dst_buf_addr>0x[0-9a-f]+)\s+"  # Destination buffer address
            r"count\s+(?P<num_elem>\d+)\s+"             # Number of elements
            r"datatype\s+(?P<data_type>\w+)\s+"         # Data type
            r"op\s+(?P<operator>\w+)\s+"                # Operator
            r"root\s+(?P<root_device>\d+)\s+"           # Root device
            r"comm\s+(?P<comm_obj_ptr>0x[0-9a-f]+)\s+"  # Comm object pointer
            r"\[nranks=(?P<nranks>\d+)\]\s+"       # Total ranks
            r"stream\s+(?P<stream_obj_ptr>0x[0-9a-f]+)" # Stream object pointer
        )

    # Match the pattern with the log line
    match = re.match(log_pattern, log_line)
    if match:
        # Return the extracted information as a dictionary
        raw_info = match.groupdict()
        return raw_info
    else:
        return None
    

def parse_ring_topo_log(log_line):
    log_pattern = (
            r"(?P<host_name>[^:]+):"                     # Host name
            r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
            r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
            r"NCCL INFO\s+"
            r"Ring 00 : (?P<prev>\d+) -> (?P<cur>\d+) -> (?P<next>\d+)"
        )

    # Match the pattern with the log line
    match = re.match(log_pattern, log_line)
    if match:
        # Return the extracted information as a dictionary
        raw_info = match.groupdict()
        return raw_info
    else:
        return None


def extract_flows_from_logs(log_files):
    ring_topos = []
    tree_topos = []
    coll_comms = []

    root_log = log_files[0]
    with open(root_log, "r") as f:
        for log_line in f.readlines():
            coll_info = parse_coll_log(log_line)
            ring_topo_info = parse_ring_topo_log(log_line)

            if coll_info is not None:
                coll_comms.append(NcclCollective(coll_info))
            elif ring_topo_info is not None:
                ring_topos.append(ring_topo_info)


    for log in log_files[1:]:
        with open(log, "r") as f:
            for log_line in f.readlines():
                ring_topo_info = parse_ring_topo_log(log_line)

                if ring_topo_info is not None:
                    ring_topos.append(ring_topo_info)

    nccl_ring = NcclAlgoRing(ring_topos)
    
    coll_flows = []
    for coll in coll_comms:
        flows, deps = nccl_ring.probe_coll_op(coll)
        coll_flows.append((flows,deps))
    
    return coll_comms, coll_flows
