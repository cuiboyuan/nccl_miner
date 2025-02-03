'''
Extract data flow from the logs. User can then use the extracted information for their downstream tasks.
'''
import re
from tqdm import tqdm
from .impl_prober import *


def parse_comm_init_log(log_line, is_start):
    if is_start:
        log_pattern = (
                r"(?P<host_name>[^:]+):"                     # Host name
                r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
                r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
                r"NCCL INFO\s+"
                r"ncclCommInitRank\s+"         # NCCL function to initialize Comm Object
                r"comm\s+(?P<comm_obj_ptr>0x[0-9a-f]+)\s+"  # Comm object pointer
                r"rank\s+(?P<cur_rank>[0-9]+)\s+"  # Current Rank
                r"nranks\s+(?P<nrank>[0-9]+)\s+"  # Total Ranks
                r"cudaDev\s+(?P<cur_device>[0-9]+)\s+"  # CUDA Device
                r"nvmlDev\s+(?P<nvml_device>[0-9]+)\s+"  # NVML Device
                r"busId\s+(?P<bus_id>[0-9a-z]+)\s+"  # Bus ID
                r"commId\s+(?P<comm_id>0x[0-9a-f]+)\s+"  # Comm ID
                r"-\s+Init START"
            )
    else:
        log_pattern = (
                r"(?P<host_name>[^:]+):"                     # Host name
                r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
                r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
                r"NCCL INFO\s+"
                r"ncclCommInitRank\s+"         # NCCL function to initialize Comm Object
                r"comm\s+(?P<comm_obj_ptr>0x[0-9a-f]+)\s+"  # Comm object pointer
                r"rank\s+(?P<cur_rank>[0-9]+)\s+"  # Current Rank
                r"nranks\s+(?P<nrank>[0-9]+)\s+"  # Total Ranks
                r"cudaDev\s+(?P<cur_device>[0-9]+)\s+"  # CUDA Device
                r"nvmlDev\s+(?P<nvml_device>[0-9]+)\s+"  # NVML Device
                r"busId\s+(?P<bus_id>[0-9a-z]+)\s+"  # Bus ID
                r"commId\s+(?P<comm_id>0x[0-9a-f]+)\s+"  # Comm ID
                r"-\s+Init COMPLETE"
            )


    # Match the pattern with the log line
    match = re.match(log_pattern, log_line)
    if match:
        # Return the extracted information as a dictionary
        raw_info = match.groupdict()
        return raw_info
    else:
        return None


def parse_comm_split_log(log_line, is_start):
    if is_start:
        log_pattern = (
                r"(?P<host_name>[^:]+):"                     # Host name
                r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
                r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
                r"NCCL INFO\s+"
                r"ncclCommSplit\s+"         # NCCL function to initialize Comm Object
                r"comm\s+(?P<comm_obj_ptr>0x[0-9a-f]+)\s+"  # Comm object pointer
                r"rank\s+(?P<cur_rank>[0-9]+)\s+"  # Current Rank
                r"nranks\s+(?P<nrank>[0-9]+)\s+"  # Total Ranks
                r"cudaDev\s+(?P<cur_device>[0-9]+)\s+"  # CUDA Device
                r"nvmlDev\s+(?P<nvml_device>[0-9]+)\s+"  # NVML Device
                r"busId\s+(?P<bus_id>[0-9a-z]+)\s+"  # Bus ID
                r"parent\s+(?P<parent_obj_ptr>0x[0-9a-f]+)\s+"  # Parent Comm Obj pointer
                r"color\s+(?P<color>[0-9]+)\s+"  # color
                r"key\s+(?P<key>[0-9]+)\s+"  # key
                r"commId\s+(?P<comm_id>0x[0-9a-f]+)\s+"  # Comm ID
                r"-\s+Init START"
            )
    else:
        log_pattern = (
                r"(?P<host_name>[^:]+):"                     # Host name
                r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
                r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
                r"NCCL INFO\s+"
                r"ncclCommSplit\s+"         # NCCL function to initialize Comm Object
                r"comm\s+(?P<comm_obj_ptr>0x[0-9a-f]+)\s+"  # Comm object pointer
                r"rank\s+(?P<cur_rank>[0-9]+)\s+"  # Current Rank
                r"nranks\s+(?P<nrank>[0-9]+)\s+"  # Total Ranks
                r"cudaDev\s+(?P<cur_device>[0-9]+)\s+"  # CUDA Device
                r"nvmlDev\s+(?P<nvml_device>[0-9]+)\s+"  # NVML Device
                r"busId\s+(?P<bus_id>[0-9a-z]+)\s+"  # Bus ID
                r"parent\s+(?P<parent_obj_ptr>0x[0-9a-f]+)\s+"  # Parent Comm Obj pointer
                r"color\s+(?P<color>[0-9]+)\s+"  # color
                r"key\s+(?P<key>[0-9]+)\s+"  # key
                r"commId\s+(?P<comm_id>0x[0-9a-f]+)\s+"  # Comm ID
                r"-\s+Init COMPLETE"
            )


    # Match the pattern with the log line
    match = re.match(log_pattern, log_line)
    if match:
        # Return the extracted information as a dictionary
        raw_info = match.groupdict()
        return raw_info
    else:
        return None


def parse_ptp_log(log_line):
    log_pattern = (
            r"(?P<host_name>[^:]+):"                     # Host name
            r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
            r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
            r"NCCL INFO\s+"
            r"(?P<ptp_op>\w+):\s+"         # Collective operation
            r"opCount\s+(?P<op_cnt>\d+)\s+"             # Operation count
            r"sendbuff\s+\(nil\)\s+"  # Source buffer address (N/A)
            r"recvbuff\s+(?P<buf_addr>0x[0-9a-f]+)\s+"  # Send/Recv buffer address
            r"count\s+(?P<num_elem>\d+)\s+"             # Number of elements
            r"datatype\s+(?P<data_type>\w+)\s+"         # Data type
            r"op\s+(?P<operator>\w+)\s+"                # Operator
            r"root\s+(?P<peer_device>\d+)\s+"           # peer device
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

    '''Step 1.
    Extract NCCL Collective/Point-to-point function calls happening on each rank.
    Extract Topology of GPUs from the log files.
    '''
    comm_id_to_rank_dev_mappings = {}
    comm_id_to_partial_ring_topo = {}
    comm_obj_to_comm_id = {}

    comms_per_device = {}
    for log in log_files:
        print(f"Parsing log file {log}..")
        with open(log, "r") as f:
            cur_comm_id = None
            init_in_progress = False
            for log_line in tqdm(f.readlines()):
                # Find Comm init call to group devices
                nccl_comm_init_call = parse_comm_init_log(log_line, is_start=True)
                if nccl_comm_init_call is not None:
                    comm_id = nccl_comm_init_call['comm_id']
                    cur_device = nccl_comm_init_call['cuda_device']
                    cur_rank = nccl_comm_init_call['cur_rank']
                    if comm_id not in comm_id_to_rank_dev_mappings:
                        comm_id_to_rank_dev_mappings[comm_id] = {int(cur_rank): int(cur_device)}
                    else:
                        comm_id_to_rank_dev_mappings[comm_id].update({int(cur_rank): int(cur_device)})

                    cur_comm_id = comm_id
                    if cur_comm_id not in comm_id_to_partial_ring_topo:
                        comm_id_to_partial_ring_topo[cur_comm_id] = []

                    cur_comm_obj = nccl_comm_init_call['comm_obj_ptr']
                    if cur_comm_obj not in comm_obj_to_comm_id:
                        comm_obj_to_comm_id[cur_comm_obj] = cur_comm_id
                    init_in_progress = True

                # Extract the topology of each group of devices (denoted by a comm object)
                if init_in_progress:
                    # Extract Ring topology info.
                    partial_ring_topo = parse_ring_topo_log(log_line)
                    if partial_ring_topo is not None:
                        comm_id_to_partial_ring_topo[cur_comm_id].append(partial_ring_topo)
                    # Extract Tree topo info
                    # TODO: ...
                    # Find Comm init end call
                    nccl_comm_init_call_end = parse_comm_init_log(log_line, is_start=False)
                    if nccl_comm_init_call_end is not None:
                        init_in_progress = False
                        cur_comm_id = None
                else:
                    # TODO: For ncclCommSplit calls, we just treat the Comm obj the same as
                    # their parents for now, which may not be correct.
                    # TODO: Update this to be correct later
                    nccl_comm_split_call = parse_comm_split_log(log_line, is_start=True)
                    if nccl_comm_split_call is not None:
                        comm_obj = nccl_comm_split_call['comm_obj_ptr']
                        parent_obj = nccl_comm_split_call['parent_obj_ptr']
                        parent_comm_id = comm_obj_to_comm_id[parent_obj]
                        comm_obj_to_comm_id[comm_obj] = parent_comm_id

                    # Parse NCCL communication calls
                    nccl_comm_call = None
                    # Extract Point-to-point communications
                    nccl_ptp_call = parse_ptp_log(log_line)
                    if nccl_ptp_call is not None:
                        nccl_comm_call = nccl_ptp_call
                    # Extract Collective communications
                    nccl_coll_call = parse_coll_log(log_line)
                    if nccl_coll_call is not None:
                        nccl_comm_call = nccl_coll_call
                    # Add that communication operation to its rank
                    if nccl_comm_call is not None:
                        op_device = nccl_comm_call['cuda_device']
                        if op_device not in comms_per_device:
                            comms_per_device[op_device] = [nccl_comm_call]
                        else:
                            comms_per_device[op_device].append(nccl_comm_call)

    print("Constructing Ring...")
    comm_obj_to_ring_algo = {}
    for comm_obj, partial_ring in comm_id_to_partial_ring_topo.items():
        nccl_ring = NcclAlgoRing(partial_ring, comm_id_to_rank_dev_mappings[comm_obj])
        comm_obj_to_ring_algo[comm_obj] = nccl_ring
    del comm_id_to_partial_ring_topo
    
    '''Step 2.
    Group NCCL function calls on each rank into Collective operations.
    Extract the Data Flow information behind all the Collective operations.

    '''
    cur_idx_per_device = {}
    max_idx_per_device = {}
    for dev in comms_per_device:
        cur_idx_per_device[dev] = 0
        max_idx_per_device[dev] = len(comms_per_device[dev])

    def are_all_logs_parsed():
        for dev, idx in cur_idx_per_device.items():
            if idx < max_idx_per_device[dev]:
                return False
        return True
    
    # For progress tracking only
    total_iter = sum([max_idx for dev, max_idx in max_idx_per_device.items()])
    cur_iter = 0

    all_comms = []
    while are_all_logs_parsed() is False:
        comm_id_to_pending_coll = {}
        for dev, nccl_calls in comms_per_device.items():
            do_advance_all_idx = False

            cur_idx = cur_idx_per_device[dev]
            max_idx = max_idx_per_device[dev]
            if cur_idx >= max_idx:
                continue

            cur_nccl_call = nccl_calls[cur_idx]
            # print(cur_nccl_call)
            if 'coll_op' in cur_nccl_call:
                nccl_call_comm_obj = cur_nccl_call['comm_obj_ptr']
                nccl_call_comm_id = comm_obj_to_comm_id[nccl_call_comm_obj]
                if nccl_call_comm_id not in comm_id_to_pending_coll:
                    comm_id_to_pending_coll[nccl_call_comm_id] = [cur_nccl_call]
                else:
                    pending_coll = comm_id_to_pending_coll[nccl_call_comm_id]
                    assert cur_nccl_call['coll_op'] == pending_coll[-1]['coll_op']
                    pending_coll.append(cur_nccl_call)
                # Check if all devices reach this collectice call.
                this_comm_obj_nranks = len(comm_id_to_rank_dev_mappings[nccl_call_comm_id])
                if len(comm_id_to_pending_coll[nccl_call_comm_id]) == this_comm_obj_nranks:
                    all_comms.append(NcclCollectiveCall(cur_nccl_call))
                    do_advance_all_idx = True

                if do_advance_all_idx:
                    for dev in cur_idx_per_device:
                        cur_idx_per_device[dev] += 1
                        cur_iter += 1
                    print(f"{cur_iter/total_iter*100}%")
                    do_advance_all_idx = False

            elif 'ptp_op' in cur_nccl_call:
                nccl_ptp_op = NcclPtpCall(cur_nccl_call)
                all_comms.append(nccl_ptp_op)
                cur_idx_per_device[dev] += 1
                cur_iter += 1
                print(f"{cur_iter/total_iter*100}%")
        
    
    comm_flows = []
    print("Probe all communication...")
    for comm in tqdm(all_comms):
        cur_comm_id = comm_obj_to_comm_id[comm.comm_obj]
        nccl_ring = comm_obj_to_ring_algo[cur_comm_id]
        nccl_operation = nccl_ring.probe_coll_op(comm)
        comm_flows.append(nccl_operation)
    
    return comm_flows
