'''
Parse relevant NCCL function calls on each devices from the NCCL log files.
'''
from tqdm import tqdm
from typing import *

from .parsing.nccl_call_type import *
# from .parsing.nccl_topology import *


def parse_nccl_calls_from_logs(log_files):
    
    '''Step 1.
    I. Identify each Communication Clique based on ncclCommInitRank & ncclCommSplit.
        i. Extract the Topology of GPUs for each Clique.
        ii. Identify devices associated with each Clique.
    II. Identify NCCL Collective/Point-to-point function calls happening on each device.
    '''
    NO_INIT = 0
    INIT_IN_PROGRESS = 1
    NORMAL = 2

    comm_id_to_comm_init_calls = {}
    comm_calls_per_device = {}
    comm_obj_to_clique_id = {}

    for log in log_files:
        print(f"Parsing log file {log}..")
        with open(log, "r") as f:
            state = NO_INIT
            cur_nccl_comm_init = None
            for log_line in tqdm(f.readlines()):
                if state == NO_INIT:
                    # Find Comm init call to group devices
                    init_call = NcclCommInitRank.parse(log_line)
                    if init_call is not None:
                        cur_nccl_comm_init = init_call
                        state = INIT_IN_PROGRESS
                        continue

                # Extract the topology of each group of devices (denoted by a comm object)
                elif state == INIT_IN_PROGRESS:
                    assert cur_nccl_comm_init is not None
                    # Extract Ring topology info.
                    cur_nccl_comm_init.parse_ring_topo(log_line)

                    # Extract Tree topo info
                    # TODO: ...

                    # Find Comm init end call
                    if cur_nccl_comm_init.parse_end(log_line):
                        comm_id = cur_nccl_comm_init.comm_id
                        comm_obj = cur_nccl_comm_init.comm_obj
                        comm_obj_to_clique_id[comm_obj] = comm_id

                        if comm_id not in comm_id_to_comm_init_calls:
                            comm_id_to_comm_init_calls[comm_id] = [cur_nccl_comm_init]
                        else:
                            comm_id_to_comm_init_calls[comm_id].append(cur_nccl_comm_init)
                        cur_nccl_comm_init = None
                        state = NORMAL
                        continue

                elif state == NORMAL:
                    # Continue looking for Comm init call to group devices
                    init_call = NcclCommInitRank.parse(log_line)
                    if init_call is not None:
                        cur_nccl_comm_init = init_call
                        state = INIT_IN_PROGRESS
                        continue

                    # TODO: For ncclCommSplit calls, we just treat the Comm obj the same as
                    # their parents for now, which may not be correct.
                    # TODO: Update this to be correct later
                    split_call = NcclCommSplit.parse(log_line)
                    if split_call is not None:
                        cur_nccl_comm_init = split_call
                        state = INIT_IN_PROGRESS
                        continue

                    # Parse NCCL communication calls
                    nccl_comm_call = None
                    # Extract Point-to-point communications
                    nccl_ptp_call = NcclPtp.parse(log_line)
                    if nccl_ptp_call is not None:
                        nccl_comm_call = nccl_ptp_call
                    # Extract Collective communications
                    nccl_coll_call = NcclCollective.parse(log_line)
                    if nccl_coll_call is not None:
                        nccl_comm_call = nccl_coll_call
                    # Add that communication operation to its rank
                    if nccl_comm_call is not None:
                        call_comm_obj = nccl_comm_call.comm_obj
                        call_clique_id = comm_obj_to_clique_id[call_comm_obj]
                        nccl_comm_call.associate_clique_id(call_clique_id)

                        op_device = nccl_comm_call.device
                        if op_device not in comm_calls_per_device:
                            comm_calls_per_device[op_device] = [nccl_comm_call]
                        else:
                            comm_calls_per_device[op_device].append(nccl_comm_call)

    print("Constructing Communication Cliques...")
    nccl_cliques = {}
    for clique_id, comm_init_calls in comm_id_to_comm_init_calls.items():
        nccl_cliques[clique_id] = NcclClique(comm_init_calls)

    return nccl_cliques, comm_calls_per_device
