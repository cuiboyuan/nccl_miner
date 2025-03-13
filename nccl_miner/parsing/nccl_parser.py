'''
Parse relevant NCCL API function calls on each device from the NCCL log files.
'''
import os
from tqdm import tqdm
from typing import *

from ..common.nccl_function import *
from ..common.nccl_function_group import NcclCommClique

def get_host_pid(filename):
    filename_pattern = r"(?P<custom_name>.+)\.(?P<host>[a-z0-9]+)\.(?P<pid>\d+)$"
    match = re.match(filename_pattern, filename)
    if match:
        return match['host'], int(match['pid'])
    else:
        return None, None

def parse_nccl_logs(log_files):
    
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

    # TODO: check if this is always true
    world_size = len(log_files)

    for log_file in log_files:
        file_name = os.path.basename(log_file)
        print(f"Parsing log file {file_name}..")
        host, pid = get_host_pid(file_name)
        with open(log_file, "r") as f:
            state = NO_INIT
            cur_nccl_comm_init = None
            for log_line in tqdm(f.readlines()):
                if state == NO_INIT:
                    # Find Comm init call to group devices
                    init_call = NcclCommInitRankFunction.parse(log_line)
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
                    init_call = NcclCommInitRankFunction.parse(log_line)
                    if init_call is not None:
                        cur_nccl_comm_init = init_call
                        state = INIT_IN_PROGRESS
                        continue

                    # TODO: For ncclCommSplit calls, we just treat the Comm obj the same as
                    # their parents for now, which may not be correct.
                    # TODO: Update this to be correct later
                    split_call = NcclCommSplitFunction.parse(log_line)
                    if split_call is not None:
                        cur_nccl_comm_init = split_call
                        state = INIT_IN_PROGRESS
                        continue

                    # Parse NCCL communication calls
                    nccl_comm_call = None
                    # Extract Point-to-point communications
                    nccl_ptp_call = NcclPtpFunction.parse(log_line)
                    if nccl_ptp_call is not None:
                        nccl_comm_call = nccl_ptp_call
                    # Extract Collective communications
                    nccl_coll_call = NcclCollectiveFunction.parse(log_line)
                    if nccl_coll_call is not None:
                        nccl_comm_call = nccl_coll_call
                    # TODO: check why will we have nranks=1
                    # this type of NCCL call will not show up in torch profiler
                    if nccl_comm_call is not None and \
                        nccl_comm_call.nranks == 1:
                        nccl_comm_call = None
                    # Add that communication operation to its rank
                    if nccl_comm_call is not None:
                        call_comm_obj = nccl_comm_call.comm_obj
                        call_clique_id = comm_obj_to_clique_id[call_comm_obj]
                        nccl_comm_call.associate_clique_id(call_clique_id)

                        op_device = nccl_comm_call.device
                        if op_device not in comm_calls_per_device:
                            comm_calls_per_device[op_device] = {
                                'host': host,
                                'pid': pid,
                                'operations': [nccl_comm_call]
                            }
                        else:
                            comm_calls_per_device[op_device]['operations'].append(nccl_comm_call)

    print("Constructing Communication Cliques...")
    nccl_cliques = {}
    for clique_id, comm_init_calls in comm_id_to_comm_init_calls.items():
        nccl_cliques[clique_id] = NcclCommClique(comm_init_calls)

    return nccl_cliques, comm_calls_per_device
