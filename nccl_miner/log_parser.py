'''
Extract data flow from the logs. User can then use the extracted information for their downstream tasks.
'''
import re
from tqdm import tqdm
from typing import *

from .impl_prober import *

class NcclCall:
    def __init__(self, log_info):
        self.host = log_info['host_name']
        self.pid = int(log_info['pid'])
        self.tid = int(log_info['tid'])

        self.device = int(log_info['cuda_device'])
        self.nranks = int(log_info['nranks'])
        self.comm_obj = log_info['comm_obj_ptr']
    
    def associate_clique_id(self, clique_id):
        self.clique_id = clique_id

class NcclCommInitRank(NcclCall):
    def __init__(self, log_info):
        super().__init__(log_info)
        self.comm_id = log_info['comm_id']
        self.cur_rank = log_info['cur_rank']
        self.nvml_device = log_info['nvml_device']
        self.bus_id = log_info['bus_id']

        self.partial_rings = {}

    @staticmethod
    def parse(log_line):
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
        # Match the pattern with the log line
        match = re.match(log_pattern, log_line)
        if match:
            # Return the extracted information as a dictionary
            raw_info = match.groupdict()
            return NcclCommInitRank(raw_info)
        else:
            return None

    def parse_end(log_line):
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
            return True
        else:
            return False

    def parse_ring_topo(self, log_line):
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
            self.partial_rings['00'] = raw_info


class NcclCommSplit(NcclCommInitRank):
    def __init__(self, log_info):
        super().__init__(log_info)
        self.parent_comm_obj = log_info['parent_obj_ptr']
        self.color = log_info['color']
        self.key = log_info['key']

    @staticmethod
    def parse(log_line):
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
                r"key\s+(?P<key>[0-9a-z]+)\s+"  # key
                r"commId\s+(?P<comm_id>0x[0-9a-f]+)\s+"  # Comm ID
                r"-\s+Init START"
            )
        # Match the pattern with the log line
        match = re.match(log_pattern, log_line)
        if match:
            # Return the extracted information as a dictionary
            raw_info = match.groupdict()
            return NcclCommSplit(raw_info)
        else:
            return None

    def parse_end(log_line):
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
                r"key\s+(?P<key>[0-9a-z]+)\s+"  # key
                r"commId\s+(?P<comm_id>0x[0-9a-f]+)\s+"  # Comm ID
                r"-\s+Init COMPLETE"
            )
        # Match the pattern with the log line
        match = re.match(log_pattern, log_line)
        if match:
            return True
        else:
            return False


class NcclPtp(NcclCall):
    def __init__(self, log_info):
        super().__init__(log_info)
        self.func = log_info['ptp_op']

        self.data_num = int(log_info['num_elem'])
        self.data_type = NcclDataType(log_info['data_type'])
        self.data_size = self.data_type.bytes * self.data_num

        self.peer_rank = int(log_info['peer_rank'])

    @staticmethod
    def parse(log_line):
        log_pattern = (
                r"(?P<host_name>[^:]+):"                     # Host name
                r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
                r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
                r"NCCL INFO\s+"
                r"(?P<ptp_op>\w+):\s+"         # Collective operation
                r"opCount\s+(?P<op_cnt>[0-9a-z]+)\s+"             # Operation count
                r"sendbuff\s+\(nil\)\s+"  # Source buffer address (N/A)
                r"recvbuff\s+(?P<buf_addr>0x[0-9a-f]+)\s+"  # Send/Recv buffer address
                r"count\s+(?P<num_elem>\d+)\s+"             # Number of elements
                r"datatype\s+(?P<data_type>\w+)\s+"         # Data type
                r"op\s+(?P<operator>\w+)\s+"                # Operator
                r"root\s+(?P<peer_rank>\d+)\s+"           # peer device
                r"comm\s+(?P<comm_obj_ptr>0x[0-9a-f]+)\s+"  # Comm object pointer
                r"\[nranks=(?P<nranks>\d+)\]\s+"       # Total ranks
                r"stream\s+(?P<stream_obj_ptr>0x[0-9a-f]+)" # Stream object pointer
            )

        # Match the pattern with the log line
        match = re.match(log_pattern, log_line)
        if match:
            # Return the extracted information as a dictionary
            raw_info = match.groupdict()
            return NcclPtp(raw_info)
        else:
            return None


class NcclCollective(NcclCall):
    def __init__(self, log_info):
        super().__init__(log_info)
        self.func = log_info['coll_op']

        self.data_num = int(log_info['num_elem'])
        self.data_type = NcclDataType(log_info['data_type'])
        self.data_size = self.data_type.bytes * self.data_num

        self.root_rank = int(log_info['root_rank'])

    @staticmethod
    def parse(log_line):
        log_pattern = (
                r"(?P<host_name>[^:]+):"                     # Host name
                r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
                r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
                r"NCCL INFO\s+"
                r"(?P<coll_op>\w+):\s+"         # Collective operation
                r"opCount\s+(?P<op_cnt>[0-9a-z]+)\s+"             # Operation count
                r"sendbuff\s+(?P<src_buf_addr>0x[0-9a-f]+)\s+"  # Source buffer address
                r"recvbuff\s+(?P<dst_buf_addr>0x[0-9a-f]+)\s+"  # Destination buffer address
                r"count\s+(?P<num_elem>\d+)\s+"             # Number of elements
                r"datatype\s+(?P<data_type>\w+)\s+"         # Data type
                r"op\s+(?P<operator>\w+)\s+"                # Operator
                r"root\s+(?P<root_rank>\d+)\s+"           # Root device
                r"comm\s+(?P<comm_obj_ptr>0x[0-9a-f]+)\s+"  # Comm object pointer
                r"\[nranks=(?P<nranks>\d+)\]\s+"       # Total ranks
                r"stream\s+(?P<stream_obj_ptr>0x[0-9a-f]+)" # Stream object pointer
            )

        # Match the pattern with the log line
        match = re.match(log_pattern, log_line)
        if match:
            # Return the extracted information as a dictionary
            raw_info = match.groupdict()
            return NcclCollective(raw_info)
        else:
            return None

class NcclClique:
    def __init__(self, nccl_init_calls: List[NcclCommInitRank]):
        self.id = None
        self.rank_to_device = {}

        partial_rings = []
        for comm_init in nccl_init_calls:
            if self.id is None:
                self.id = comm_init.comm_id
            assert self.id == comm_init.comm_id
            # Map the rank in this clique to actual CUDA device.
            self.rank_to_device[comm_init.cur_rank] = comm_init.device
            # Construct the Ring.
            partial_rings.append(nccl_init_calls.partial_rings)
            # TODO: Construct the Tree.
            # ...
        self.ring_algo = NcclAlgoRing(partial_rings, self.rank_to_device)


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
                    init_call = NcclCommInitRank.parse(log_line, is_start=True)
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
    for clique_id, comm_init_calls in comm_id_to_comm_init_calls:
        nccl_cliques[clique_id] = NcclClique(comm_init_calls)

    return nccl_cliques, comm_calls_per_device


def extract_flows_from_logs(log_files):

    comm_cliques, comms_per_device = parse_nccl_calls_from_logs(log_files)
    
    '''Step 2.
    Based on Clique and Func Calls Per Device info, group Func Calls into Operations.
    A Collective Operation is at the Clique-level, involving all devices within that Clique.
    A PTP (SendRecv) Operation only involves src & dst devices.
    All operations will be ordered based on earliest finish time (EFT). TODO: Think about whether this is an issue.
    Operations will be probed to extract Data Flow information
    '''
    cur_idx_per_device = {}
    max_idx_per_device = {}
    for cur_dev in comms_per_device:
        cur_idx_per_device[cur_dev] = 0
        max_idx_per_device[cur_dev] = len(comms_per_device[cur_dev])

    def are_all_logs_parsed():
        for dev, idx in cur_idx_per_device.items():
            if idx < max_idx_per_device[dev]:
                return False
        return True
    
    # For progress tracking only
    total_iter = sum([max_idx for dev, max_idx in max_idx_per_device.items()])
    cur_iter = 0

    # TODO: Assume every operation blocks.

    all_comms = []
    while are_all_logs_parsed() is False:
        comm_id_to_pending_coll = {}
        comm_id_to_pending_send = {}
        comm_id_to_pending_recv = {}
        for cur_dev, nccl_calls in comms_per_device.items():
            do_advance_all_idx = False

            cur_idx = cur_idx_per_device[cur_dev]
            max_idx = max_idx_per_device[cur_dev]
            if cur_idx >= max_idx:
                continue

            cur_nccl_call = nccl_calls[cur_idx]
            cur_call_comm_id = comm_obj_to_comm_id[cur_nccl_call['comm_obj_ptr']]
            # print(cur_nccl_call)
            if 'coll_op' in cur_nccl_call:
                if cur_call_comm_id not in comm_id_to_pending_coll:
                    comm_id_to_pending_coll[cur_call_comm_id] = [cur_nccl_call]
                else:
                    pending_coll = comm_id_to_pending_coll[cur_call_comm_id]
                    assert cur_nccl_call['coll_op'] == pending_coll[-1]['coll_op']
                    pending_coll.append(cur_nccl_call)
                # Check if all devices reach this collectice call.
                this_comm_obj_nranks = len(comm_id_to_rank_dev_mappings[cur_call_comm_id])
                if len(comm_id_to_pending_coll[cur_call_comm_id]) == this_comm_obj_nranks:
                    all_comms.append(NcclCollectiveCall(cur_nccl_call))

                    # Advance the indices
                    for coll in comm_id_to_pending_coll[cur_call_comm_id]:
                        pending_dev = coll['cuda_device']
                        cur_idx_per_device[pending_dev] += 1
                        cur_iter += 1
                        print(f"{cur_iter/total_iter*100}%", end="\r")

            elif 'ptp_op' in cur_nccl_call:
                if cur_nccl_call['ptp_op'] == "Send":
                    src = cur_nccl_call['cuda_device']
                    if cur_call_comm_id in comm_id_to_pending_recv \
                         and src in comm_id_to_pending_recv[cur_call_comm_id]:
                        # Complete the Send-Receive Data flow
                        send_recv_call = comm_id_to_pending_recv[cur_call_comm_id][src]
                        nccl_ptp_op = NcclSendRecvCall(send_recv_call)
                        print(send_recv_call)
                        all_comms.append(nccl_ptp_op)
                        # Advance the index
                        cur_idx_per_device[send_recv_call['cuda_device']] += 1
                        cur_idx_per_device[cur_dev] += 1
                        # For progress tracking only
                        cur_iter += 2
                        print(f"{cur_iter/total_iter*100}%", end="\r")
                    else:
                        # Need to wait for Receive call
                        dst = cur_nccl_call['peer_device']
                        if cur_call_comm_id not in comm_id_to_pending_send:
                            comm_id_to_pending_send[cur_call_comm_id] = {dst: cur_nccl_call}
                        else:
                            comm_id_to_pending_send[cur_call_comm_id].update({dst: cur_nccl_call})

                elif cur_nccl_call['ptp_op'] == "Recv":
                    # print(cur_nccl_call)
                    dst = cur_nccl_call['cuda_device']
                    if cur_call_comm_id in comm_id_to_pending_send \
                         and dst in comm_id_to_pending_send[cur_call_comm_id]:
                        # Complete the Send-Receive Data flow
                        send_recv_call = comm_id_to_pending_send[cur_call_comm_id][dst]
                        nccl_ptp_op = NcclSendRecvCall(send_recv_call)
                        print(send_recv_call)
                        all_comms.append(nccl_ptp_op)
                        # Advance the index
                        cur_idx_per_device[send_recv_call['cuda_device']] += 1
                        cur_idx_per_device[cur_dev] += 1
                        # For progress tracking only
                        cur_iter += 2
                        print(f"{cur_iter/total_iter*100}%", end="\r")
                    else:
                        # Need to wait for Send call
                        src = cur_nccl_call['peer_device']
                        if cur_call_comm_id not in comm_id_to_pending_recv:
                            comm_id_to_pending_recv[cur_call_comm_id] = {src: cur_nccl_call}
                        else:
                            comm_id_to_pending_recv[cur_call_comm_id].update({src: cur_nccl_call})
        
    
    comm_flows = []
    print("Probe all communication...")
    for comm in tqdm(all_comms):
        cur_comm_id = comm_obj_to_comm_id[comm.comm_obj]
        nccl_ring = comm_obj_to_ring_algo[cur_comm_id]
        nccl_operation = nccl_ring.probe_coll_op(comm)
        comm_flows.append(nccl_operation)
    
    return comm_flows
