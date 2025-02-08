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
        self.nranks = int(log_info['nrank'])
        self.comm_obj = log_info['comm_obj_ptr']
    
    def associate_clique_id(self, clique_id):
        self.clique_id = clique_id

class NcclCommInitRank(NcclCall):
    def __init__(self, log_info):
        super().__init__(log_info)
        self.comm_id = log_info['comm_id']
        self.cur_rank = int(log_info['cur_rank'])
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

    def parse_end(self, log_line):
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
            self.partial_rings['00'] = {
                'prev': int(raw_info['prev']),
                'cur': int(raw_info['cur']),
                'next': int(raw_info['next']),
            }


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

    def parse_end(self, log_line):
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
                r"\[nranks=(?P<nrank>\d+)\]\s+"       # Total ranks
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
        
    def __repr__(self):
        return f"[{self.device}] {self.func} {self.comm_obj}"


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
                r"\[nranks=(?P<nrank>\d+)\]\s+"       # Total ranks
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
        
    def __repr__(self):
        return f"[{self.device}] {self.func} {self.comm_obj}"

class NcclClique:
    def __init__(self, nccl_init_calls: List[NcclCommInitRank]):
        self.id = None
        self.rank_to_device = {}
        self.device_to_rank = {}

        partial_rings = []
        for comm_init in nccl_init_calls:
            if self.id is None:
                self.id = comm_init.comm_id
            assert self.id == comm_init.comm_id
            # Map the rank in this clique to actual CUDA device.
            self.rank_to_device[comm_init.cur_rank] = comm_init.device
            self.device_to_rank[comm_init.device] = comm_init.cur_rank
            # Construct the Ring.
            partial_rings.append(comm_init.partial_rings)
            # TODO: Construct the Tree.
            # ...
        print("Constructing Ring")
        self.ring_algo = NcclAlgoRing(partial_rings, self.rank_to_device)
    
    def get_device_rank(self, dev):
        return self.device_to_rank[dev]
    
    def get_rank_device(self, rank):
        return self.rank_to_device[rank]


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

    pending_ptp_calls = []
    pending_coll_calls = []
    while are_all_logs_parsed() == False:
        for dev, nccl_calls in comms_per_device.items():
            cur_idx = cur_idx_per_device[dev]
            max_idx = max_idx_per_device[dev]
            if cur_idx >= max_idx:
                continue

            cur_nccl_call = nccl_calls[cur_idx]
            if isinstance(cur_nccl_call, NcclPtp):
                cur_cliq_id = cur_nccl_call.clique_id
                clique = comm_cliques[cur_cliq_id]

                unblocked = False
                for _ in range(len(pending_ptp_calls)):
                    pending_ptp = pending_ptp_calls.pop(0)

                    pending_cliq_id = pending_ptp.clique_id
                    if cur_cliq_id == pending_cliq_id:
                        cur_func = cur_nccl_call.func
                        cur_device = cur_nccl_call.device
                        cur_peer = clique.get_rank_device(cur_nccl_call.peer_rank)

                        pending_func = pending_ptp.func
                        pending_device = pending_ptp.device
                        pending_peer = clique.get_rank_device(pending_ptp.peer_rank)
                        if (cur_func == 'Send' and pending_func == "Recv") \
                            or (cur_func == 'Recv' and pending_func == "Send"):
                            if ((cur_device == pending_peer) \
                                or (cur_peer == pending_device)):
                                # Unblocked
                                if (cur_nccl_call.data_num == pending_ptp.data_num \
                                    and cur_nccl_call.data_type == pending_ptp.data_type \
                                    and cur_nccl_call.data_size == pending_ptp.data_size):
                                    # Unblocked, add this to final comm operations
                                    if cur_func == 'Send':
                                        src_dev = cur_device
                                        dst_dev = pending_device
                                    else:
                                        src_dev = pending_device
                                        dst_dev = cur_device

                                    flow = NcclDataFlow(src_dev, dst_dev, cur_nccl_call.data_size)
                                    sendrecv_op = NcclCommunicationOperation(
                                        "SendRecv",
                                        cur_nccl_call.data_type,
                                        cur_nccl_call.data_num,
                                        cur_nccl_call.data_size,
                                        [src_dev, dst_dev],
                                        {flow.id: flow},
                                        {})
                                    all_comms.append(sendrecv_op)
                                    unblocked = True
                                    break
                    # Not unblocked, continue waiting in pending queue
                    pending_ptp_calls.append(pending_ptp)

                if not unblocked:
                    pending_ptp_calls.append(cur_nccl_call)

            elif isinstance(cur_nccl_call, NcclCollective):
                cur_cliq_id = cur_nccl_call.clique_id
                clique = comm_cliques[cur_cliq_id]

                unblocked = False
                matched = False
                for _ in range(len(pending_coll_calls)):
                    # print(">>>")
                    pending_coll, num_matched = pending_coll_calls.pop(0)
                    # print(pending_coll_calls)

                    pending_cliq_id = pending_coll.clique_id
                    cur_func = cur_nccl_call.func
                    pending_func = pending_coll.func
                    if cur_cliq_id == pending_cliq_id \
                        and cur_func == pending_func:
                        # Matched.
                        if (cur_nccl_call.data_num == pending_coll.data_num \
                            and cur_nccl_call.data_type == pending_coll.data_type \
                            and cur_nccl_call.data_size == pending_coll.data_size):
                            # Matched.
                            matched = True
                            num_matched += 1
                            if num_matched == len(clique.rank_to_device):
                                unblocked = True
                                coll_op = clique.ring_algo.probe_coll_op(cur_nccl_call)
                                all_comms.append(coll_op)
                                break
                    # not unblocked, continue waiting in pending queue
                    pending_coll_calls.append((pending_coll, num_matched))
                    # print(pending_coll_calls)
                    # print("<<<")
                    
                if not unblocked and not matched:
                    if len(clique.rank_to_device) > 1:
                        pending_coll_calls.append((cur_nccl_call, 1))
                    else:
                        coll_op = clique.ring_algo.probe_coll_op(cur_nccl_call)
                        all_comms.append(coll_op)


            cur_idx_per_device[dev] += 1

            print(cur_idx_per_device)

    assert len(pending_ptp_calls) == 0
    assert len(pending_coll_calls) == 0
    
    return all_comms
