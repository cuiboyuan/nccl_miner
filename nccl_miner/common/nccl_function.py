'''
NCCL API functions to be parsed in the NCCL logs.
For details on NCCL APIs, see here:
https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2134/user-guide/docs/api.html
'''
import re
from typing import *

from .nccl_data_type import *
from .topology import *


class NcclFunction:
    def __init__(self, log_info):
        self.host = log_info['host_name']
        self.pid = int(log_info['pid'])
        self.tid = int(log_info['tid'])

        self.device = int(log_info['cuda_device'])
        self.nranks = int(log_info['nrank'])
        self.comm_obj = log_info['comm_obj_ptr']

    def associate_clique_id(self, clique_id):
        self.clique_id = clique_id

    def associate_group_id(self, group_id):
        self.group_id = group_id

    def associate_time(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    def associate_algo(self, algo):
        self.algo = algo

    def associate_protocol(self, proto):
        self.protocol = proto

    def associate_semantics(self, semantics):
        self.semantics = semantics


class NcclCommInitRankFunction(NcclFunction):
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
            return NcclCommInitRankFunction(raw_info)
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


class NcclCommSplitFunction(NcclCommInitRankFunction):
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
            return NcclCommSplitFunction(raw_info)
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


class NcclPtpFunction(NcclFunction):
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
            return NcclPtpFunction(raw_info)
        else:
            return None

    def __repr__(self):
        return f"[{self.device}] {self.func} {self.comm_obj}"


class NcclCollectiveFunction(NcclFunction):
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
            return NcclCollectiveFunction(raw_info)
        else:
            return None

    def __repr__(self):
        return f"[{self.device}] {self.func} {self.comm_obj}"
