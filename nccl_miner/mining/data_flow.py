
from nccl_miner.common.data_type import *

class NcclDataFlow:
    # counter used to generate flow id.
    global_flow_counter = 0

    def __init__(self, src, dst, size, name=None, dur=10):
        self.src = src
        self.dst = dst
        self.size = size
        self.data_name = name
        self.duration = dur
        self.id = NcclDataFlow.global_flow_counter
        NcclDataFlow.global_flow_counter += 1

    def __repr__(self):
        return f"Flow {self.id}:{self.src}->{self.dst}[{self.size} bytes]"


class NcclCommunicationOperation:
    def __init__(self, op_name, data_type, data_num, data_size, devs, flows, deps):
        self.name = op_name
        self.data_type = data_type
        self.data_num = data_num
        self.data_size = data_size
        self.data_flows = flows
        self.dependencies = deps
        self.devices = devs
        
    def __repr__(self):
        return f"{self.name}: {self.data_size} bytes ({self.data_num} {self.data_type})"
