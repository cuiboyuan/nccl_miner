'''
PyTorch operations to be parsed in Torch Profiler data.
'''
import re

class CudaComm():
    def __init__(self):
        self.device = None
        self.data_type = None
        self.func = None

        self.start_time = None
        self.duration = None
        self.end_time = None


class CudaCollective(CudaComm):
    def __init__(self):
        self.op = None
        self.algo = None
        self.protocol = None

    @staticmethod
    def parse(event):
        reduce_pattern = r"ncclDevKernel_(\w+)_(\w+)_(\w+)_(\w+)_(\w+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        reduce_match = re.match(reduce_pattern, event['name'])
        broadcast_pattern = r"ncclDevKernel_(\w+)_(\w+)_(\w+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        broadcast_match = re.match(broadcast_pattern, event['name'])
        if reduce_match:
            func_name, op, dtype, algo, protocol = reduce_match.groups()
            comm = CudaCollective()
            comm.device = event['pid']
            comm.func = func_name
            comm.op = op
            # TODO: change this to NcclDataType
            comm.data_type = dtype
            comm.algo = algo
            comm.protocol = protocol
            comm.start_time = event['ts']
            comm.duration = event['dur']
            comm.end_time = comm.start_time + comm.duration
            return comm
        elif broadcast_match:
            func_name, algo, protocol = broadcast_match.groups()
            comm = CudaCollective()
            comm.device = event['pid']
            comm.func = func_name
            comm.algo = algo
            comm.protocol = protocol
            comm.start_time = event['ts']
            comm.duration = event['dur']
            comm.end_time = comm.start_time + comm.duration
            return comm
        return None


class CudaPtp(CudaComm):
    def __init__(self):
        super().__init__()

    @staticmethod
    def parse(event):
        ptp_pattern = r"ncclDevKernel_SendRecv\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        ptp_match = re.match(ptp_pattern, event['name'])
        if ptp_match:
            comm = CudaPtp()
            comm.device = event['pid']
            comm.func = "SendRecv"
            comm.start_time = event['ts']
            comm.duration = event['dur']
            comm.end_time = comm.start_time + comm.duration
            return comm
        return None


class CudaLocal():
    def __init__(self):
        pass


class CpuLocal():
    def __init__(self):
        pass