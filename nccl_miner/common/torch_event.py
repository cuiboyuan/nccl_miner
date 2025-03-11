'''
PyTorch operations to be parsed in Torch Profiler data.
'''
import re

class CudaComm():
    def __init__(self):
        self.device = None
        self.func = None
        self.op = None
        self.data_type = None
        self.algo = None
        self.protocol = None

        self.start_time = None
        self.end_time = None
        self.duration = None

    @staticmethod
    def parse(event):
        coll_pattern = r"ncclDevKernel_(\w+)_(\w+)_(\w+)_(\w+)_(\w+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        coll_match = re.match(coll_pattern, event['name'])
        if coll_match:
            func_name, op, dtype, algo, protocol = coll_match.groups()
            comm = CudaComm()
            comm.device = event['pid']
            comm.func = func_name
            comm.op = op
            comm.data_type = dtype
            comm.algo = algo
            comm.protocol = protocol
            comm.start_time = event['ts']
            comm.end_time = event['ts'] + event['dur']
            comm.duration = event['dur']
            return comm

        ptp_pattern = r"ncclDevKernel_SendRecv(ncclDevComm*, unsigned long, ncclWork*)"
        ptp_match = re.match(ptp_pattern, event['name'])
        if ptp_match:
            comm = CudaComm()
            comm.device = event['pid']
            comm.func = "SendRecv"
            comm.start_time = event['ts']
            comm.end_time = event['ts'] + event['dur']
            comm.duration = event['dur']
            return comm

        return None

class CudaLocal():
    def __init__(self):
        pass


class CpuLocal():
    def __init__(self):
        pass