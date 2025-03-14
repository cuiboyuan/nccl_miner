'''
PyTorch operations to be parsed in Torch Profiler data.
'''
import re

class CudaComm():
    def __init__(self, func=None, device=None):
        self.device = device
        self.data_type = None
        self.func = func

        self.start_time = None
        self.duration = None
        self.end_time = None

        self.kernel_id = None

    def associate_kernel_id(self, kernel_id):
        self.kernel_id = kernel_id
    
    def __repr__(self):
        return f"{self.func} at {self.start_time} for {self.duration} ms"


class CudaCollective(CudaComm):
    def __init__(self, func=None, device=None):
        super().__init__(func, device)
        self.op = None
        self.algo = None
        self.protocol = None

    def associate_kernel(self, event):
        reduce_pattern = r"ncclDevKernel_(\w+)_(\w+)_(\w+)_(\w+)_(\w+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        reduce_match = re.match(reduce_pattern, event['name'])
        broadcast_pattern = r"ncclDevKernel_(\w+)_(\w+)_(\w+)\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        broadcast_match = re.match(broadcast_pattern, event['name'])
        if reduce_match:
            func_name, op, dtype, algo, protocol = reduce_match.groups()
            self.device = event['pid']
            self.func = func_name
            self.op = op
            # TODO: change this to NcclDataType
            self.data_type = dtype
            self.algo = algo
            self.protocol = protocol
            self.start_time = event['ts']
            self.duration = event['dur']
            self.end_time = self.start_time + self.duration
        elif broadcast_match:
            func_name, algo, protocol = broadcast_match.groups()
            self.device = event['pid']
            self.func = func_name
            self.algo = algo
            self.protocol = protocol
            self.start_time = event['ts']
            self.duration = event['dur']
            self.end_time = self.start_time + self.duration


class CudaPtp(CudaComm):
    def __init__(self, func=None, device=None):
        super().__init__(func, device)

    def associate_kernel(self, event):
        ptp_pattern = r"ncclDevKernel_SendRecv\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        ptp_match = re.match(ptp_pattern, event['name'])
        if ptp_match:
            self.device = event['pid']
            self.func = "SendRecv"
            self.start_time = event['ts']
            self.duration = event['dur']
            self.end_time = self.start_time + self.duration


class CudaLocal():
    def __init__(self):
        pass


class CpuLocal():
    def __init__(self):
        pass