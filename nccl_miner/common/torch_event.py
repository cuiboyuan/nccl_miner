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

        self.cpu_start_time = None
        self.cpu_end_time = None

    def associate_kernel_id(self, kernel_id):
        self.kernel_id = kernel_id

    def associate_cpu_event(self, event):
        self.cpu_start_time = event['ts']
        self.cpu_end_time = event['ts'] + event['dur']

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
            try:
                assert self.func is None or func_name == self.func
            except AssertionError:
                print(f"AssertionError: {self.func} != {func_name}")
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
            try:
                assert self.func is None or func_name == self.func
            except AssertionError:
                print(f"AssertionError: {self.func} != {func_name}")
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


class CudaCoalesced(CudaComm):
    def __init__(self, func=None, device=None):
        super().__init__(func, device)
        self.op_list = []

    def add_cuda_op(self, cuda_op):
        assert isinstance(cuda_op, CudaComm)
        self.op_list.append(cuda_op)


class CudaLocal():
    def __init__(self, name=None, device=None):
        self.name = name
        self.device = device
        self.start_time = None
        self.duration = None

    def associate_context_events(self, context_events):
        # TODO: Naive implementation for now, need to deduce the semantics of the data
        sorted_context_events = sorted(context_events, key=lambda event: event['ts'])
        self.context = [event['name'] for event in sorted_context_events]
        self.name = self.context[0]

    def associate_cpu_event(self, event):
        self.cpu_start_time = event['ts']
        self.cpu_end_time = event['ts'] + event['dur']

    def associate_kernel_id(self, kernel_id):
        self.kernel_id = kernel_id

    def associate_kernel(self, event):
        # TODO: need to obtain more info than just timestamps, like kernel
        self.start_time = event['ts']
        self.duration = event['dur']
        self.end_time = self.start_time + self.duration

    def associate_timestamps(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time