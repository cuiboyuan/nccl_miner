'''
PyTorch operations to be parsed in Torch Profiler data.
'''
import re
from ..misc.logger import *
from .torch_utils import *

class Semantics():
    def __init__(self, name=None):
        self.name = name

class CudaOp():
    def __init__(self, name=None, device=None):
        self.name = name
        self.device = device
        self.start_time = None
        self.duration = None
        self.end_time = None

        self.cpu_start_time = None
        self.cpu_end_time = None

    def associate_cpu_event(self, event):
        self.cpu_start_time = event['ts']
        self.cpu_end_time = event['ts'] + event['dur']

    def associate_timestamps(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time

    def __repr__(self):
        return f"{self.name} on device {self.device} at {self.start_time} for {self.duration} ms"


class CudaLocal(CudaOp):
    def __init__(self, name=None, device=None):
        super().__init__(name, device)
        self.semantics = Semantics("Unknown")

    def associate_context_events(self, active_events, past_relevant_events, past_local_ops=None):
        # TODO: Naive implementation for now, need to deduce the semantics of the data
        sorted_context_events = sorted(active_events, key=lambda event: event['ts'])
        context = [event for event in sorted_context_events]
        self.name = context[0]['name']
        if is_forward_pass(self.name):
            self.semantics.name = "Forward Pass"
        elif is_backward_pass(self.name):
            self.semantics.name = "Backward Pass"
        elif is_optimizer_step(self.name):
            self.semantics.name = "Optimizer Step"
        elif is_c10d_communication(self.name):
            self.semantics.name = "C10D Communication"

    def associate_kernel(self, event):
        # TODO: need to obtain more info than just timestamps, like kernel
        self.associate_timestamps(event['ts'], event['ts'] + event['dur'])


class CudaComm(CudaOp):
    def __init__(self, func=None, device=None):
        super().__init__(func, device)
        self.data_type = None
        self.has_kernel = False

        self.semantics = Semantics("Unknown")

    def associate_context_events(self, active_events, past_relevant_events, past_local_ops=None):
        pass


    def __repr__(self):
        return f"{self.name} at {self.start_time} for {self.duration} ms"


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

        func_name, op, dtype, algo, protocol = None, None, None, None, None
        if reduce_match:
            func_name, op, dtype, algo, protocol = reduce_match.groups()
        elif broadcast_match:
            func_name, algo, protocol = broadcast_match.groups()
        else:
            return

        try:
            assert self.name is None or func_name == self.name
        except AssertionError:
            loge(f"AssertionError: {self.name} != {func_name}")

        self.associate_timestamps(event['ts'], event['ts'] + event['dur'])
        self.device = event['pid']
        self.name = func_name
        self.op = op
        self.data_type = dtype # TODO: change this to NcclDataType
        self.algo = algo
        self.protocol = protocol

class CudaPtp(CudaComm):
    def __init__(self, func=None, device=None):
        super().__init__(func, device)

    def associate_kernel(self, event):
        ptp_pattern = r"ncclDevKernel_SendRecv\(ncclDevComm\*, unsigned long, ncclWork\*\)"
        ptp_match = re.match(ptp_pattern, event['name'])
        if ptp_match:
            self.device = event['pid']
            self.associate_timestamps(event['ts'], event['ts'] + event['dur'])
