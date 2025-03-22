from ..common.torch_event import CudaLocal

# TODO: Naive implementation, read torch profiler code to get more accurate information

def is_backward_pass(op):
    assert isinstance(op, CudaLocal)
    return "Backward" in op.name or "autograd" in op.name

def is_optimizer_step(op):
    assert isinstance(op, CudaLocal)
    return "Optimizer" in op.name

def is_c10d_communication(op):
    assert isinstance(op, CudaLocal)
    return "c10d" in op.name

def is_forward_pass(op):
    assert isinstance(op, CudaLocal)
    return not is_backward_pass(op) and \
           not is_optimizer_step(op) and \
           not is_c10d_communication(op)