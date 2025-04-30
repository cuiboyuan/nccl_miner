
# TODO: Naive implementation, read torch profiler code to get more accurate information

def is_backward_pass(name):
    return "Backward" in name or "autograd" in name

def is_optimizer_step(name):
    return "Optimizer" in name

def is_c10d_communication(name):
    return "c10d" in name

def is_forward_pass(name):
    return not is_backward_pass(name) and \
           not is_optimizer_step(name) and \
           not is_c10d_communication(name)
