'''
Scripts that combine or correlate operations from different logs/devices.
'''
from ..common.nccl_function import NcclPtpFunction, NcclCollectiveFunction, NcclFunction
from ..common.nccl_function_group import NcclCommClique, NcclPtpFunctionGroup, NcclCollectiveFunctionGroup
from ..common.torch_event import *

def group_nccl_calls_across_devices(comm_cliques, calls_per_device):
    all_func_groups = {}  # map between group id and function group
    assigned_calls = set()  # Keep track of operations that have been assigned a group ID

    for device, calls in calls_per_device.items():
        for call in calls:
            if call in assigned_calls:
                continue  # Skip if the operation is already assigned to a group

            clique = comm_cliques[call.clique_id]
            if clique is None:
                raise ValueError(f"No clique found for clique_id {call.clique_id}")
            # Attempt to find matching operations on other devices
            matched_calls = {device: call}
            if isinstance(call, NcclPtpFunction):
                for other_device, other_calls in calls_per_device.items():
                    if other_device == device:
                        continue
                    for other_call in other_calls:
                        if other_call in assigned_calls:
                            continue
                        if isinstance(other_call, NcclPtpFunction):
                            # Implement the logic to determine if `call` matches `other_call` for PTP functions
                            if (call.clique_id == other_call.clique_id and
                                ((call.func == "Send" and other_call.func == "Recv") or
                                 (call.func == "Recv" and other_call.func == "Send")) and
                                call.device == clique.get_rank_device(other_call.peer_rank) and
                                other_call.device == clique.get_rank_device(call.peer_rank) and
                                call.data_num == other_call.data_num and
                                call.data_type == other_call.data_type and
                                call.data_size == other_call.data_size):
                                matched_calls[other_device] = other_call
                                break

            elif isinstance(call, NcclCollectiveFunction):
                if len(clique.rank_to_device) == 1:  # Special case: nrank is 1
                    func_group = NcclCollectiveFunctionGroup({device: call}, clique)
                    all_func_groups[func_group.id] = func_group
                    assigned_calls.add(call)
                    continue

                if len(clique.rank_to_device) > 1:  # Ensure it's a multi-device clique
                    for other_device, other_calls in calls_per_device.items():
                        if other_device == device:
                            continue
                        for other_call in other_calls:
                            if other_call in assigned_calls:
                                continue
                            if isinstance(other_call, NcclCollectiveFunction):
                                # Implement the logic to determine if `call` matches `other_call` for Collective functions
                                if (call.clique_id == other_call.clique_id and
                                    call.func == other_call.func and
                                    call.data_num == other_call.data_num and
                                    call.data_type == other_call.data_type and
                                    call.data_size == other_call.data_size):
                                    matched_calls[other_device] = other_call
                                    break

                    # Ensure all devices in the clique have a matched call
                    if len(matched_calls) != len(clique.rank_to_device):
                        matched_calls = {device: call}  # Reset to only include the original call

            # If a match is found, create a function group and assign a group ID
            if len(matched_calls) > 1:
                if isinstance(call, NcclPtpFunction):
                    func_group = NcclPtpFunctionGroup(matched_calls, clique)
                elif isinstance(call, NcclCollectiveFunction):
                    func_group = NcclCollectiveFunctionGroup(matched_calls, clique)
                func_group_id = func_group.id
                all_func_groups[func_group_id] = func_group
                for matched_device, matched_call in matched_calls.items():
                    assigned_calls.add(matched_call)
            else:
                # If no match is found, raise an error
                raise ValueError(f"No matching operations found for call {call} on device {device}")

    return all_func_groups


def link_nccl_torch_calls(nccl_calls_per_device, torch_calls_per_device):
    gpu_ops = {}
    for device in nccl_calls_per_device:
        # Sanity check
        try:
            assert torch_calls_per_device[device]['host'] == nccl_calls_per_device[device]['host']
            assert torch_calls_per_device[device]['pid'] == nccl_calls_per_device[device]['pid']
        except AssertionError:
            print("[ERROR] Sanity check failed, below should be equal:")
            print(f"torch device {device}: {torch_calls_per_device[device]['host']}:{torch_calls_per_device[device]['pid']}")
            print(f"nccl device  {device}: {nccl_calls_per_device[device]['host']}:{nccl_calls_per_device[device]['pid']}")
            raise AssertionError
        print(f"Host: {torch_calls_per_device[device]['host']}, PID: {torch_calls_per_device[device]['pid']}")

        torch_ops = torch_calls_per_device[device]['operations']
        nccl_ops = nccl_calls_per_device[device]['operations']
        print(f"number of torch ops: {len(torch_ops)}")
        print(f"number of nccl ops: {len(nccl_ops)}")

        torch_idx = 0
        for nccl_op in nccl_ops:
            assert isinstance(nccl_op, NcclPtpFunction) or \
                  isinstance(nccl_op, NcclCollectiveFunction)
            # Find the corresponding torch operation
            torch_op = torch_ops[torch_idx]

            if not torch_op.has_kernel:
                print(f"Skipping torch op {torch_op} due to no associated kernel calls.")
                torch_idx += 1
                continue

            if isinstance(torch_op, CudaCollective) and \
                isinstance(nccl_op, NcclCollectiveFunction):
                # Ensure it's the same comm call
                try:
                    assert torch_op.name == nccl_op.func
                except AssertionError:
                    print("[ERROR] Sanity check failed, below should be equal:")
                    print(f"torch event: {torch_op.name}")
                    print(f"nccl event: {nccl_op.func}")
                    raise AssertionError
                # link info from torch to nccl
                nccl_op.associate_algo(torch_op.algo)
                nccl_op.associate_protocol(torch_op.protocol)
                nccl_op.associate_time(torch_op.start_time, torch_op.end_time)
                nccl_op.associate_semantics(torch_op.semantics)
                torch_idx += 1
            elif isinstance(torch_op, CudaPtp) and \
                isinstance(nccl_op, NcclPtpFunction):
                # Ensure it's the same comm call
                try:
                    assert nccl_op.func == "Send" or nccl_op.func == "Recv"
                except AssertionError:
                    print("[ERROR] Sanity check failed, below should be Send or Recv:")
                    print(f"nccl op: {nccl_op.func}")
                    raise AssertionError
                # link info from torch to nccl
                nccl_op.associate_time(torch_op.start_time, torch_op.end_time)
                nccl_op.associate_semantics(torch_op.semantics)
                torch_idx += 1
            elif isinstance(torch_op, CudaLocal):
                pass

            if device in gpu_ops:
                gpu_ops[device].append(nccl_op)
            else:
                gpu_ops[device] = [nccl_op]

    return gpu_ops