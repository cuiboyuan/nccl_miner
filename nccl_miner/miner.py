
from .nccl_parser import *
from .mining.core import probe_coll_op
from .mining.data_flow import *

def extract_flows_from_logs(log_files):

    comm_cliques, comms_per_device = parse_nccl_calls_from_logs(log_files)
    
    '''Step 2.
    Based on Clique and Func Calls Per Device info, group Func Calls into Operations.
    A Collective Operation is at the Clique-level, involving all devices within that Clique.
    A PTP (SendRecv) Operation only involves src & dst devices.
    All operations will be ordered based on earliest finish time (EFT). TODO: Think about whether this is an issue.
    Operations will be probed to extract Data Flow information
    '''
    cur_idx_per_device = {}
    max_idx_per_device = {}
    for cur_dev in comms_per_device:
        cur_idx_per_device[cur_dev] = 0
        max_idx_per_device[cur_dev] = len(comms_per_device[cur_dev])

    def are_all_logs_parsed():
        for dev, idx in cur_idx_per_device.items():
            if idx < max_idx_per_device[dev]:
                return False
        return True
    
    # For progress tracking only
    total_iter = sum([max_idx for dev, max_idx in max_idx_per_device.items()])
    cur_iter = 0

    # TODO: Assume every operation blocks.

    all_comms = []

    pending_ptp_calls = []
    pending_coll_calls = []
    while are_all_logs_parsed() == False:
        for dev, nccl_calls in comms_per_device.items():
            cur_idx = cur_idx_per_device[dev]
            max_idx = max_idx_per_device[dev]
            if cur_idx >= max_idx:
                continue

            cur_nccl_call = nccl_calls[cur_idx]
            if isinstance(cur_nccl_call, NcclPtp):
                cur_cliq_id = cur_nccl_call.clique_id
                clique = comm_cliques[cur_cliq_id]

                unblocked = False
                for _ in range(len(pending_ptp_calls)):
                    pending_ptp = pending_ptp_calls.pop(0)

                    pending_cliq_id = pending_ptp.clique_id
                    if cur_cliq_id == pending_cliq_id:
                        cur_func = cur_nccl_call.func
                        cur_device = cur_nccl_call.device
                        cur_peer = clique.get_rank_device(cur_nccl_call.peer_rank)

                        pending_func = pending_ptp.func
                        pending_device = pending_ptp.device
                        pending_peer = clique.get_rank_device(pending_ptp.peer_rank)
                        if (cur_func == 'Send' and pending_func == "Recv") \
                            or (cur_func == 'Recv' and pending_func == "Send"):
                            if ((cur_device == pending_peer) \
                                or (cur_peer == pending_device)):
                                # Unblocked
                                if (cur_nccl_call.data_num == pending_ptp.data_num \
                                    and cur_nccl_call.data_type == pending_ptp.data_type \
                                    and cur_nccl_call.data_size == pending_ptp.data_size):
                                    # Unblocked, add this to final comm operations
                                    if cur_func == 'Send':
                                        src_dev = cur_device
                                        dst_dev = pending_device
                                    else:
                                        src_dev = pending_device
                                        dst_dev = cur_device

                                    flow = NcclDataFlow(src_dev, dst_dev, cur_nccl_call.data_size)
                                    sendrecv_op = NcclCommunicationOperation(
                                        "SendRecv",
                                        cur_nccl_call.data_type,
                                        cur_nccl_call.data_num,
                                        cur_nccl_call.data_size,
                                        [src_dev, dst_dev],
                                        {flow.id: flow},
                                        {})
                                    all_comms.append(sendrecv_op)
                                    unblocked = True
                                    break
                    # Not unblocked, continue waiting in pending queue
                    pending_ptp_calls.append(pending_ptp)

                if not unblocked:
                    pending_ptp_calls.append(cur_nccl_call)

            elif isinstance(cur_nccl_call, NcclCollective):
                cur_cliq_id = cur_nccl_call.clique_id
                clique = comm_cliques[cur_cliq_id]

                unblocked = False
                matched = False
                for _ in range(len(pending_coll_calls)):
                    # print(">>>")
                    pending_coll, num_matched = pending_coll_calls.pop(0)
                    # print(pending_coll_calls)

                    pending_cliq_id = pending_coll.clique_id
                    cur_func = cur_nccl_call.func
                    pending_func = pending_coll.func
                    if cur_cliq_id == pending_cliq_id \
                        and cur_func == pending_func:
                        # Matched.
                        if (cur_nccl_call.data_num == pending_coll.data_num \
                            and cur_nccl_call.data_type == pending_coll.data_type \
                            and cur_nccl_call.data_size == pending_coll.data_size):
                            # Matched.
                            matched = True
                            num_matched += 1
                            if num_matched == len(clique.rank_to_device):
                                unblocked = True
                                coll_op = probe_coll_op(clique.ring_algo, cur_nccl_call)
                                all_comms.append(coll_op)
                                break
                    # not unblocked, continue waiting in pending queue
                    pending_coll_calls.append((pending_coll, num_matched))
                    # print(pending_coll_calls)
                    # print("<<<")
                    
                if not unblocked and not matched:
                    if len(clique.rank_to_device) > 1:
                        pending_coll_calls.append((cur_nccl_call, 1))
                    else:
                        coll_op = probe_coll_op(clique.ring_algo, cur_nccl_call)
                        all_comms.append(coll_op)


            cur_idx_per_device[dev] += 1

    assert len(pending_ptp_calls) == 0
    assert len(pending_coll_calls) == 0
    
    return all_comms
