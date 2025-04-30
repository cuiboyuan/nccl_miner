'''
Extract data flows and their dependencies from NCCL operations.
Here, I'm deducing the data flow and depdencies from the collective operations based on my understanding of the NCCL codebase.
NCCL codebase: https://github.com/NVIDIA/nccl
'''
from typing import *

from .data_flow import *
from ..common.nccl_function_group import NcclCollectiveFunctionGroup, NcclPtpFunctionGroup


def deduce_flow_dependencies(coll_group):
    data_flows = {}
    dependencies = {}

    if isinstance(coll_group, NcclPtpFunctionGroup):
        flow = DataFlow(coll_group.src, coll_group.dst, coll_group.data_size)
        data_flows[flow.id] = flow

    elif isinstance(coll_group, NcclCollectiveFunctionGroup):
        # Collective operations
        coll_op = coll_group.main_operation
        multi_ring = coll_group.get_algo()

        if coll_op.func == "Broadcast":
            # Based on NCCL implementation in src/device/broadcast.h:runRing()
            # TODO: Naive understanding for now, need more details
            for _, ring in multi_ring.rings.items():
                cur_node = coll_op.root_rank
                next_node = ring.get_next_node(cur_node)

                prev_flow_id = None
                while next_node != coll_op.root_rank:
                    # send data to next GPU
                    cur_flow = DataFlow(multi_ring.rank_to_device(cur_node),
                                            multi_ring.rank_to_device(next_node),
                                            coll_op.data_size,
                                            name=f"{coll_op.root_rank}'s Data")

                    cur_flow_id = cur_flow.id
                    data_flows[cur_flow_id] = cur_flow
                    # add dependencies
                    # flow from the root is the first flow, it has zero deps
                    if cur_node != coll_op.root_rank:
                        if prev_flow_id is None:
                            # should not enter here.
                            assert False
                        # need to wait for the previous flow to finish.
                        if cur_flow_id not in dependencies:
                            dependencies[cur_flow_id] = [prev_flow_id]
                        else:
                            dependencies[cur_flow_id].append(prev_flow_id)
                    # done cur data flow, move on to the next
                    prev_flow_id = cur_flow_id

                    cur_node = next_node
                    next_node = ring.get_next_node(cur_node)

            
        elif coll_op.func == "AllReduce":
            # Based on NCCL implementation in src/device/all_reduce.h:runRing()
            # TODO: Naive understanding, need more reading on NCCL codes
            nranks = multi_ring.size
            for _, ring in multi_ring.rings.items():
                for idx, rank in enumerate(ring.nodes):
                    # 1. reduce and copy to next GPU (nranks-1 steps)
                    # my understanding: cur_node reduce on its own node, and transfer whatever it receives to the next GPU as is
                    # push data to next GPU
                    prev_flow_id = None
                    j = 0
                    cur_node = rank
                    next_node = ring.get_next_node(cur_node)
                    while j < (nranks-1):
                        cur_flow = DataFlow(multi_ring.rank_to_device(cur_node),
                                                multi_ring.rank_to_device(next_node),
                                                coll_op.data_size,
                                                name=f"{rank}'s data")
                        cur_flow_id = cur_flow.id
                        data_flows[cur_flow_id] = cur_flow

                        if prev_flow_id is None:
                            # first flow, zero dependency
                            pass
                        else:
                            # need to wait for the previous flow to finish.                    
                            if cur_flow_id not in dependencies:
                                dependencies[cur_flow_id] = [prev_flow_id]
                            else:
                                dependencies[cur_flow_id].append(prev_flow_id)
                        prev_flow_id = cur_flow_id

                        cur_node = next_node
                        next_node = ring.get_next_node(cur_node)
                        j += 1
                    # 2. copy to next GPU (nranks-1 steps)
                    # my understanding: now, broadcast the final reduced result to all nodes in the ring.
                    while j < 2*(nranks-1):
                        cur_flow = DataFlow(multi_ring.rank_to_device(cur_node),
                                                multi_ring.rank_to_device(next_node),
                                                coll_op.data_size,
                                                name=f"all-reduced result")
                        cur_flow_id = cur_flow.id
                        data_flows[cur_flow_id] = cur_flow

                        assert prev_flow_id is not None
                        # need to wait for the previous flow to finish.
                        if cur_flow_id not in dependencies:
                            dependencies[cur_flow_id] = [prev_flow_id]
                        else:
                            dependencies[cur_flow_id].append(prev_flow_id)                        
                        prev_flow_id = cur_flow_id

                        cur_node = next_node
                        next_node = ring.get_next_node(cur_node)
                        j += 1

        elif coll_op.func == "AllGather":
            # Based on NCCL implementation in src/device/all_gather.h:runRing()
            # TODO: Naive understanding, need more reading on NCCL codes
            nranks = multi_ring.size
            for _, ring in multi_ring.rings.items():
                shard_size = coll_op.data_size // nranks
                for idx, rank in enumerate(ring.nodes):
                    # 1. Push my piece of data to next GPU
                    cur_node = rank
                    next_node = ring.get_next_node(cur_node)
                    prev_flow_id = None
                    # 2. Pass around to other GPUs
                    while next_node != rank:
                        # Get flow
                        cur_flow = DataFlow(multi_ring.rank_to_device(cur_node),
                                                multi_ring.rank_to_device(next_node),
                                                shard_size,
                                                name=f"{rank}'s data shard")
                        cur_flow_id = cur_flow.id
                        data_flows[cur_flow_id] = cur_flow

                        if prev_flow_id is not None:
                            # need to wait for the previous flow to finish.
                            if cur_flow_id not in dependencies:
                                dependencies[cur_flow_id] = [prev_flow_id]
                            else:
                                dependencies[cur_flow_id].append(prev_flow_id)

                        # advance to the next pair in Ring
                        cur_node = next_node
                        next_node = ring.get_next_node(cur_node)
                        prev_flow_id = cur_flow_id

        elif coll_op.func == "Reduce":
            pass

        elif coll_op.func == "ReduceScatter":
            pass

    return data_flows, dependencies
