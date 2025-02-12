'''
Reflect the meaning of values in the NCCL debugging logs. This information comes from the NCCL database.
'''
from typing import *

from .data_flow import *


def probe_coll_op(ring_topology, coll_op):
    data_flows = {}
    dependencies = {}

    participating_devs = []

    if coll_op.func == "Send":
        participating_devs = [coll_op.device]

    elif coll_op.func == "Recv":
        participating_devs = [coll_op.device]

    else:
        # Collective operations that involve all devices
        participating_devs = ring_topology.devices

        if coll_op.func == "Broadcast":
            # Based on NCCL implementation in src/device/broadcast.h:runRing()
            # TODO: Naive understanding for now, need more details
            for _, ring in ring_topology.rings.items():
                cur_node = coll_op.root_rank
                next_node = ring.get_next_node(cur_node)

                prev_flow_id = None
                while next_node != coll_op.root_rank:
                    # send data to next GPU
                    cur_flow = NcclDataFlow(ring_topology.rank_to_device(cur_node),
                                            ring_topology.rank_to_device(next_node),
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
            nranks = ring_topology.size
            for _, ring in ring_topology.rings.items():
                for idx, rank in enumerate(ring.nodes):
                    # 1. reduce and copy to next GPU (nranks-1 steps)
                    # my understanding: cur_node reduce on its own node, and transfer whatever it receives to the next GPU as is
                    # push data to next GPU
                    prev_flow_id = None
                    j = 0
                    cur_node = rank
                    next_node = ring.get_next_node(cur_node)
                    while j < (nranks-1):
                        cur_flow = NcclDataFlow(ring_topology.rank_to_device(cur_node),
                                                ring_topology.rank_to_device(next_node),
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
                        cur_flow = NcclDataFlow(ring_topology.rank_to_device(cur_node),
                                                ring_topology.rank_to_device(next_node),
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
            nranks = ring_topology.size
            for _, ring in ring_topology.rings.items():
                shard_size = coll_op.data_size // nranks
                for idx, rank in enumerate(ring.nodes):
                    # 1. Push my piece of data to next GPU
                    cur_node = rank
                    next_node = ring.get_next_node(cur_node)
                    prev_flow_id = None
                    # 2. Pass around to other GPUs
                    while next_node != rank:
                        # Get flow
                        cur_flow = NcclDataFlow(ring_topology.rank_to_device(cur_node),
                                                ring_topology.rank_to_device(next_node),
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

    nccl_op = NcclCommunicationOperation(coll_op.func,
                                            coll_op.data_type,
                                            coll_op.data_num,
                                            coll_op.data_size,
                                            participating_devs,
                                            data_flows,
                                            dependencies)
    return nccl_op
