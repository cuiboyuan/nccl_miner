from nccl_miner.mining.data_flow import DataFlow
from nccl_miner.common.nccl_function_group import NcclPtpFunctionGroup, NcclCollectiveFunctionGroup


def deduce_flow_timestamps(coll_group):
    '''
    Deduce the start and end time of the data flows in the collective group.
    '''
    data_flows = coll_group.data_flows
    dependencies = coll_group.dependencies

    if isinstance(coll_group, NcclPtpFunctionGroup):
        flow = DataFlow(coll_group.src, coll_group.dst, coll_group.data_size,
                        start_time=coll_group.common_overlap_start_time, end_time=coll_group.common_overlap_end_time)
        data_flows[flow.id] = flow

    elif isinstance(coll_group, NcclCollectiveFunctionGroup):
        coll_op = coll_group.main_operation
        multi_ring = coll_group.get_algo()

        if coll_op.func == "Broadcast":
            for _, ring in multi_ring.rings.items():
                cur_node = coll_op.root_rank
                next_node = ring.get_next_node(cur_node)

                flow_ts_offset = coll_group.common_overlap_start_time
                flow_duration = coll_group.common_overlap_end_time - coll_group.common_overlap_start_time
                flow_duration /= (ring.size - 1)

                while next_node != coll_op.root_rank:
                    cur_flow = DataFlow(multi_ring.rank_to_device(cur_node),
                                        multi_ring.rank_to_device(next_node),
                                        coll_op.data_size,
                                        start_time=flow_ts_offset,
                                        end_time=flow_ts_offset + flow_duration - 1)
                    flow_ts_offset += flow_duration
                    data_flows[cur_flow.id] = cur_flow

                    cur_node = next_node
                    next_node = ring.get_next_node(cur_node)

        elif coll_op.func == "AllReduce":
            nranks = multi_ring.size
            for _, ring in multi_ring.rings.items():
                coll_group_offset = coll_group.common_overlap_start_time
                coll_group_duration = coll_group.common_overlap_end_time - coll_group.common_overlap_start_time
                flow_duration = coll_group_duration / ((ring.size - 1) * 2)

                for rank in ring.nodes:
                    flow_ts_offset = coll_group_offset
                    j = 0
                    cur_node = rank
                    next_node = ring.get_next_node(cur_node)
                    while j < (nranks - 1):
                        cur_flow = DataFlow(multi_ring.rank_to_device(cur_node),
                                            multi_ring.rank_to_device(next_node),
                                            coll_op.data_size,
                                            start_time=flow_ts_offset,
                                            end_time=flow_ts_offset + flow_duration - 1)
                        data_flows[cur_flow.id] = cur_flow
                        flow_ts_offset += flow_duration

                        cur_node = next_node
                        next_node = ring.get_next_node(cur_node)
                        j += 1

                    while j < 2 * (nranks - 1):
                        cur_flow = DataFlow(multi_ring.rank_to_device(cur_node),
                                            multi_ring.rank_to_device(next_node),
                                            coll_op.data_size,
                                            start_time=flow_ts_offset,
                                            end_time=flow_ts_offset + flow_duration - 1)
                        data_flows[cur_flow.id] = cur_flow
                        flow_ts_offset += flow_duration

                        cur_node = next_node
                        next_node = ring.get_next_node(cur_node)
                        j += 1

        elif coll_op.func == "AllGather":
            nranks = multi_ring.size
            for _, ring in multi_ring.rings.items():
                coll_group_offset = coll_group.common_overlap_start_time
                coll_group_duration = coll_group.common_overlap_end_time - coll_group.common_overlap_start_time
                flow_duration = coll_group_duration / ((ring.size - 1) * 2)

                shard_size = coll_op.data_size // nranks
                for rank in ring.nodes:
                    flow_ts_offset = coll_group_offset
                    cur_node = rank
                    next_node = ring.get_next_node(cur_node)

                    while next_node != rank:
                        cur_flow = DataFlow(multi_ring.rank_to_device(cur_node),
                                            multi_ring.rank_to_device(next_node),
                                            shard_size,
                                            start_time=flow_ts_offset,
                                            end_time=flow_ts_offset + flow_duration - 1)
                        data_flows[cur_flow.id] = cur_flow
                        flow_ts_offset += flow_duration

                        cur_node = next_node
                        next_node = ring.get_next_node(cur_node)

    return data_flows