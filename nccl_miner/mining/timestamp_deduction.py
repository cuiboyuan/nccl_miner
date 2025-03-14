from nccl_miner.mining.data_flow import DataFlow
from nccl_miner.common.nccl_function_group import NcclPtpFunctionGroup, NcclCollectiveFunctionGroup


def deduce_flow_timestamps(coll_group):
    '''
    Deduce the start and end time of the data flows in the collective group.
    '''
    data_flows = coll_group.data_flows
    dependencies = coll_group.dependencies

    if isinstance(coll_group, NcclPtpFunctionGroup):
        for flow in data_flows.values():
            flow.associate_time(coll_group.common_overlap_start_time, coll_group.common_overlap_end_time)
            

    elif isinstance(coll_group, NcclCollectiveFunctionGroup):
        # Collective operations
        coll_op = coll_group.main_operation
        multi_ring = coll_group.get_algo()

        if coll_op.func == "Broadcast":
            for _, ring in multi_ring.rings.items():
                # Determine data flow's timestamps
                # TODO: right now, I'm assuming that the data flow duration is 
                # evenly distributed across the collective operation's duration
                # This is a naive assumption, need to refine this.
                flow_ts_offset = coll_group.common_overlap_start_time
                flow_duration = coll_group.common_overlap_end_time - coll_group.common_overlap_start_time
                flow_duration /= (ring.size - 1)

                assigned_flows = set()
                while len(assigned_flows) < len(data_flows):
                    assigned_flows_this_pass = set()
                    for flow_id, flow in data_flows.items():
                        if flow_id in assigned_flows:
                            continue
                        if flow_id not in dependencies or \
                            all(dep in assigned_flows for dep in dependencies[flow_id]):
                            flow.associate_time(flow_ts_offset, flow_ts_offset + flow_duration - 1)
                            assigned_flows_this_pass.add(flow_id)
                    assigned_flows.update(assigned_flows_this_pass)
                    flow_ts_offset += flow_duration

        elif coll_op.func == "AllReduce":
            for _, ring in multi_ring.rings.items():
                # Determine data flow's timestamps
                # TODO: right now, I'm assuming that the data flow duration is 
                # evenly distributed across the collective operation's duration
                # This is a naive assumption, need to refine this.
                flow_ts_offset = coll_group.common_overlap_start_time
                flow_duration = coll_group.common_overlap_end_time - coll_group.common_overlap_start_time
                flow_duration /= ((ring.size - 1) * 2)

                assigned_flows = set()
                while len(assigned_flows) < len(data_flows):
                    assigned_flows_this_pass = set()
                    for flow_id, flow in data_flows.items():
                        if flow_id in assigned_flows:
                            continue
                        if flow_id not in dependencies or \
                            all(dep in assigned_flows for dep in dependencies[flow_id]):
                            flow.associate_time(flow_ts_offset, flow_ts_offset + flow_duration - 1)
                            assigned_flows_this_pass.add(flow_id)
                    assigned_flows.update(assigned_flows_this_pass)
                    flow_ts_offset += flow_duration

        elif coll_op.func == "Reduce":
            pass
        elif coll_op.func == "AllGather":
            pass
        elif coll_op.func == "Gather":
            pass
        elif coll_op.func == "Scatter":
            pass
        elif coll_op.func == "ReduceScatter":
            pass