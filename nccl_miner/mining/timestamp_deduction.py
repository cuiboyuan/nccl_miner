from pulp import LpProblem, LpVariable, value, LpMinimize, PULP_CBC_CMD

from nccl_miner.misc.logger import *
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
            assert isinstance(flow, DataFlow)

            src_op_start_ts = coll_group.get_src_operation().start_time
            src_op_end_ts = coll_group.get_src_operation().end_time
            dst_op_start_ts = coll_group.get_dst_operation().start_time
            dst_op_end_ts = coll_group.get_dst_operation().end_time

            # Calculate the duration for send and receive for the data flow.
            # TODO: naive assumption, need better estimation.
            total_duration = (dst_op_end_ts - src_op_start_ts) // 2
            flow_send_start = src_op_start_ts
            flow_send_end = min(src_op_end_ts, src_op_start_ts + total_duration)

            # NOTE: this is for visualization only. I want to draw an arrow from src sending
            # to dst receiving to represent a data flow, but Perfetto UI doesn't support the start
            # time of an arrow to be greater than its end time.
            flow_recv_start = max(flow_send_end, dst_op_start_ts) + 1 # add 1 for visualization purpose
            flow_recv_end = dst_op_end_ts

            # Ensure the calculated times meet the constraints
            if flow_recv_start > dst_op_end_ts:
                raise ValueError("flow_recv_start > dst_op_end_ts: Cannot start receiving after the Recv op on dst has ended.")
            if flow_send_end > src_op_end_ts:
                raise ValueError("flow_send_end >= src_op_end_ts: Cannot continue sending after the Send op on src has ended.")

            flow.associate_time(flow_send_start, flow_send_end, flow_recv_start, flow_recv_end)

    elif isinstance(coll_group, NcclCollectiveFunctionGroup):
        ts_deduction_problem = LpProblem("Timestamp_Deduction", LpMinimize)

        lp_vars = {}
        # PuLP can't solve var with large numbers, need to shift the ts.
        min_time = min(
            min(coll_group.device_operations[flow.src].start_time, coll_group.device_operations[flow.dst].start_time)
            for flow in data_flows.values()
        )
        #
        # Create the LP variables for the start and end time of the Send and Recv operations of each data flow
        #
        for flow_id, flow in data_flows.items():
            assert isinstance(flow, DataFlow)
            src_device = flow.src
            dst_device = flow.dst

            src_op_start_ts = coll_group.device_operations[src_device].start_time - min_time
            src_op_end_ts = coll_group.device_operations[src_device].end_time - min_time
            dst_op_start_ts = coll_group.device_operations[dst_device].start_time - min_time
            dst_op_end_ts = coll_group.device_operations[dst_device].end_time - min_time

            lp_vars[flow_id] = {
            "send_start": LpVariable(f"flow_{flow_id}_send_start",
                lowBound=src_op_start_ts, upBound=src_op_end_ts, cat="Continuous"),
            "send_end": LpVariable(f"flow_{flow_id}_send_end",
                  lowBound=src_op_start_ts, upBound=src_op_end_ts, cat="Continuous"),
            "recv_start": LpVariable(f"flow_{flow_id}_recv_start",
                lowBound=dst_op_start_ts, upBound=dst_op_end_ts, cat="Continuous"),
            "recv_end": LpVariable(f"flow_{flow_id}_recv_end",
                  lowBound=dst_op_start_ts, upBound=dst_op_end_ts, cat="Continuous")
            }
        #
        # Add LP constraints for flow dependencies
        #
        for cur_flow_id, dep_flow_ids in dependencies.items():
            for dep_flow_id in dep_flow_ids:
                # LP Constraint: a flow only start after its dependent flow ends
                ts_deduction_problem += lp_vars[cur_flow_id]["send_start"] - lp_vars[dep_flow_id]["recv_end"] >= 1

        #
        # Flow duration estimation
        #
        # First, estimate the "Ground Truth" of each data flow's duration.
        # Right now, it is based on coll_group's duration on each device.
        # TODO: Re-visit this part, and see if we can get a better "Ground truth".
        num_flow_op_per_device = {device : 0 for device in coll_group.device_operations}
        total_bytes_per_device = {device : 0 for device in coll_group.device_operations}
        for flow_id, flow in data_flows.items():
            assert isinstance(flow, DataFlow)
            num_flow_op_per_device[flow.src] += 1
            num_flow_op_per_device[flow.dst] += 1
            total_bytes_per_device[flow.src] += flow.size
            total_bytes_per_device[flow.dst] += flow.size
        dur_per_flow = {}
        for flow_id, flow in data_flows.items():
            assert isinstance(flow, DataFlow)
            coll_op_src_dur = coll_group.device_operations[flow.src].end_time - coll_group.device_operations[flow.src].start_time
            coll_op_dst_dur = coll_group.device_operations[flow.dst].end_time - coll_group.device_operations[flow.dst].start_time
            flow_send_dur = coll_op_src_dur * (flow.size / total_bytes_per_device[flow.src])
            flow_recv_dur = coll_op_dst_dur * (flow.size / total_bytes_per_device[flow.dst])
            dur_per_flow[flow_id] = (flow_send_dur, flow_recv_dur)
        #
        # Create LP vars for deviation between estimated and actual duration, and
        # add LP constraints for the final LP objective.
        #
        dur_deviation_vars = []
        for flow_id, flow_var in lp_vars.items():
            send_dur, recv_dur = dur_per_flow[flow_id]
            # LP Constraint:
            # the send/recv duration of the flow should be as close as possible to the estimated duration
            send_deviation = LpVariable(f"flow_{flow_id}_send_deviation", lowBound=0, cat="Continuous")
            ts_deduction_problem += (flow_var["send_end"] - flow_var["send_start"]) - send_dur <= send_deviation
            ts_deduction_problem += send_dur - (flow_var["send_end"] - flow_var["send_start"]) <= send_deviation
            dur_deviation_vars.append(send_deviation)
            recv_deviation = LpVariable(f"flow_{flow_id}_recv_deviation", lowBound=0, cat="Continuous")
            ts_deduction_problem += (flow_var["recv_end"] - flow_var["recv_start"]) - recv_dur <= recv_deviation
            ts_deduction_problem += recv_dur - (flow_var["recv_end"] - flow_var["recv_start"]) <= recv_deviation
            dur_deviation_vars.append(recv_deviation)
            # LP Constraint:
            # Recv can only start after Send has started.
            ts_deduction_problem += flow_var["recv_start"] - flow_var["send_start"] >= 1

            # LP Constraint:
            # Send/recv duration of the flow should be at least 1 us
            # NOTE: mainly for visualization purpose, but it's also unlikely in real life
            # for the duration to be less than 1 us.
            ts_deduction_problem += flow_var["send_end"] - flow_var["send_start"] >= 1
            ts_deduction_problem += flow_var["recv_end"] - flow_var["recv_start"] >= 1
            # LP Constraint:
            # Recv can only start after Send has ended.
            # NOTE: this is for visualization only. I want to draw an arrow from src sending
            # to dst receiving to represent a data flow, but Perfetto UI doesn't support the start
            # time of an arrow to be greater than its end time.
            ts_deduction_problem += flow_var["recv_start"] - flow_var["send_end"] >= 1

        # LP Objective: minimize the sum of all deviations
        ts_deduction_problem += sum(dur_deviation_vars), "Objective"

        # Solve the LP and error handling
        logd("Solving LP...")
        ts_deduction_problem.solve(PULP_CBC_CMD(msg=False))
        if ts_deduction_problem.status == -1:
            raise ValueError("Failed to solve the timestamp deduction problem.")

        #
        # Restore the original ts back from the shift at the start.
        #
        for flow_id, lp_var in lp_vars.items():
            data_flows[flow_id].associate_time(
                (value(lp_var["send_start"]) + min_time),
                (value(lp_var["send_end"]) + min_time),
                (value(lp_var["recv_start"]) + min_time),
                (value(lp_var["recv_end"]) + min_time))

        # Print the final LP solution
        for flow_id, lp_var in lp_vars.items():
            logd(f"Flow {flow_id}:")
            logd(f"  send_start: {value(lp_var['send_start']) + min_time}")
            logd(f"  send_end: {value(lp_var['send_end']) + min_time}")
            logd(f"  recv_start: {value(lp_var['recv_start']) + min_time}")
            logd(f"  recv_end: {value(lp_var['recv_end']) + min_time}")