from pulp import LpProblem, LpVariable, value, LpMinimize

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

            # Calculate the duration for src and dst flows
            # TODO: naive assumption, need better estimation.
            total_duration = (dst_op_end_ts - src_op_start_ts) // 2
            src_flow_start = src_op_start_ts
            src_flow_end = min(src_op_end_ts, src_op_start_ts + total_duration)

            dst_flow_start = max(src_flow_end, dst_op_start_ts) + 1 # add 1 for visualization purpose
            dst_flow_end = dst_op_end_ts

            # Ensure the calculated times meet the constraints
            if dst_flow_start >= dst_flow_end:
                raise ValueError("Destination flow start time exceeds or equals destination operation end time.")
            if src_flow_end > src_op_end_ts:
                raise ValueError("Source flow end time exceeds source operation end time.")

            flow.associate_time(src_flow_start, src_flow_end, dst_flow_start, dst_flow_end)

    elif isinstance(coll_group, NcclCollectiveFunctionGroup):
        ts_deduction_problem = LpProblem("Timestamp Deduction", LpMinimize)

        lp_vars = {}
        # PuLP can't solve var with large numbers, need to shift the ts.
        min_time = min(
            min(coll_group.device_operations[flow.src].start_time, coll_group.device_operations[flow.dst].start_time)
            for flow in data_flows.values()
        )
        #
        # Create the LP variables for the start and end time of each data flow
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
            "src_start": LpVariable(f"flow_{flow_id}_src_start",
                lowBound=src_op_start_ts, upBound=src_op_end_ts, cat="Continuous"),
            "src_end": LpVariable(f"flow_{flow_id}_src_end",
                  lowBound=src_op_start_ts, upBound=src_op_end_ts, cat="Continuous"),
            "dst_start": LpVariable(f"flow_{flow_id}_dst_start",
                lowBound=dst_op_start_ts, upBound=dst_op_end_ts, cat="Continuous"),
            "dst_end": LpVariable(f"flow_{flow_id}_dst_end",
                  lowBound=dst_op_start_ts, upBound=dst_op_end_ts, cat="Continuous")
            }
        #
        # Add LP constraints for flow dependencies
        #
        for cur_flow_id, dep_flow_ids in dependencies.items():
            for dep_flow_id in dep_flow_ids:
                # LP Constraint: a flow only start after its dependent flow ends
                ts_deduction_problem += lp_vars[cur_flow_id]["src_start"] - lp_vars[dep_flow_id]["dst_end"] >= 1

        #
        # Flow duration estimation
        #
        # First, estimate the "Ground Truth" of each data flow's duration.
        # Right now, it is based on coll_group's duration on each device
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
            flow_src_dur = coll_op_src_dur * (flow.size / total_bytes_per_device[flow.src])
            flow_dst_dur = coll_op_dst_dur * (flow.size / total_bytes_per_device[flow.dst])
            dur_per_flow[flow_id] = (flow_src_dur, flow_dst_dur)
        #
        # Create LP vars for deviation between estimated and actual duration, and
        # add LP constraints for the final LP objective.
        #
        dur_deviation_vars = []
        for flow_id, flow_var in lp_vars.items():
            src_dur, dst_dur = dur_per_flow[flow_id]
            # LP Constraint:
            # the src/dst flow duration should be as close as possible to the estimated duration
            src_deviation = LpVariable(f"flow_{flow_id}_src_deviation", lowBound=0, cat="Continuous")
            ts_deduction_problem += (flow_var["src_end"] - flow_var["src_start"]) - src_dur <= src_deviation
            ts_deduction_problem += src_dur - (flow_var["src_end"] - flow_var["src_start"]) <= src_deviation
            dur_deviation_vars.append(src_deviation)
            dst_deviation = LpVariable(f"flow_{flow_id}_dst_deviation", lowBound=0, cat="Continuous")
            ts_deduction_problem += (flow_var["dst_end"] - flow_var["dst_start"]) - dst_dur <= dst_deviation
            ts_deduction_problem += dst_dur - (flow_var["dst_end"] - flow_var["dst_start"]) <= dst_deviation
            dur_deviation_vars.append(dst_deviation)
            # LP Constraint:
            # flow duration should be at least 1us
            # NOTE: mainly for visualization purpose, but it's also unlikely in real life
            # for the duration to be less than 1us.
            ts_deduction_problem += flow_var["src_end"] - flow_var["src_start"] >= 1
            ts_deduction_problem += flow_var["dst_end"] - flow_var["dst_start"] >= 1
            # LP Constraint:
            # there should be a gap of 1us between the end and start of send/recv
            # NOTE: mainly for visualization purpose.
            ts_deduction_problem += flow_var["dst_start"] - flow_var["src_end"] >= 1

        # LP Objective: minimize the sum of all deviations
        ts_deduction_problem += sum(dur_deviation_vars), "Objective"

        # Solve the LP and error handling
        ts_deduction_problem.solve()
        if ts_deduction_problem.status == -1:
            raise ValueError("Failed to solve the timestamp deduction problem.")

        #
        # Recover the original `ts`` back from the shift at the start.
        #
        for flow_id, lp_var in lp_vars.items():
            data_flows[flow_id].associate_time(
                (value(lp_var["src_start"]) + min_time),
                (value(lp_var["src_end"]) + min_time),
                (value(lp_var["dst_start"]) + min_time),
                (value(lp_var["dst_end"]) + min_time))

        # # Print the final LP solution
        # for flow_id, lp_var in lp_vars.items():
        #     print(f"Flow {flow_id}:")
        #     print(f"  src_start: {value(lp_var['src_start']) + min_time}")
        #     print(f"  src_end: {value(lp_var['src_end']) + min_time}")
        #     print(f"  dst_start: {value(lp_var['dst_start']) + min_time}")
        #     print(f"  dst_end: {value(lp_var['dst_end']) + min_time}")