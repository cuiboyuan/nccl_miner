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

            # Constraint: 
            # duration of send/recv in a data flow is greater than 1us
            # NOTE: mainly for visualization purpose.
            ts_deduction_problem += lp_vars[flow_id]["src_end"] - lp_vars[flow_id]["src_start"] >= 1
            ts_deduction_problem += lp_vars[flow_id]["dst_end"] - lp_vars[flow_id]["dst_start"] >= 1
            # there should be a gap of 1us between the end and start of send/recv
            ts_deduction_problem += lp_vars[flow_id]["dst_start"] - lp_vars[flow_id]["src_end"] >= 1
        
        for cur_flow_id, dep_flow_ids in dependencies.items():
            for dep_flow_id in dep_flow_ids:
                # Constraint: a flow only start after its dependent flow ends
                ts_deduction_problem += lp_vars[cur_flow_id]["src_start"] - lp_vars[dep_flow_id]["dst_end"] >= 1
        
        # TODO: try to find a reasonable objective

        ts_deduction_problem.solve()

        if ts_deduction_problem.status == -1:
            raise ValueError("Failed to solve the timestamp deduction problem.")

        # Print the final solution
        for flow_id, lp_var in lp_vars.items():
            print(f"Flow {flow_id}:")
            print(f"  src_start: {value(lp_var['src_start']) + min_time}")
            print(f"  src_end: {value(lp_var['src_end']) + min_time}")
            print(f"  dst_start: {value(lp_var['dst_start']) + min_time}")
            print(f"  dst_end: {value(lp_var['dst_end']) + min_time}")

        for flow_id, lp_var in lp_vars.items():
            data_flows[flow_id].associate_time(
                (value(lp_var["src_start"]) + min_time),
                (value(lp_var["src_end"]) + min_time),
                (value(lp_var["dst_start"]) + min_time),
                (value(lp_var["dst_end"]) + min_time))
