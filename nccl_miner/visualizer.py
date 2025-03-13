'''
Generate Perfetto JSON traces to visualize data flows.
'''
import json
from copy import deepcopy
from tqdm import tqdm

from .parsing.parser_pipeline import *
from .miner_pipeline import mine_torch_nccl_pipeline
from .common.nccl_function import NcclPtpFunction, NcclCollectiveFunction

CPU_OP_TID = 0
GPU_OP_TID = 100
DATA_FLOW_TID = 200
global_arrow_id = 0

def flow_tid(src_id, dst_id):
    return 100*src_id + dst_id

def arrow_id(src_id, dst_id):
    return int(f"{src_id}{dst_id}")

def data_flow_events(flow):
    '''
    Given a NcclDataFlow, create events that represent sending and receiving data in Chrome Trace.
    '''
    global global_arrow_id

    ts_offset = flow.start_time
    events = []
    # Start
    events.append({
        'ph':'X',
        "cat": "data_flow",
        'name': f"Send to {flow.dst}",
        'pid':flow.src,
        'tid':DATA_FLOW_TID+flow.dst,
        'ts':ts_offset,
        'dur':flow.duration/2-1,
        'args': {
            "bytes": flow.size,
            "name": flow.data_name
        },
        'id': flow.id
    })
    # End
    events.append({
        'ph':'X',
        "cat": "data_flow",
        'name': f"Recv from {flow.src}",
        'pid':flow.dst,
        'tid':DATA_FLOW_TID+flow.src,
        'ts':ts_offset+flow.duration/2,
        'dur':flow.duration/2-1,
        'args': {
            "bytes": flow.size,
            "name": flow.data_name
        },
        'id': flow.id
    })

    # flow start
    events.append({
        'ph':'s',
        "cat": "communication",
        'id': global_arrow_id,
        'pid':flow.src,
        'tid':DATA_FLOW_TID+flow.dst,
        'ts':ts_offset+flow.duration/2-1,
    })
    # flow end
    events.append({
        'ph':'f',
        "cat": "communication",
        'id': global_arrow_id,
        'pid':flow.dst,
        'tid':DATA_FLOW_TID+flow.src,
        'ts':ts_offset+flow.duration/2,
        'bp':'e'
    })
    global_arrow_id += 1
    return events


def gen_trace_events_from_flows(cur_data_flows, dependencies, all_data_flow):
    '''
    cur_data_flows: 
        A list of NcclDataFlow to be added to the trace.
        Represents data flows whose dependencies are fulfilled or with zero dependency.
    dependencies: 
        Available flow dependencies that can be triggered by cur_data_flows
    all_data_flow: 
        All data flows globally, regardless of whether it is already added to the trace.
        Used to find the next set of data flows that are triggered by cur_data_flows through dependencies.

    Return: a tuple of trace_events (List of Dict) and an ending timestamp (int).
        trace_events contain all flows in cur_data_flows, as well as all flows triggered through dependencies.
    '''
    global global_arrow_id

    trace_events = []
    if len(dependencies) == 0:
        # Base case:
        # No flow dependencies that can be triggered, so just add all current flows to trace.
        for cur_flow in cur_data_flows:
            flow_events = data_flow_events(cur_flow)
            trace_events.extend(flow_events)
            # no deps to trigger
            # ...
        return trace_events
    else:
        # Recursive case
        for cur_flow in cur_data_flows:
            # First, add all current flows to trace, like in the base case.
            flow_events = data_flow_events(cur_flow)
            trace_events.extend(flow_events)

            # Now, we need to check whether current flows fulfilled some dependencies.
            new_data_flows = [] # Next set of flows whose deps are fulfilled
            new_dependencies = deepcopy(dependencies) # make a copy to avoid python error
            for next_flow_id, deps in dependencies.items():
                next_flow = all_data_flow[next_flow_id]
                # check if any dep is fulfilled
                if cur_flow.id in deps:
                    new_dependencies[next_flow_id].remove(cur_flow.id)
                    # Add an arrow in the trace to represent dependency
                    trace_events.append({
                        "ph": "s",
                        "cat": "flow_dependency",
                        "id": global_arrow_id,
                        "ts": cur_flow.end_time-1,
                        "pid": cur_flow.dst,
                        "tid": DATA_FLOW_TID+cur_flow.src
                    })
                    trace_events.append({
                        "ph": "f",
                        "cat": "flow_dependency",
                        "id": global_arrow_id,
                        "ts": next_flow.start_time,
                        "pid": next_flow.src,
                        "tid": DATA_FLOW_TID+next_flow.dst,
                        'bp':'e'
                    })
                    global_arrow_id += 1

                # Check if all deps are fulfilled for next_flow
                if len(new_dependencies[next_flow_id]) == 0:
                    # Remove empty deps, and next_flow is ready to be added.
                    new_dependencies.pop(next_flow_id)
                    new_data_flows.append(next_flow)

            # Now, we have a new set of flows whose deps are all fulfilled
            if len(new_data_flows) > 0:
                # Recursively get flows triggered by new_data_flows and new_deps
                new_events = gen_trace_events_from_flows(new_data_flows, new_dependencies, all_data_flow)
                trace_events.extend(new_events)

        return trace_events


def generate_chrome_trace(nccl_log_files, torch_prof_files, out_json):

    print("Extracting flows from the logs...")
    cpu_ops, gpu_ops, data_flow_groups = mine_torch_nccl_pipeline(nccl_log_files, torch_prof_files)
    print("Extracted.")

    ## for visualization
    events = []
    print("Generating trace visualizations...")
    flow_group_ts = {}
    for group_id, flow_group in data_flow_groups.items():
        data_flows = flow_group.data_flows
        data_deps = flow_group.dependencies
        # First, find flows with zero-deps
        entry_flows = []
        for flow_id, flow in data_flows.items():
            if flow_id not in data_deps or len(data_deps[flow_id])==0:
                entry_flows.append(flow)
        # Get all flow events
        flow_events = gen_trace_events_from_flows(entry_flows, data_deps, data_flows)
        if len(flow_events) > 0:
            events.extend(flow_events)
        # Find the start and end timestamp of the flow group
        ts_offset = None
        end_ts = None
        for flow_id, flow in data_flows.items():
            if ts_offset is None or flow.start_time < ts_offset:
                ts_offset = flow.start_time
            if end_ts is None or flow.end_time > end_ts:
                end_ts = flow.end_time
        flow_group_ts[group_id] = (ts_offset, end_ts)
        
    # End of the NCCL op for each participating device
    for device, all_gpu_op in gpu_ops.items():
        for gpu_op in all_gpu_op:
            assert isinstance(gpu_op, NcclPtpFunction) or isinstance(gpu_op, NcclCollectiveFunction)

            flow_group = data_flow_groups[gpu_op.group_id]
            group_start, group_end = flow_group_ts[gpu_op.group_id]
            events.append({
                "ph": "X",
                "cat": "coll_op",
                'name': f"{gpu_op.func} (devices: {flow_group.all_devices})",
                'pid': device,
                'tid': GPU_OP_TID,
                'ts': group_start,
                'dur': group_end - group_start,
                'args': {
                    "data_type": str(gpu_op.data_type),
                    "num": gpu_op.data_num,
                }
            })

    chrome_trace = {
        "traceEvents": events
    }
    print("Generated.")
    print(f"Writing to file {out_json}...")
    with open(out_json, "w") as f:
        json.dump(chrome_trace, f, indent=4)
    print("Done.")

