import json
from copy import deepcopy
from tqdm import tqdm

from .flow_extractor import *

DEVICE_OP_TID = 100
DATA_FLOW_TID = 200
global_arrow_id = 0

def flow_tid(src_id, dst_id):
    return 100*src_id + dst_id

def arrow_id(src_id, dst_id):
    return int(f"{src_id}{dst_id}")

def data_flow_events(flow, ts_offset):
    '''
    Given a NcclDataFlow, create events that represent sending and receiving data in Chrome Trace.
    The events will start at ts_offset.
    '''
    global global_arrow_id

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


def gen_trace_events_from_flows(cur_data_flows, dependencies, ts_offset, all_data_flow):
    '''
    cur_data_flows: 
        A list of NcclDataFlow to be added to the trace.
        Represents data flows whose dependencies are fulfilled or with zero dependency.
    dependencies: 
        Available flow dependencies that can be triggered by cur_data_flows
    ts_offset: 
        Assigned starting timestamp of all flows in cur_data_flows. Used to create traces.
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
        max_end_ts = ts_offset
        # No flow dependencies that can be triggered, so just add all current flows to trace.
        for cur_flow in cur_data_flows:
            flow_events = data_flow_events(cur_flow, ts_offset)
            trace_events.extend(flow_events)
            # Add flow duration to ts_offset
            end_ts = ts_offset + cur_flow.duration
            # no deps to trigger
            # ...
            if end_ts > max_end_ts:
                max_end_ts = end_ts
        return trace_events, max_end_ts
    else:
        max_end_ts = ts_offset
        # Recursive case
        for cur_flow in cur_data_flows:
            # First, add all current flows to trace, like in the base case.
            flow_events = data_flow_events(cur_flow, ts_offset)
            trace_events.extend(flow_events)
            # Add flow duration to ts_offset
            end_ts = ts_offset + cur_flow.duration

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
                        "ts": end_ts-1,
                        "pid": cur_flow.dst,
                        "tid": DATA_FLOW_TID+cur_flow.src
                    })
                    trace_events.append({
                        "ph": "f",
                        "cat": "flow_dependency",
                        "id": global_arrow_id,
                        "ts": end_ts,
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
                new_events, end_ts = gen_trace_events_from_flows(new_data_flows, new_dependencies, end_ts, all_data_flow)
                trace_events.extend(new_events)
                # The ending ts of this function is the end of the last event
            if end_ts > max_end_ts:
                max_end_ts = end_ts

        return trace_events, max_end_ts


def generate_chrome_trace(log_files, out_json):

    print("Extracting flows from the logs...")
    comm_events = extract_flows_from_logs(log_files)
    print("Extracted.")
    # print(comm_events)
    # print(coll_flows)

    ## for visualization
    events = []
    print("Generating trace visualizations...")
    ts_offset = 0
    for idx, coll in tqdm(enumerate(comm_events)):
        data_flows = coll.data_flows
        data_deps = coll.dependencies
        # First, find flows with zero-deps
        entry_flows = []
        for flow_id, flow in data_flows.items():
            if flow_id not in data_deps or len(data_deps[flow_id])==0:
                entry_flows.append(flow)
        # Get all flow events
        flow_events, end_ts = gen_trace_events_from_flows(entry_flows, data_deps, ts_offset, data_flows)
        if len(flow_events) > 0:
            events.extend(flow_events)
        else:
            end_ts += 10
        # End of the NCCL op for each participating device
        for device in coll.devices:
            events.append({
                "ph": "X",
                "cat": "coll_op",
                'name': f"{coll.name} (device: {coll.devices})",
                'pid': device,
                'tid': DEVICE_OP_TID,
                'ts': ts_offset,
                'dur': end_ts - ts_offset,
                'args': {
                    "data_type": str(coll.data_type),
                    "num": coll.data_num,
                }
            })
        ts_offset = end_ts+2

    chrome_trace = {
        "traceEvents": events
    }
    print("Generated.")
    print(f"Writing to file {out_json}...")
    with open(out_json, "w") as f:
        json.dump(chrome_trace, f, indent=4)
    print("Done.")

