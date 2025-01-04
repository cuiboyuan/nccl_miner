import re
import json
from nccl_miner.flow_extractor import extract_flows_from_logs

def device_tid(device_id):
    return 100 + device_id

def data_flow_events(flow, ts_offset):
    '''
    Given a NcclDataFlow, create events that represent sending and receiving data in Chrome Trace.
    The events will start at ts_offset.
    '''
    events = []
    # Start
    events.append({
        "cat": "trace",
        'name': f"Send {flow.size} bytes",
        'ph':'B',
        'pid':2,
        'tid':device_tid(flow.src),
        'ts':ts_offset
    })
    events.append({
        "cat": "trace",
        'name': f"Recv {flow.size} bytes",
        'ph':'B',
        'pid':2,
        'tid':device_tid(flow.dst),
        'ts':ts_offset
    })
    end_ts = ts_offset + 10
    # End
    events.append({
        "cat": "trace",
        'name': f"Send {flow.size} bytes",
        'ph':'E',
        'pid':2,
        'tid':device_tid(flow.src),
        'ts':end_ts
    })
    events.append({
        "cat": "trace",
        'name': f"Recv {flow.size} bytes",
        'ph':'E',
        'pid':2,
        'tid':device_tid(flow.dst),
        'ts':end_ts
    })
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
    trace_events = []
    if len(dependencies) == 0:
        # Base case:
        # No flow dependencies that can be triggered, so just add all current flows to trace.
        for cur_flow in cur_data_flows:
            flow_events = data_flow_events(cur_flow, ts_offset)
            trace_events.extend(flow_events)
            # no deps to trigger
        # All flows here have the same duration, so just hard-code return ending ts.
        end_ts = ts_offset+10
        return trace_events, end_ts
    else:
        max_end_ts = 0
        # Recursive case
        for cur_flow in cur_data_flows:
            # First, add all current flows to trace, like in the base case.
            flow_events = data_flow_events(cur_flow, ts_offset)
            trace_events.extend(flow_events)

            # Now, we need to check whether current flows fulfilled some dependencies.
            new_data_flows = [] # Next set of flows whose deps are fulfilled
            new_dependencies = dependencies.copy() # make a copy to avoid python error
            for next_flow_id, deps in dependencies.items():
                next_flow = all_data_flow[next_flow_id]
                # check if any dep is fulfilled
                if cur_flow.id in deps:
                    new_dependencies[next_flow_id].remove(cur_flow.id)
                    # Add an arrow in the trace to represent dependency
                    trace_events.append({
                        "cat": "trace",
                        "name": "flow",
                        "id": cur_flow.id,
                        "ph": "s",
                        "ts": ts_offset+7,
                        "pid": 2,
                        "tid": device_tid(cur_flow.src)
                    })
                    trace_events.append({
                        "cat": "trace",
                        "name": "flow",
                        "id": cur_flow.id,
                        "ph": "f",
                        "bp": "e",
                        "ts": ts_offset+13,
                        "pid": 2,
                        "tid": device_tid(next_flow.src)
                    })

                # Check if all deps are fulfilled for next_flow
                if len(new_dependencies[next_flow_id]) == 0:
                    # Remove empty deps, and next_flow is ready to be added.
                    new_dependencies.pop(next_flow_id)
                    new_data_flows.append(next_flow)

            # Now, we have a new set of flows whose deps are all fulfilled
            if len(new_data_flows) > 0:
                # All cur_flows have the same duration, so just hard-code ending ts
                end_ts = ts_offset + 10
                # Recursively get flows triggered by new_data_flows and new_deps
                new_events, new_end_ts = gen_trace_events_from_flows(new_data_flows, new_dependencies, end_ts, all_data_flow)
                trace_events.extend(new_events)
                # The ending ts of this function is the end of the last event
                max_end_ts = new_end_ts if new_end_ts > max_end_ts else max_end_ts

        return trace_events, max_end_ts




def main(log_files):

    coll_events, coll_flows = extract_flows_from_logs(log_files)
    # print(coll_events)
    # print(coll_flows)

    ## for visualization
    events = []
    ts_offset = 0
    for idx, coll in enumerate(coll_events):
        # Start of a NCCL op
        events.append({
            "cat": "trace",
            'name': f"{coll.func}: {coll.data_num} {coll.data_type}",
            'ph':'B',
            'pid':1,
            'tid':1,
            'ts':ts_offset
        })

        data_flows, data_deps = coll_flows[idx]

        # First, find flows with zero-deps
        entry_flows = []
        for flow_id, flow in data_flows.items():
            if flow_id not in data_deps or len(data_deps[flow_id])==0:
                entry_flows.append(flow)
        # Get all flow events
        flow_events, end_ts = gen_trace_events_from_flows(entry_flows, data_deps, ts_offset, data_flows)
        events.extend(flow_events)
        
        # End of the NCCL op
        events.append({
            "cat": "trace",
            'name': f"{coll.func}: {coll.data_num} {coll.data_type}",
            'ph':'E',
            'pid':1,
            'tid':1,
            'ts':end_ts
        })
        ts_offset = end_ts
    
    chrome_trace = {
        "traceEvents": events
    }
    with open("dml_trace.json", "w") as f:
        json.dump(chrome_trace, f, indent=4)


if __name__ == "__main__":
    main(["example_nccl_logs/four_gpu_p2p_shm_disabled/nccl_logs.192-222-54-170.6559",
          "example_nccl_logs/four_gpu_p2p_shm_disabled/nccl_logs.192-222-54-170.6561",
          "example_nccl_logs/four_gpu_p2p_shm_disabled/nccl_logs.192-222-54-170.6562",
          "example_nccl_logs/four_gpu_p2p_shm_disabled/nccl_logs.192-222-54-170.6564"])

