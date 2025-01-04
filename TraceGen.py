import re
import json
from nccl_miner.flow_extractor import extract_flows_from_logs


def data_flow_events(flow, ts_offset):
    events = []
    # Start
    events.append({
        "cat": "trace",
        'name': f"Send {flow.size} bytes",
        'ph':'B',
        'pid':2,
        'tid':100+flow.src,
        'ts':ts_offset
    })
    events.append({
        "cat": "trace",
        'name': f"Recv {flow.size} bytes",
        'ph':'B',
        'pid':2,
        'tid':100+flow.dst,
        'ts':ts_offset
    })
    end_ts = ts_offset + 10
    # End
    events.append({
        "cat": "trace",
        'name': f"Send {flow.size} bytes",
        'ph':'E',
        'pid':2,
        'tid':100+flow.src,
        'ts':end_ts
    })
    events.append({
        "cat": "trace",
        'name': f"Recv {flow.size} bytes",
        'ph':'E',
        'pid':2,
        'tid':100+flow.dst,
        'ts':end_ts
    })
    return events


def gen_trace_events_from_flows(data_flows, dependencies, ts_offset, all_data_flow):
    timestamp = ts_offset
    events = []
    if len(dependencies) == 0:
        # base case
        for flow in data_flows:
            flow_events = data_flow_events(flow, ts_offset)
            events.extend(flow_events)
            # no deps to trigger
        return events, ts_offset+10
    else:
        max_end_ts = 0
        # recursive case
        for flow in data_flows:
            flow_events = data_flow_events(flow, ts_offset)
            events.extend(flow_events)
            # check dependencies
            new_data_flows = []
            new_dependencies = dependencies.copy()
            for next_flow_id, deps in dependencies.items():
                if flow.id in deps:
                    # update dependencies
                    new_dependencies[next_flow_id].remove(flow.id)
                if len(new_dependencies[next_flow_id]) == 0:
                    # trigger dependencies
                    new_dependencies.pop(next_flow_id)
                    next_flow = all_data_flow[next_flow_id]
                    new_data_flows.append(next_flow)
                    # add trace events for deps
                    events.append({
                        "cat": "trace",
                        "name": "flow",
                        "id": 200+flow.id,
                        "ph": "s",
                        "ts": ts_offset+7,
                        "pid": 2,
                        "tid": 100+flow.src
                    })
                    events.append({
                        "cat": "trace",
                        "name": "flow",
                        "id": 200+flow.id,
                        "ph": "f",
                        "bp": "e",
                        "ts": ts_offset+13,
                        "pid": 2,
                        "tid": 100+next_flow.src
                    })
            
            if len(new_data_flows) > 0:
                # trigger new data flow events
                new_events, new_end_ts = gen_trace_events_from_flows(new_data_flows, new_dependencies, ts_offset+10, all_data_flow)
                max_end_ts = new_end_ts if new_end_ts > max_end_ts else max_end_ts
                events.extend(new_events)
        return events, max_end_ts




def main(log_files):

    coll_events, coll_flows = extract_flows_from_logs(log_files)
    # print(coll_events)
    # print(coll_flows)

    ## for visualization
    events = []
    ts_offset = 0
    for idx, coll in enumerate(coll_events):
        # start of a NCCL op
        events.append({
            "cat": "trace",
            'name': f"{coll.func}: {coll.data_num} {coll.data_type}",
            'ph':'B',
            'pid':1,
            'tid':1,
            'ts':ts_offset
        })

        data_flows, data_deps = coll_flows[idx]

        # first, find flows without dep
        entry_flows = []
        for flow_id, flow in data_flows.items():
            if flow_id not in data_deps or len(data_deps[flow_id])==0:
                entry_flows.append(flow)
        # add events to trace
        flow_events, end_ts = gen_trace_events_from_flows(entry_flows, data_deps, ts_offset, data_flows)
        ts_offset = end_ts
        events.extend(flow_events)
        
        # end of the NCCL op
        events.append({
            "cat": "trace",
            'name': f"{coll.func}: {coll.data_num} {coll.data_type}",
            'ph':'E',
            'pid':1,
            'tid':1,
            'ts':ts_offset
        })
    
    chrome_trace = {
        "traceEvents": events
    }
    with open("dml_trace.json", "w") as f:
        json.dump(chrome_trace, f, indent=4)


if __name__ == "__main__":
    # log = "104-171-202-216:21232:21232 [1] NCCL INFO Broadcast: opCount 1 sendbuff 0x7582abbcf000 recvbuff 0x7582abbcf000 count 112 datatype 4 op 0 root 0 comm 0x57335165fc40 [nranks=2] stream 0x5733511d85d0"
    # ret = parse_coll_log(log)
    # print(ret)

    # topo_log = "104-171-202-216:21232:21301 [1] NCCL INFO Ring 00 : 0 -> 1 -> 0"
    # ret = parse_topo_ring_log(topo_log)
    # print(ret)
    main(["example_nccl_logs/four_gpu_p2p_shm_disabled/nccl_logs.192-222-54-170.6559",
          "example_nccl_logs/four_gpu_p2p_shm_disabled/nccl_logs.192-222-54-170.6561",
          "example_nccl_logs/four_gpu_p2p_shm_disabled/nccl_logs.192-222-54-170.6562",
          "example_nccl_logs/four_gpu_p2p_shm_disabled/nccl_logs.192-222-54-170.6564"])

