import re
import json
from nccl_miner.flow_extractor import extract_flows_from_logs


def main(log_files):

    coll_events, coll_flows = extract_flows_from_logs(log_files)
    print(coll_events)
    print(coll_flows)

    ## for visualization
    events = []
    ts_offset = 0
    for coll in coll_events:
        # start of a NCCL op
        events.append({
            "cat": "trace",
            'name': f"{coll.func}: {coll.data_num} {coll.data_type}",
            'ph':'B',
            'pid':1,
            'tid':1,
            'ts':ts_offset
        })
        
        # for flows, deps in coll_flows:
        #     # data at
        #     events.append({
        #         "cat": "trace",
        #         'name': f"Sending {flow.data_num} bytes",
        #         'ph':'B',
        #         'pid':2,
        #         'tid':flow.src,
        #         'ts':ts_offset
        #     })
        #     events.append({
        #         "cat": "trace",
        #         'name': f"Receiving {coll.data_num} {coll.data_type}",
        #         'ph':'B',
        #         'pid':2,
        #         'tid':flow.dst,
        #         'ts':ts_offset
        #     })
        ts_offset += 10
        #     events.append({
        #         "cat": "trace",
        #         'name': f"Sending {coll.data_num} {coll.data_type}",
        #         'ph':'E',
        #         'pid':2,
        #         'tid':flows.src,
        #         'ts':ts_offset
        #     })
        #     events.append({
        #         "cat": "trace",
        #         'name': f"Receiving {coll.data_num} {coll.data_type}",
        #         'ph':'E',
        #         'pid':2,
        #         'tid':flows.dst,
        #         'ts':ts_offset
        #     })

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
    main(["example_nccl_logs/two_gpu/nccl_logs.104-171-202-216.21230",
          "example_nccl_logs/two_gpu/nccl_logs.104-171-202-216.21232"])

