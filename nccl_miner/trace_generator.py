'''
Generate Perfetto JSON traces to visualize data flows.
'''
import json

from .common.nccl_function import NcclPtpFunction, NcclCollectiveFunction
from .common.torch_event import CudaLocal

CPU_OP_TID = 0
GPU_OP_TID = 100
DATA_FLOW_TID = 200
global_arrow_id = 0


def gen_events_from_data_flows(data_flows):
    '''
    Generate events for multiple data flows.
    Assign TID dynamically to avoid overlapping events on the same thread.
    Keep a memory of each flow event's TID for reuse in dependency function.
    '''
    global global_arrow_id

    events = []
    assigned_tids = {}
    flow_tid_map = {}

    def get_available_tid(pid, start_time, end_time):
        '''
        Find an available TID for the given PID and time range.
        Log the PID, start/end time, and the assigned TID.
        '''
        if pid not in assigned_tids:
            assigned_tids[pid] = []

        for tid, time_ranges in enumerate(assigned_tids[pid]):
            if all(end_time <= existing_start or \
                    start_time >= existing_end for existing_start, existing_end in time_ranges):
                # No overlap with any existing range, reuse this TID
                assigned_tids[pid][tid].append((start_time, end_time))
                print(f"PID: {pid}, Start: {start_time}, End: {end_time}, Assigned TID: {tid} (reused)")
                return tid

        # No reusable TID found, assign a new one
        new_tid = len(assigned_tids[pid])
        assigned_tids[pid].append([(start_time, end_time)])
        print(f"PID: {pid}, Start: {start_time}, End: {end_time}, Assigned TID: {new_tid} (new)")
        return new_tid

    for flow_id, flow in data_flows.items():
        print(flow)
        # Sending event
        send_tid = get_available_tid(flow.src, flow.src_start_time, flow.src_start_time + flow.src_duration)
        events.append({
            'ph': 'X',
            "cat": "data_flow",
            'name': f"Send to {flow.dst}",
            'pid': flow.src,
            'tid': DATA_FLOW_TID + send_tid,
            'ts': flow.src_start_time,
            'dur': flow.src_duration,
            'args': {
                "bytes": flow.size,
                "name": flow.data_name
            },
            'id': flow_id
        })

        # Receiving event
        recv_tid = get_available_tid(flow.dst, flow.dst_start_time, flow.dst_start_time + flow.dst_duration)
        events.append({
            'ph': 'X',
            "cat": "data_flow",
            'name': f"Recv from {flow.src}",
            'pid': flow.dst,
            'tid': DATA_FLOW_TID + recv_tid,
            'ts': flow.dst_start_time,
            'dur': flow.dst_duration,
            'args': {
                "bytes": flow.size,
                "name": flow.data_name
            },
            'id': flow_id
        })

        # Store TIDs for reuse in dependency function
        flow_tid_map[flow_id] = {
            "send_tid": DATA_FLOW_TID + send_tid,
            "recv_tid": DATA_FLOW_TID + recv_tid
        }

        # Flow start
        events.append({
            'ph': 's',
            "cat": "data_flow",
            'id': global_arrow_id,
            'pid': flow.src,
            'tid': DATA_FLOW_TID + send_tid,
            'ts': flow.src_end_time,
            'bind_id': flow_id
        })
        # Flow end
        events.append({
            'ph': 'f',
            "cat": "data_flow",
            'id': global_arrow_id,
            'pid': flow.dst,
            'tid': DATA_FLOW_TID + recv_tid,
            'ts': flow.dst_start_time,
            'bind_id': flow_id,
            'bp': 'e'
        })
        global_arrow_id += 1

    return events, flow_tid_map


def gen_arrows_from_dependencies(dependencies, data_flows, flow_tid_map):
    '''
    Generate arrows representing dependencies between data flows.
    Reuse TIDs from flow_tid_map.
    '''
    global global_arrow_id

    arrows = []
    for flow_id, deps in dependencies.items():
        current_flow = data_flows[flow_id]
        current_tids = flow_tid_map[flow_id]
        for dep_id in deps:
            dep_flow = data_flows[dep_id]
            dep_tids = flow_tid_map[dep_id]

            # Start arrow for dependency
            arrows.append({
                'ph': 's',
                "cat": "data_flow",
                'id': global_arrow_id,
                'pid': dep_flow.dst,
                'tid': dep_tids["recv_tid"],
                'ts': dep_flow.dst_end_time,
                'bind_id': dep_flow.id
            })
            # End arrow for dependency
            arrows.append({
                'ph': 'f',
                "cat": "data_flow",
                'id': global_arrow_id,
                'pid': current_flow.src,
                'tid': current_tids["send_tid"],
                'ts': current_flow.src_start_time,
                'bind_id': current_flow.id,
                'bp': 'e'
            })
            global_arrow_id += 1

    return arrows

def generate_perfetto_trace(cpu_ops, gpu_ops, data_flow_groups, out_json):
    events = []

    all_data_flows = {}
    all_dependencies = {}
    for group_id, flow_group in data_flow_groups.items():
        data_flows = flow_group.data_flows
        dependencies = flow_group.dependencies

        all_data_flows.update(data_flows)
        all_dependencies.update(dependencies)

    # Generate events from data flows
    flow_events, flow_tid_map = gen_events_from_data_flows(all_data_flows)
    events.extend(flow_events)

    # Generate arrows from dependencies
    dependency_arrows = gen_arrows_from_dependencies(all_dependencies, all_data_flows, flow_tid_map)
    events.extend(dependency_arrows)

    for device, all_gpu_op in gpu_ops.items():
        for gpu_op in all_gpu_op:
            assert isinstance(gpu_op, NcclPtpFunction) or isinstance(gpu_op, NcclCollectiveFunction)

            flow_group = data_flow_groups[gpu_op.group_id]
            events.append({
                "ph": "X",
                "cat": "coll_op",
                'name': f"nccl:{gpu_op.func} ({gpu_op.group_id})",
                'pid': device,
                'tid': GPU_OP_TID,
                'ts': gpu_op.start_time,
                'dur': gpu_op.end_time - gpu_op.start_time,
                'args': {
                    "data_type": str(gpu_op.data_type),
                    "data_num": gpu_op.data_num,
                    "bytes": gpu_op.data_size,
                }
            })

    for device, all_cpu_op in cpu_ops.items():
        for cpu_op in all_cpu_op:
            assert isinstance(cpu_op, CudaLocal)

            events.append({
                "ph": "X",
                "cat": "local_op",
                'name': f"{cpu_op.name}",
                'pid': device,
                'tid': CPU_OP_TID,
                'ts': cpu_op.start_time,
                'dur': cpu_op.end_time - cpu_op.start_time,
                'args': {
                    "semantics": cpu_op.semantics.name
                }
            })

    # Write the events to the output JSON file
    perfetto_trace = {
        "traceEvents": events
    }
    with open(out_json, "w") as f:
        json.dump(perfetto_trace, f, indent=4)
    print(f"Perfetto trace generated with {len(events)} events.")
    print(f"Output written to {out_json}")