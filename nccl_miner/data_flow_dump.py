import json

from nccl_miner.mining.data_flow import DataFlow
from nccl_miner.common.nccl_function_group import NcclFunctionGroup

def dump_data_flows(data_flow_groups, out_json):
    """
    Dumps data flows and their dependencies into a JSON file.

    Args:
        data_flow_groups (dict): A dictionary where keys are group IDs and values are flow group objects.
                                 Each flow group object should have `data_flows` and `dependencies` attributes.
        out_json (str): The path to the output JSON file.

    The output JSON file will all the data flows.
    Key is the flow id, value is a dictionary containing:
        - src: The source device of the data flow.
        - dst: The destination device of the data flow.
        - start_time: The start timestamp of the data flow.
        - end_time: The end timestamp of the data flow.
        - bytes: The size of the data flow in bytes.
        - content: The content of the data flow (e.g., function name).
        - dependencies: A list of flow IDs that this flow depends on.
    """
    all_data_flows = {}

    for group_id, flow_group in data_flow_groups.items():
        assert isinstance(flow_group, NcclFunctionGroup)
        data_flows = flow_group.data_flows
        dependencies = flow_group.dependencies

        for flow_id, flow in data_flows.items():
            assert isinstance(flow, DataFlow)
            # find out the content of the flow
            if flow.data_name is None:
                flow_content = f"{flow_group.func}"
            else:
                flow_content = f"{flow_group.func}: {flow.data_name}"
            # find out the dependencies of the flow
            dep = []
            if flow_id in dependencies:
                dep = dependencies[flow_id]
            # Create a new dictionary for each flow to avoid reference issues
            all_data_flows[flow_id] = {
                "src": flow.src,
                "dst": flow.dst,
                "start_time": flow.flow_start_time,
                "end_time": flow.flow_end_time,
                "bytes": flow.size,
                "content": flow_content,
                "dependencies": dep,
            }

    with open(out_json, 'w') as f:
        json.dump(all_data_flows, f, indent=4)
    print(f"Data flows and dependencies dumped to {out_json}")
