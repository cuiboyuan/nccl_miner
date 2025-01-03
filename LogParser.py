import re
import json

class NcclDataType:
    def __init__(self, dtype_id):
        dtype = int(dtype_id)
        if dtype == 0:
            self.name = "ncclInt8"
            self.type = "int"
            self.num_bits = 8
            self.bytes = 1
        elif dtype == 1:
            self.name = "ncclUint8"
            self.type = "int"
            self.num_bits = 8
            self.bytes = 1
        elif dtype == 2:
            self.name = "ncclInt32"
            self.type = "int"
            self.num_bits = 32
            self.bytes = 4
        elif dtype == 3:
            self.name = "ncclUint32"
            self.type = "int"
            self.num_bits = 32
            self.bytes = 4
        elif dtype == 4:
            self.name = "ncclInt64"
            self.type = "int"
            self.num_bits = 64
            self.bytes = 8
        elif dtype == 5:
            self.name = "ncclUint64"
            self.type = "int"
            self.num_bits = 64
            self.bytes = 8
        elif dtype == 6:
            self.name = "ncclFloat16"
            self.type = "float"
            self.num_bits = 16
            self.bytes = 2
        elif dtype == 7:
            self.name = "ncclFloat32"
            self.type = "float"
            self.num_bits = 32
            self.bytes = 4
        elif dtype == 8:
            self.name = "ncclFloat64"
            self.type = "float"
            self.num_bits = 64
            self.bytes = 8
        elif dtype == 9:
            self.name = "ncclBfloat16"
            self.type = "float"
            self.num_bits = 16
            self.bytes = 2
        else:
            raise ValueError(f"Invalid NcclDataType: {dtype}")
    
    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name


class CollectiveOperation:
    def __init__(self, coll_info):
        # self.host = coll_info['host_name']
        # self.pid = int(coll_info['pid'])
        # self.tid = int(coll_info['tid'])

        # self.device = int(coll_info['cuda_device'])
        
        self.func = coll_info['coll_op']
        
        self.data_num = int(coll_info['num_elem'])
        self.data_type = NcclDataType(coll_info['data_type'])
        self.data_size = self.data_type.bytes * self.data_num

        self.root = int(coll_info['root_device'])

        # self.src_buf = coll_info['src_buf_addr']
        # self.dst_buf = coll_info['dst_buf_addr']
    
    def __repr__(self):
        return f"{self.func}: {self.data_size} bytes ({self.data_num} {self.data_type})"


class DataFlow:
    def __init__(self, src, dst, size):
        self.src = src
        self.dst = dst
        self.size = size

class NcclRing:
    def __init__(self, all_topo_info):
        incomplete_rings = {}
        for topo in all_topo_info:
            ring_id = topo['ring_id']
            prev = topo['prev']
            cur = topo['cur']
            next = topo['next']
            if ring_id not in incomplete_rings:
                incomplete_rings[ring_id] = {cur: next}
            else:
                incomplete_rings[ring_id][cur] = next

        self.rings = {}
        for ring_id, tmp_ring in incomplete_rings.items():
            entry_node = None
            for n in tmp_ring:
                entry_node = n
                break

            ring = []
            cur_node = None
            while cur_node != entry_node:
                if cur_node is None:
                    cur_node = entry_node
                cur_node = tmp_ring[cur_node]
                ring.append(cur_node)

            self.rings[ring_id] = ring
        
        self.size = None
        for id, ring in self.rings.items():
            if self.size is None:
                self.size = len(ring)
            else:
                assert self.size == len(ring)
    
    def __repr__(self):
        ret = ""
        for id, ring in self.rings.items():
            ring_str = "->".join(ring)
            ret += f"{id}: {ring_str}\n"
        return ret

    def collective_communication(self, coll_op):
        data_flows = []
        dependencies = []
        if coll_op.func == "Broadcast":
            # assume single rail/channel
            for _, ring in self.rings:
                root_idx = ring.index(coll_op.root)
                cur_idx = root_idx
                if cur_idx >= self.size-1:
                    next_idx = 0
                else:
                    next_idx = cur_idx+1

                while next_idx != root_idx:
                    cur_node = ring[cur_idx]
                    next_node = ring[next_idx]
                    data_flows.append((cur_node, next_node, coll_op.data_size))

                    cur_idx = next_idx
                    if cur_idx >= self.size-1:
                        next_idx = 0
                    else:
                        next_idx = cur_idx+1

            
        elif coll_op.func == "AllReduce":
            pass




def parse_log(log_line):
    log_patterns = {
        'ring_topo': (
            r"(?P<host_name>[^:]+):"                     # Host name
            r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
            r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
            r"NCCL INFO\s+"
            r"Ring (?P<ring_id>\d+) : (?P<prev>\d+) -> (?P<cur>\d+) -> (?P<next>\d+)"
        ),
        'coll_op': (
            r"(?P<host_name>[^:]+):"                     # Host name
            r"(?P<pid>\d+):(?P<tid>\d+)\s+"             # PID and TID
            r"\[(?P<cuda_device>[^\]]*)\]\s+"           # CUDA device
            r"NCCL INFO\s+"
            r"(?P<coll_op>\w+):\s+"         # Collective operation
            r"opCount\s+(?P<op_cnt>\d+)\s+"             # Operation count
            r"sendbuff\s+(?P<src_buf_addr>0x[0-9a-f]+)\s+"  # Source buffer address
            r"recvbuff\s+(?P<dst_buf_addr>0x[0-9a-f]+)\s+"  # Destination buffer address
            r"count\s+(?P<num_elem>\d+)\s+"             # Number of elements
            r"datatype\s+(?P<data_type>\w+)\s+"         # Data type
            r"op\s+(?P<operator>\w+)\s+"                # Operator
            r"root\s+(?P<root_device>\d+)\s+"           # Root device
            r"comm\s+(?P<comm_obj_ptr>0x[0-9a-f]+)\s+"  # Comm object pointer
            r"\[nranks=(?P<nranks>\d+)\]\s+"       # Total ranks
            r"stream\s+(?P<stream_obj_ptr>0x[0-9a-f]+)" # Stream object pointer
        ),
    }

    for name, pattern in log_patterns.items():
        # Match the pattern with the log line
        match = re.match(pattern, log_line)
        if match:
            # Return the extracted information as a dictionary
            raw_info = match.groupdict()
            return name, raw_info
    return None, None


def main(log_files):
    ring_topos = []
    tree_topos = []
    coll_comms = []
    root_log = log_files[0]
    with open(root_log, "r") as f:
        for log_line in f.readlines():
            name, info = parse_log(log_line)

            if name == "coll_op":
                coll_op = CollectiveOperation(info)
                coll_comms.append(coll_op)
            elif name == "ring_topo":
                ring_topos.append(info)
            elif name == "tree_topo":
                tree_topos.append(info)

    for log in log_files[1:]:
        with open(log, "r") as f:
            for log_line in f.readlines():
                name, info = parse_log(log_line)
                if name == "ring_topo":
                    ring_topos.append(info)
                elif name == "tree_topo":
                    tree_topos.append(info)

    nccl_ring = NcclRing(ring_topos)
    print(nccl_ring)

    # for op in coll_comms:
    #     print(op)

    ## for visualization
    events = []
    ts_offset = 0
    for coll in coll_comms:
        # start of a NCCL op
        events.append({
            "cat": "trace",
            'name': coll.func,
            'ph':'B',
            'pid':1,
            'tid':1,
            'ts':ts_offset
        })
        
        # assume single rail/channel ring
        for rank in range(nccl_ring.size):
            events.append({
                "cat": "trace",
                'name': f"{coll.data_num} {coll.data_type}",
                'ph':'B',
                'pid':2,
                'tid':rank,
                'ts':ts_offset
            })
            ts_offset += 10
            events.append({
                "cat": "trace",
                'name': f"{coll.data_num} {coll.data_type}",
                'ph':'E',
                'pid':2,
                'tid':rank,
                'ts':ts_offset
            })

        # end of the NCCL op
        events.append({
            "cat": "trace",
            'name': coll.func,
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
    main(["two_gpu_nccl_logs/nccl_logs.104-171-202-216.21230","two_gpu_nccl_logs/nccl_logs.104-171-202-216.21232"])

