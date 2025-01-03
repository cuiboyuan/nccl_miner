'''
Reflect the meaning of values in the NCCL debugging logs. This information comes from the NCCL database.
'''

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


class NcclCollective:
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


class NcclDataFlow:
    def __init__(self, src, dst, size, id=None):
        self.src = src
        self.dst = dst
        self.size = size
        self.id = id


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

    def probe_coll_op(self, coll_op):
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
                    data_flows.append(NcclDataFlow(cur_node, next_node, coll_op.data_size))

                    cur_idx = next_idx
                    if cur_idx >= self.size-1:
                        next_idx = 0
                    else:
                        next_idx = cur_idx+1

            
        elif coll_op.func == "AllReduce":
            pass
