'''
Reflect the meaning of values in the NCCL debugging logs. This information comes from the NCCL database.
'''
from .nccl_utils import *

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
    # counter used to generate flow id.
    global_flow_counter = 0

    def __init__(self, src, dst, size, name=None):
        self.src = src
        self.dst = dst
        self.size = size
        self.data_name = name
        self.duration = 10
        self.id = NcclDataFlow.global_flow_counter
        NcclDataFlow.global_flow_counter += 1

    def __repr__(self):
        return f"Flow {self.id}:{self.src}->{self.dst}[{self.size} bytes]"


class NcclRing:
    def __init__(self, ring_id):
        self.complete = False
        self.id = ring_id
        self.nodes = []
        self.incomplete_ring = {}
    
    def add_node(self, cur, next):
        if self.complete:
            return
        self.incomplete_ring[int(cur)] = int(next)
    
    def finalize_ring(self):
        # Find a random starting point
        entry_node = None
        for n in self.incomplete_ring:
            entry_node = n
            break
        # Sort the nodes in order to reflect ring structure
        cur_node = None
        while cur_node != entry_node:
            if cur_node is None:
                cur_node = entry_node
            self.nodes.append(cur_node)
            cur_node = self.incomplete_ring[cur_node]
        self.size = len(self.nodes)
        self.complete = True

    def __len__(self):
        return len(self.nodes)
    
    def get_next_node(self, cur_node):
        cur_idx = self.nodes.index(cur_node)
        if cur_idx >= self.size-1:
            next_idx = 0
        else:
            next_idx = cur_idx+1
        return self.nodes[next_idx]
    
    def __getitem__(self, index):
        return self.nodes[index]
    
    def __repr__(self):
        ring_str = "->".join(self.nodes)
        return ring_str


class NcclAlgoRing:
    def __init__(self, all_topo_info):
        incomplete_rings = {}
        for topo in all_topo_info:
            ring_id = '00'
            prev = topo['prev']
            cur = topo['cur']
            next = topo['next']
            if ring_id not in incomplete_rings:
                incomplete_rings[ring_id] = NcclRing(ring_id)
            incomplete_rings[ring_id].add_node(cur, next)

        self.rings = {}
        for ring_id, ring in incomplete_rings.items():
            ring.finalize_ring()
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
            ret += f"{id}: {ring}\n"
        return ret

    def probe_coll_op(self, coll_op):
        data_flows = {}
        dependencies = {}

        if coll_op.func == "Broadcast":
            # Based on NCCL implementation in src/device/broadcast.h:runRing()
            # TODO: Naive understanding for now, need more details
            for _, ring in self.rings.items():
                cur_node = coll_op.root
                next_node = ring.get_next_node(cur_node)

                prev_flow_id = None
                while next_node != coll_op.root:
                    # send data to next GPU
                    cur_flow = NcclDataFlow(cur_node, next_node, coll_op.data_size, name=f"{coll_op.root}'s Data")
                    cur_flow_id = cur_flow.id
                    data_flows[cur_flow_id] = cur_flow
                    # add dependencies
                    # flow from the root is the first flow, it has zero deps
                    if cur_node != coll_op.root:
                        if prev_flow_id is None:
                            # should not enter here.
                            assert False
                        # need to wait for the previous flow to finish.
                        if cur_flow_id not in dependencies:
                            dependencies[cur_flow_id] = [prev_flow_id]
                        else:
                            dependencies[cur_flow_id].append(prev_flow_id)
                    # done cur data flow, move on to the next
                    prev_flow_id = cur_flow_id

                    cur_node = next_node
                    next_node = ring.get_next_node(cur_node)

            
        elif coll_op.func == "AllReduce":
            # Based on NCCL implementation in src/device/all_reduce.h:runRing()
            # TODO: Naive understanding, need more reading on NCCL codes
            nranks = self.size
            for _, ring in self.rings.items():
                for idx, rank in enumerate(ring.nodes):
                    # 1. reduce and copy to next GPU (nranks-1 steps)
                    # my understanding: cur_node reduce on its own node, and transfer whatever it receives to the next GPU as is
                    # push data to next GPU
                    prev_flow_id = None
                    j = 0
                    cur_node = rank
                    next_node = ring.get_next_node(cur_node)
                    while j < (nranks-1):
                        cur_flow = NcclDataFlow(cur_node, next_node, coll_op.data_size, name=f"{rank}'s Data")
                        cur_flow_id = cur_flow.id
                        data_flows[cur_flow_id] = cur_flow

                        if prev_flow_id is None:
                            # first flow, zero dependency
                            pass
                        else:
                            # need to wait for the previous flow to finish.
                            if cur_flow_id not in dependencies:
                                dependencies[cur_flow_id] = [prev_flow_id]
                            else:
                                dependencies[cur_flow_id].append(prev_flow_id)                        
                        prev_flow_id = cur_flow_id

                        cur_node = next_node
                        next_node = ring.get_next_node(cur_node)
                        j += 1
                    # 2. copy to next GPU (nranks-1 steps)
                    # my understanding: now, broadcast the final reduced result to all nodes in the ring.
                    while j < 2*(nranks-1):
                        cur_flow = NcclDataFlow(cur_node, next_node, coll_op.data_size, name=f"All-reduced Data")
                        cur_flow_id = cur_flow.id
                        data_flows[cur_flow_id] = cur_flow

                        assert prev_flow_id is not None
                        # need to wait for the previous flow to finish.
                        if cur_flow_id not in dependencies:
                            dependencies[cur_flow_id] = [prev_flow_id]
                        else:
                            dependencies[cur_flow_id].append(prev_flow_id)                        
                        prev_flow_id = cur_flow_id

                        cur_node = next_node
                        next_node = ring.get_next_node(cur_node)
                        j += 1


        elif coll_op.func == "AllGather":
            # Based on NCCL implementation in src/device/all_gather.h:runRing()
            pass

        return data_flows, dependencies
