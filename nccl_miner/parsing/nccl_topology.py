class Ring:
    def __init__(self, ring_id):
        self.complete = False
        self.id = ring_id
        self.nodes = []
        self.incomplete_ring = {}
    
    def add_node(self, cur, next):
        if self.complete:
            return
        self.incomplete_ring[cur] = next
    
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


class NcclRing:
    def __init__(self, partial_rings, rank_to_dev):
        incomplete_rings = {}
        for partial in partial_rings:
            for ring_id, topo in partial.items():
                prev = topo['prev']
                cur = topo['cur']
                next = topo['next']
                if ring_id not in incomplete_rings:
                    incomplete_rings[ring_id] = Ring(ring_id)
                incomplete_rings[ring_id].add_node(cur, next)

        self.rank_to_device_mapping = rank_to_dev
        self.devices = [dev for _, dev in self.rank_to_device_mapping.items()]

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
    
    def rank_to_device(self, rank):
        return self.rank_to_device_mapping[rank]