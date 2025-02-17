class Ring:
    def __init__(self, nodes, ring_id):
        self.id = ring_id
        self.nodes = nodes
        self.size = len(nodes)

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
    def __init__(self, all_rings, rank_to_dev):
        self.rank_to_device_mapping = rank_to_dev
        self.devices = [dev for _, dev in self.rank_to_device_mapping.items()]

        self.rings = all_rings
        self.size = None
        # check if all rings have equal sizes
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