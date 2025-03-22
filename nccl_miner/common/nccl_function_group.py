'''
Group NCCL functions across different devices into Function Groups because they belong to the same collective operation.
*This is not Group calls in NCCL
'''
from typing import *

from .nccl_function import NcclCommInitRankFunction, NcclPtpFunction, NcclCollectiveFunction
from .topology import Ring, MultiRing

'''
Function groups of NCCL communicator initialization calls.
NCCL associate each API call with a communication object, which will be repr. as Clique here.
This helps us to identify which NCCL APIs should be group together.
'''
class NcclCommClique:
    def __init__(self, nccl_init_calls: List[NcclCommInitRankFunction]):
        self.id = None
        self.rank_to_device = {}
        self.device_to_rank = {}

        partial_rings = []
        partial_trees = []
        for comm_init in nccl_init_calls:
            if self.id is None:
                self.id = comm_init.comm_id
            assert self.id == comm_init.comm_id
            # Map the rank in this clique to actual CUDA device.
            self.rank_to_device[comm_init.cur_rank] = comm_init.device
            self.device_to_rank[comm_init.device] = comm_init.cur_rank
            # Construct the Ring.
            partial_rings.append(comm_init.partial_rings)
            # TODO: Construct the Tree.
            # ...

        print("Constructing Ring...")
        # Complete full ring from partial rings
        incomplete_rings = {}
        for partial in partial_rings:
            for ring_id, topo in partial.items():
                prev = topo['prev']
                cur = topo['cur']
                next = topo['next']
                if ring_id not in incomplete_rings:
                    incomplete_rings[ring_id] = {}
                incomplete_rings[ring_id][cur] = next

        all_rings = {}
        for ring_id, ring_dict in incomplete_rings.items():
            # finalize the ring
            # Find a random starting point
            entry_node = None
            for n in ring_dict:
                entry_node = n
                break
            # Sort the nodes in order to reflect ring structure
            ring_nodes = []
            cur_node = None
            while cur_node != entry_node:
                if cur_node is None:
                    cur_node = entry_node
                ring_nodes.append(cur_node)
                cur_node = ring_dict[cur_node]

            ring = Ring(ring_nodes, ring_id)
            all_rings[ring_id] = ring
        self.ring_algo = MultiRing(all_rings, self.rank_to_device)
        
        # Complete full tree from partial trees
        # TODO: ...
    
    def get_device_rank(self, dev):
        return self.device_to_rank[dev]
    
    def get_rank_device(self, rank):
        return self.rank_to_device[rank]

'''
Function groups of NCCL Collective/SendRecv API calls.
After successfully identified NCCL API groups (through Cliques above), this class repr. a group of NCCL APIs across different devices that belong together.
'''
class NcclFunctionGroup:
    # counter used to generate group id.
    global_id_counter = 0
    def __init__(self,
                 ops_per_device,
                 clique: NcclCommClique):
        self.device_operations = ops_per_device
        self.clique = clique
        self.id = NcclFunctionGroup.global_id_counter
        NcclFunctionGroup.global_id_counter += 1

        self.all_devices = []
        for device, op in ops_per_device.items():
            op.associate_group_id(self.id)
            self.all_devices.append(device)

    def associate_data_flows(self, data_flow, deps):
        self.data_flows = data_flow
        self.dependencies = deps


class NcclCollectiveFunctionGroup(NcclFunctionGroup):
    def __init__(self, ops_per_device, clique):
        super().__init__(ops_per_device, clique)

        self.main_operation = None
        for device, op in ops_per_device.items():
            assert isinstance(op, NcclCollectiveFunction)
            if self.main_operation is None:
                self.main_operation = op
    
    def get_algo(self):
        # TODO: support only ring algo for now,
        # change to support the correct algo
        return self.clique.ring_algo


class NcclPtpFunctionGroup(NcclFunctionGroup):
    def __init__(self, ops_per_device, clique):
        super().__init__(ops_per_device, clique)
        self.src = None
        self.dst = None
        for device, op in ops_per_device.items():
            assert isinstance(op, NcclPtpFunction)
            if op.func == "Send":
                self.src = device
                self.data_num = op.data_num
                self.data_type = op.data_type
                self.data_size = op.data_size
            elif op.func == "Recv":
                self.dst = device