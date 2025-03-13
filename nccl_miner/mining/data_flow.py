'''
Contain a simple data structure representing a data flow.
'''
class DataFlow:
    # counter used to generate flow id.
    global_id_counter = 0

    def __init__(self, src, dst, size, name=None, start_time=None, end_time=None):
        self.src = src
        self.dst = dst
        self.size = size
        self.data_name = name
        if start_time is not None and end_time is not None:
            self.start_time = start_time
            self.end_time = end_time
            self.duration = end_time - start_time
        self.id = DataFlow.global_id_counter
        DataFlow.global_id_counter += 1

    def __repr__(self):
        return f"Flow {self.id}:{self.src}->{self.dst}[{self.size} bytes]"


class CommunicationOperation:
    def __init__(self, op_name, data_type, data_num, data_size, devs, flows, deps):
        self.name = op_name
        self.data_type = data_type
        self.data_num = data_num
        self.data_size = data_size
        self.data_flows = flows
        self.dependencies = deps
        self.devices = devs
        
    def __repr__(self):
        return f"{self.name}: {self.data_size} bytes ({self.data_num} {self.data_type})"
