'''
Contain a simple data structure representing a data flow.
'''
class DataFlow:
    # counter used to generate flow id.
    global_id_counter = 0

    def __init__(self, src, dst, size, name=None):
        self.src = src
        self.dst = dst
        self.size = size
        self.data_name = name
        self.id = DataFlow.global_id_counter
        DataFlow.global_id_counter += 1

    def associate_time(self, src_start_ts, src_end_ts,
                           dst_start_ts, dst_end_ts):
        self.src_start_time = src_start_ts
        self.src_end_time = src_end_ts
        self.src_duration = src_end_ts - src_start_ts

        self.dst_start_time = dst_start_ts
        self.dst_end_time = dst_end_ts
        self.dst_duration = dst_end_ts - dst_start_ts

        self.flow_start_time = src_start_ts
        self.flow_end_time = dst_end_ts
        self.flow_duration = self.flow_end_time - self.flow_start_time

    def __repr__(self):
        return f"Flow {self.id}:{self.src}->{self.dst}[{self.size} bytes]"
