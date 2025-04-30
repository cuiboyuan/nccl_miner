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

    def associate_time(self, send_start_ts, send_end_ts,
                           recv_start_ts, recv_end_ts):
        self.send_start_time = send_start_ts
        self.send_end_time = send_end_ts
        self.send_duration = send_end_ts - send_start_ts

        self.recv_start_time = recv_start_ts
        self.recv_end_time = recv_end_ts
        self.recv_duration = recv_end_ts - recv_start_ts

        self.flow_start_time = send_start_ts
        self.flow_end_time = recv_end_ts
        self.flow_duration = self.flow_end_time - self.flow_start_time

    def __repr__(self):
        return f"Flow {self.id}:{self.src}->{self.dst}[{self.size} bytes]"
