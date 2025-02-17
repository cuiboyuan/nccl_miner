

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