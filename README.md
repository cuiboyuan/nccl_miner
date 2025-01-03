# dml-tracer
A tool that traces data flows in Distributed Machine Learning. The tracer collects data flow information such as size and type of the flow, source and destination, and flow dependencies.

# Example: GPT2

## Install required dependencies
```
pip install -r requirements.txt
```

## Usage

### NCCL Log Path
```
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_TOPO_DUMP_FILE="nccl_topo.xml" python GPT2Dist.py | tee nccl_log.txt
```
NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

NCCL API: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html 

NCCL Data Types: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t

### NVBit Path
```
cd dml_tracer
make
<should generated dml_tracer.so file>
```
```
cd <root of this repo>
LD_PRELOAD=dml_tracer/dml_tracer.so python GPT2Dist.py
```
