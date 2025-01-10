# NCCL Miner
NCCL Miner "Nickel Miner" is a tool that extracts data flow information from NCCL logs in Distributed Machine Learning. Flow information includes the size and type of the flow, the source and destination of the flow, and dependencies between the flows, etc.

## NCCL Material

NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

NCCL API: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html 

NCCL Data Types: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t

# Example Usage: GPT2

## Step 1: Obtain NCCL Logs

### Install required dependencies
```
pip install -r requirements.txt
```
### Run Distributed Training Example
```
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=nccl_logs.%h.%p python GPT2Dist.py
```

## Step 2: Generate Trace of Data Flow

### Using Existing Example Logs
```
python TraceGen.py example_nccl_logs/four_gpu_p2p_shm_disabled/
```
The output is in `nccl_trace.json`

## Step 3: View Trace
Open `nccl_trace.json` in chrome://tracing
