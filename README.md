# NCCL Miner
NCCL Miner "Nickel Miner" is a tool for extracting data flow information from NCCL logs in Distributed Machine Learning usecases. Flow information includes the size and type of the flow, the source and destination of the flow, and dependencies between the flows, etc.

## File Structure
- `example_nccl_logs/`: NCCL logs gathered in real training scenarios for example usage
- `example_topo/`: Hardware topology detected by NCCL for example usage
- `mingpt/`: From https://github.com/karpathy/minGPT, a clean, simple PyTorch implementation of GPT-2 model
- `nccl_miner/`: Main folder containing scripts extracting flow-level information from NCCL logs
- `train_gpt2.py`: Example training GPT-2 with torchrun
- `trace_gen.py`: Generate Chrome traces from data flow extracted by NCCL Miner

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
Then put all `nccl_logs.*` files in a folder.

## Step 2: Generate Trace of Data Flow

### Parse the NCCL Logs into Trace
```
python TraceGen.py <your nccl log folder>
```
The output trace is called `nccl_trace.json`.

You can also use existing logs as an example:
```
python TraceGen.py example_nccl_logs/four_gpu_p2p_shm_disabled/
```

## Step 3: View Trace
Open `nccl_trace.json` in chrome://tracing
