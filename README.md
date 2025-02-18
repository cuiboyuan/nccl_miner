# NCCL Miner
NCCL Miner "Nickel Miner" is a tool for extracting data flow information from NCCL logs in Distributed Machine Learning usecases. Flow information includes the size and type of the flow, the source and destination of the flow, and dependencies between the flows, etc.

## File Structure
- `example_nccl_logs/`: NCCL logs gathered in real training scenarios for example usage
- `example_topo/`: Hardware topology detected by NCCL for example usage
- `experiments/picotron`: Configs to run 4-D parallelism techniques to gather example logs with Picotron
- `mingpt/`: From https://github.com/karpathy/minGPT, a clean, simple PyTorch implementation of GPT-2 model
- `nccl_miner/`: Main folder containing scripts extracting flow-level information from NCCL logs
- `picotron/`: From https://github.com/huggingface/picotron, a simple educational project helping people quickly get familiar with all techniques in distributed training.
- `main.py`: Generate Chrome traces from data flow extracted by NCCL Miner
- `train_gpt2.py`: Example training GPT-2 with torchrun
- `train_picotron.py`: From https://github.com/huggingface/picotron, slightly modified scripts to train distributedly with 4-D parallelism techniques.

## NCCL Material

NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

NCCL API: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html 

NCCL Data Types: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t

# Example Usage: Picotron

Data Parallelism:
```
rm -rf experiments/picotron/dp/dp_logs
mkdir experiments/picotron/dp/dp_logs

NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=experiments/picotron/dp/dp_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 3 train_picotron.py --config experiments/picotron/dp/config.json
```

Context Parallelism:
```
rm -rf experiments/picotron/cp/cp_logs
mkdir experiments/picotron/cp/cp_logs

NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=experiments/picotron/cp/cp_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 3 train_picotron.py --config experiments/picotron/cp/config.json
```

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
python main.py example_nccl_logs/four_gpu_p2p_shm_disabled/
```

## Step 3: View Trace
Open `nccl_trace.json` in chrome://tracing
