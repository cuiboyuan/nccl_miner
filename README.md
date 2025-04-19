# NCCL Miner
NCCL Miner "Nickel Miner" is a tool for extracting data flow information from NCCL logs in Distributed Machine Learning usecases. Flow information includes the size and type of the flow, the source and destination of the flow, and dependencies between the flows, etc.

## File Structure
Core:
- `nccl_miner/`: Main folder of the project.
  - `common/`: Common data structures that are used by all components of the project
    - `nccl_data_type.py`: Contains utility to calculate number of bytes of each NCCL data type
    - `nccl_function.py`: Contains data structures that represent individual NCCL function calls happening on each GPU.
    - `nccl_function_group.py`: Contains data structures that represent multiple NCCL functions performing the same task. For example, if an AllReduce operation on 3 GPUs occurs, then there'll be 1 ncclAllReduce function on each GPU. A "function group" represents all 3 of them, as they are performing the same task.
    - `topology.py`: Contains data structures representing topologies used by NCCL, such as Ring and Tree.
    - `torch_event.py`: Contains data structures representing each event from Torch profiler output.
    - `torch_utils.py`: Contains utilities to interpret data from Torch profiler.
  - `parsing/`: Parse raw data from NCCL logs and Torch profiler output.
    - `nccl_parser.py`: Parse data from NCCL log files.
    - `torch_parser.py`: Parse data form Torch profiler output.
    - `combiner.py`: Pre-process and combines data across multiple files
  - `mining/`: Core of this project. Contains operations on extracting flow-level information
    - `data_flow.py`: Data structures representing a data flow
    - `dependency_deduction.py`: Deduce how data flow works under the hood from NCCL collective operations.
    - `timestamp_deduction.py`: Deduce the start time and duration of data flows based on existing info from Torch profiler and NCCL logs.
- `main.py`: Entry-point scripts to extract flow-level information from log files.
- `train_gpt2.py`: Example training GPT-2 with torchrun
- `train_picotron.py`: Copied from https://github.com/huggingface/picotron, slightly modified scripts to train distributedly with 4-D parallelism techniques.

External:
- `mingpt/`: From https://github.com/karpathy/minGPT, a clean, simple PyTorch implementation of GPT-2 model
- `picotron/`: From https://github.com/huggingface/picotron, a simple educational project helping people quickly get familiar with all techniques in distributed training.

Experiment Data and Configs:
- `example_nccl_logs/`: NCCL logs gathered in real training scenarios for example usage
- `example_topo/`: Hardware topology detected by NCCL for example usage
- `experiments/picotron`: Configs to run 4-D parallelism techniques to gather example logs with Picotron

## NCCL Material

NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

NCCL API: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html

NCCL Data Types: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t

# Example Usage: Picotron

See https://github.com/cuiboyuan/nccl_miner/pull/12 for details.

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
