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

## Example Usage

First install all the dependencies, please make sure that the Python version is **3.12**:
```
pip install -r requirements.txt
```

### Picotron Experiments

1. Create two new directories to store NCCL logs and Torch Profiler outputs, correspondingly:

   - For example (for Data Parallelism Experiment):
      ```
      mkdir -p experiments/picotron/dp/dp_logs
      mkdir -p experiments/picotron/dp/torch_profiler
      ```

2. Gather NCCL and Torch logs using Picotron Project:
   - Data Parallelism:
     - Mostly AllReduce operation.
     - Command: `NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=experiments/picotron/dp/dp_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 3 train_picotron.py --config experiments/picotron/dp/config.json  --torch_profiler_path experiments/picotron/dp/torch_profiler`
   - Pipeline Parallelism:
     - Mostly Send/Recv operation.
     - Command: `NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=experiments/picotron/pp/pp_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 3 train_picotron.py --config experiments/picotron/pp/config.json  --torch_profiler_path experiments/picotron/pp/torch_profiler`

3. Run NCCL Miner scripts to generate traces:
   - Command: `python main.py <nccl_log_dir> <torch_log_dir> -o <out_trace_json_path> -d <out_flow_dump_json_path>`
       - `<nccl_log_dir>`: Parent directory containing all NCCL log files only. Should be the directory of NCCL_DEBUG_FILE
       - `<torch_log_dir>`: Parent direcotry containing all Torch profiler output only. Should be the value of arg `--torch_profiler_path`
       - `<out_trace_json_path>`: Path to store the Perfetto trace JSON file
       - `<out_flow_dump_json_path>`: Path to a JSON file to dump all data flows

4. Upload trace JSON file to https://ui.perfetto.dev/

## NCCL Material

NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

NCCL API: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html

NCCL Data Types: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t