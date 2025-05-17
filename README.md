# NCCL Miner
NCCL Miner "Nickel Miner" is a tool for extracting data flow information from NCCL logs in Distributed Machine Learning usecases. Flow information includes the size and type of the flow, the source and destination of the flow, and dependencies between the flows, etc.

# File Structure
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

Experiments:
- `experiments/`:
  - `4d_parallelism/`: Contain several distributed training experiments with 4D Parallelism (Data, Tensor, Pipeline, and Context Paralallelism). Based on the Picotron project from https://github.com/huggingface/picotron, a minimalistic 4D-parallelism distributed training framework for education purpose.
  - `gpt2/`: An example to train GPT2 model distributively. The GPT2 model comes from MinGPT: https://github.com/karpathy/minGPT, a clean, simple PyTorch implementation of GPT-2 model.

Misc:
- `docs/`: Some documents explaining how NcclMiner system works.
- `example_nccl_logs/`: NCCL logs gathered in real training scenarios for example usage
- `example_topo/`: Hardware topology detected by NCCL for example usage

# General Usage
   - Command: `python main.py <nccl_log_dir> <torch_log_dir> -o <out_trace_json_path> -d <out_flow_dump_json_path>`
       - `<nccl_log_dir>`: Parent directory containing all NCCL log files only. Should be the directory of NCCL_DEBUG_FILE
       - `<torch_log_dir>`: Parent direcotry containing all Torch profiler output only. Should be the value of arg `--torch_profiler_path`
       - `<out_trace_json_path>`: Path to store the Perfetto trace JSON file
       - `<out_flow_dump_json_path>`: Path to a JSON file to dump all data flows

# An Example Walkthrough

This example walkthrough gives an overview on how the NcclMiner system works. The steps can be divided into two stages:
1. Collect NCCL and Torch logs from the distributed training example.
   - In this walkthrough, we will use the Data Parallelism example under `experiments/4d_parallelism/`.
   - In the real-world usage, the user should replace this stage with their own Distributed ML workload.
2. Extract Flow-level info from NCCL and Torch logs.
   - At this stage, user will use the scripts in NcclMiner system to extract flow-level info.

NOTE: This example assume you have 3 GPUs available in your dev environment. If that's not the case, please modify `experiments/4d_parallelism/dp/config.json` correspondinly to suit your environment (See section below for details).

## Stage 1: Collect NCCL and Torch Logs

This Data Parallelism example relies on the Picotron project, which provides a convenient way to test different models and parallelism techniques.

### Install Dependencies

The very first step to run this example is to ensure that your **Python version is 3.12** (other versions are not tested for now).

Here're several options to install all dependencies:

#### Using Conda
```
pip install -r requirements.txt
```

#### Using uv
`pyproject.toml` file is working in progress.

#### Manual Installation
If all the steps above does not work for you, try manually installing with `pip`:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton
pip install packaging
pip install wheel
pip install psutil
pip install flash-attn==2.7.3 --no-build-isolation
pip install datasets
pip install transformers
pip install wandb
pip install huggingface_hub[hf_transfer]
pip install pulp
```
If you are using other environment setup tools, please make sure that your Python version is 3.12 and you have installed all the dependencies above.

### Run the Distributed Training Script

1. Go to the path containing this example: `cd experiments/4d_parallelism/`
2. Create two new directories to store NCCL and Torch output logs, correspondingly:
   - If directories already exist, please remove them:
      ```
      rm -rf dp/nccl_logs
      rm -rf dp/torch_logs
      ```
   - Create the directories:
      ```
      mkdir dp/nccl_logs
      mkdir dp/torch_logs
      ```

3. Run the training scripts:
    ```
    NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=dp/nccl_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 3 train_picotron.py --config dp/config.json --torch_profiler_path dp/torch_logs
    ```
    - This command is running on 3 GPUs. Change `--nproc_per_node` value in this command and `dp_size` value in `dp/config.json` to the number of GPUs you wish to use in your environment.

4. Wait for command and check if log outputs are generated successfully:
    - You should see three text files under `dp/nccl_logs` and three JSON files under `dp/torch_logs`.

## Stage 2: Extract Flow-level Info

1. Go back to the root of this project

2. Run NCCL Miner scripts to generate traces from the logs collected in Stage 1:
   ```
   python main.py experiments/4d_parallelism/dp/nccl_logs/ experiments/4d_parallelism/dp/torch_logs/ -o experiments/4d_parallelism/dp/dp_trace.json -d experiments/4d_parallelism/dp/dp_flow_dump.json
   ```

3. Wait for command to complete and check the output:
   - There should be two JSON files generated under `experiments/4d_parallelism/dp` :
     - `dp_trace.json` and `dp_flow_dump.json`

4. Upload `dp_trace.json` to https://ui.perfetto.dev/

## Interpretation of Output

Working in progress. See here for now:
- https://github.com/cuiboyuan/nccl_miner/pull/12
- https://github.com/cuiboyuan/nccl_miner/pull/25


## NCCL Material

NCCL Environment Variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html

NCCL API: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html

NCCL Data Types: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t