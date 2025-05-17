## Available Experiments

### Data Parallelism over 3 GPUs:
- Mostly AllReduce operations
- Command to generate logs:
    ```
    rm -rf dp/nccl_logs
    rm -rf dp/torch_logs

    mkdir dp/nccl_logs
    mkdir dp/torch_logs

    NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=dp/nccl_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 3 train_picotron.py --config dp/config.json --torch_profiler_path dp/torch_logs
    ```
- Command to generate trace from logs:
    ```
    cd ../..
    python main.py experiments/4d_parallelism/dp/nccl_logs/ experiments/4d_parallelism/dp/torch_logs/ -o experiments/4d_parallelism/dp/dp_trace.json -d experiments/4d_parallelism/dp/dp_flow_dump.json
    ```

### Pipeline Parallelism over 3 GPUs:
- Mostly Send/Recv operations
- Command to generate logs:
    ```
    rm -rf pp/nccl_logs
    rm -rf pp/torch_logs

    mkdir pp/nccl_logs
    mkdir pp/torch_logs

    NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=pp/nccl_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 3 train_picotron.py --config pp/config.json --torch_profiler_path pp/torch_logs
    ```
- Command to generate trace from logs:
    ```
    cd ../..
    python main.py experiments/4d_parallelism/pp/nccl_logs/ experiments/4d_parallelism/pp/torch_logs/ -o experiments/4d_parallelism/pp/pp_trace.json -d experiments/4d_parallelism/pp/pp_flow_dump.json
    ```

### Tensor Parallelism over 2 GPUs:
- Mostly AllReduce operations
- Takes longer to run
- Generates larger NCCL and Torch logs, hence longer for NcclMiner to process
- Command to generate logs:
    ```
    rm -rf tp/nccl_logs
    rm -rf tp/torch_logs

    mkdir tp/nccl_logs
    mkdir tp/torch_logs

    NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=tp/nccl_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 2 train_picotron.py --config tp/config.json --torch_profiler_path tp/torch_logs
    ```

### Context Parallelism over 3 GPUs:
- Mostly Send/Recv operations.
- Takes longer to run
- Generates larger NCCL and Torch logs, hence longer for NcclMiner to process
- Command to generate logs:
    ```
    rm -rf cp/nccl_logs
    rm -rf cp/torch_logs

    mkdir cp/nccl_logs
    mkdir cp/torch_logs

    NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=cp/nccl_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 3 train_picotron.py --config cp/config.json --torch_profiler_path cp/torch_logs
    ```