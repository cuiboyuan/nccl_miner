#include <assert.h>

#include "nvbit_tool.h"

#include "nvbit.h"

#include "utils/utils.h"

FILE* fptr = nullptr;

void nvbit_tool_init(CUcontext ctx) {
    printf("nvbit tool init\n");
}

void nvbit_at_init() {
    int verbose = 0;
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    fprintf(stderr, "nvbit_at_init() called\n");
    fflush(stderr);

    fptr = fopen("/home/ubuntu/meng-project/trace.txt", "w+");
    fputs("Start\n", fptr);
    fflush(fptr);
}

void nvbit_at_ctx_init(CUcontext ctx) {
    printf("nvbit at context init\n");
        fputs("Ctx Init\n", fptr);
    fflush(fptr);
}


void nvbit_at_ctx_term(CUcontext ctx) {
    printf("nvbit at context term\n");
        fputs("Ctx Term\n", fptr);
    fflush(fptr);
}

void nvbit_at_term() {
    printf("nvbit at term\n");
    fputs("End\n", fptr);
    fclose(fptr);
}


void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* event_name, void* params,
                         CUresult* pStatus) {
    /* cast params to launch parameter based on cbid since if we are here
    * we know these are the right parameters types */
    CUfunction func;
    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel) {
        cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
        func = p->f;
        printf("Event %s\n", nvbit_get_func_name(ctx, func));
    }
    
    /* Identify all the possible CUDA Memcpy events */
    switch (cbid) {
        case API_CUDA_cuMemcpyHtoD:
        case API_CUDA_cu64MemcpyHtoD:
        case API_CUDA_cuMemcpyHtoDAsync:
        case API_CUDA_cu64MemcpyHtoDAsync:
        case API_CUDA_cuMemcpyHtoD_v2:
        case API_CUDA_cuMemcpyHtoDAsync_v2:
        case API_CUDA_cuMemcpyHtoD_v2_ptds:
        case API_CUDA_cuMemcpyHtoDAsync_v2_ptsz:
            // HtoD
            // break;
        case API_CUDA_cuMemcpyDtoH:
        case API_CUDA_cu64MemcpyDtoH:
        case API_CUDA_cu64MemcpyDtoHAsync:
        case API_CUDA_cuMemcpyDtoHAsync:
        case API_CUDA_cuMemcpyDtoHAsync_v2:
        case API_CUDA_cuMemcpyDtoH_v2:
        case API_CUDA_cuMemcpyDtoH_v2_ptds:
        case API_CUDA_cuMemcpyDtoHAsync_v2_ptsz:
            // DtoH
            // break;
        case API_CUDA_cuMemcpyDtoD:
        case API_CUDA_cu64MemcpyDtoD:
        case API_CUDA_cuMemcpyDtoDAsync:
        case API_CUDA_cu64MemcpyDtoDAsync:
        case API_CUDA_cuMemcpyDtoD_v2:
        case API_CUDA_cuMemcpyDtoD_v2_ptds:
        case API_CUDA_cuMemcpyDtoDAsync_v2:
        case API_CUDA_cuMemcpyDtoDAsync_v2_ptsz:
            // DtoD
            // break;
        case API_CUDA_cuMemcpyAtoH:
        case API_CUDA_cuMemcpyAtoHAsync:
        case API_CUDA_cuMemcpyAtoH_v2:
        case API_CUDA_cuMemcpyAtoHAsync_v2:
        case API_CUDA_cuMemcpyAtoH_v2_ptds:
        case API_CUDA_cuMemcpyAtoHAsync_v2_ptsz:
            // AtoH
            // break;
        case API_CUDA_cuMemcpyHtoA:
        case API_CUDA_cuMemcpyHtoAAsync:
        case API_CUDA_cuMemcpyHtoA_v2:
        case API_CUDA_cuMemcpyHtoAAsync_v2:
        case API_CUDA_cuMemcpyHtoA_v2_ptds:
        case API_CUDA_cuMemcpyHtoAAsync_v2_ptsz:
            // HtoA
            // break;
        case API_CUDA_cuMemcpyAtoA:
        case API_CUDA_cuMemcpyAtoA_v2:
        case API_CUDA_cuMemcpyAtoA_v2_ptds:
            // AtoA
            // break;
        case API_CUDA_cuMemcpy_v2:
        case API_CUDA_cuMemcpy:
        case API_CUDA_cuMemcpyAsync:
        case API_CUDA_cuMemcpyPeer:
        case API_CUDA_cuMemcpyPeerAsync:
        case API_CUDA_cuMemcpy_ptds:
        case API_CUDA_cuMemcpyPeer_ptds:
        case API_CUDA_cuMemcpyAsync_ptsz:
        case API_CUDA_cuMemcpyPeerAsync_ptsz:
            // Memcpy
            char trace_name[64];
            if (is_exit) {
                sprintf(trace_name, "<< %s", event_name);
            } else {
                sprintf(trace_name, ">> %s", event_name);
            }
            printf("%s\n", trace_name);
            fprintf(fptr, "%s\n", trace_name);
            fflush(fptr);
            break;
        default:
            break;
    }
}