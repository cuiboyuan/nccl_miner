#include <assert.h>

#include "nvbit_tool.h"

#include "nvbit.h"

#include "utils/utils.h"


void nvbit_at_init() {
    int verbose = 0;
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    fprintf(stderr, "nvbit_at_init() called\n");
    fflush(stderr);
}

void nvbit_at_ctx_init(CUcontext ctx) {
    printf("nvbit at context init\n");
}

void nvbit_at_term() {
    printf("nvbit at term\n");
}

void nvbit_at_ctx_term(CUcontext ctx) {
    printf("nvbit at context term\n");
}


void nvbit_tool_init(CUcontext ctx) {
    printf("nvbit tool init\n");
}


void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* event_name, void* params,
                         CUresult* pStatus) {
     /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) 
        {
            /* cast params to launch parameter based on cbid since if we are here
            * we know these are the right parameters types */
            CUfunction func;
            if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
                cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
                func = p->f;
            } else {
                cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
                func = p->f;
            }
            printf("Event %s\n", nvbit_get_func_name(ctx, func));
        }
}