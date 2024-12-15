#include <stdio.h>

#include "nvbit_tool.h"

#include "nvbit.h"


void nvbit_at_init() {
    printf("nvbit at init\n");
}


void nvbit_at_term() {
    printf("nvbit at term\n");
}


void nvbit_at_ctx_init(CUcontext ctx) {
    printf("nvbit at context init\n");
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
    printf("Event\n");
}