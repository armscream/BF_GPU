#ifndef BF_GPU_INCLUDES_PUSH_CONSTANTS_CULLING_COMMAND_RESET_PC_GLSL
#define BF_GPU_INCLUDES_PUSH_CONSTANTS_CULLING_COMMAND_RESET_PC_GLSL

#include "../SharedGpuTypes.glsl"

struct CullingCommandResetPC {
    uint64_t frameGlobalContextBufferAddr;
};

#endif