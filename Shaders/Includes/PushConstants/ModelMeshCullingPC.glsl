#ifndef BF_GPU_INCLUDES_PC_MODEL_CULLING_PASS_GLSL
#define BF_GPU_INCLUDES_PC_MODEL_CULLING_PASS_GLSL

#include "../SharedGpuTypes.glsl"

struct ModelMeshCullingPC {
    uint64_t frameGlobalContextBufferAddr;
    uint lodCount;
    float lodBias;
};

#endif