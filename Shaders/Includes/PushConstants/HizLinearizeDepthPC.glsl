#ifndef BF_GPU_INCLUDES_PUSH_CONSTANTS_HIZ_LINEARIZE_DEPTH_PC_GLSL
#define BF_GPU_INCLUDES_PUSH_CONSTANTS_HIZ_LINEARIZE_DEPTH_PC_GLSL

#include "../SharedGpuTypes.glsl"

struct HizLinearizeDepthPC {
    uint64_t frameGlobalContextBufferAddr;
    vec2 outImageSize;
};

#endif