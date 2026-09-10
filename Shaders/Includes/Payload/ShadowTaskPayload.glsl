#ifndef BF_GPU_INCLUDES_SHADOW_TASK_PAYLOAD_H
#define BF_GPU_INCLUDES_SHADOW_TASK_PAYLOAD_H

struct ShadowTaskPayload {
    uint drawId;
    uint entityId;
    uint transformDenseIdx;
    uint lightShadowDenseIdx;
    uint cascadeIdx;
    uint modelDenseIndex;
    uint meshletIndices[32];
};

#endif