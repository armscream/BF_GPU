#ifndef BF_GPU_INCLUDES_COMMON_INDIRECT_COMMAND_GLSL
#define BF_GPU_INCLUDES_COMMON_INDIRECT_COMMAND_GLSL

#include "../Core.glsl"

// ---------------------------------------------------------------------------
// Indirect command structs.
//
// The Bifrost renderer drives traditional geometry with
// vkCmdDrawIndexedIndirect, which consumes VkDrawIndexedIndirectCommand
// (5 u32 = 20 bytes). The host-side mirror in BF_GPU/Gpu_Types.odin
// (GPU_Indirect_Command) matches that layout, and the matching buffer
// stride is 20 bytes per slot (see vulkan_buffer_layout_kind).
//
// The meshlet pipeline drives its task+mesh dispatch with
// vkCmdDrawMeshTasksIndirectEXT, which consumes VkDrawMeshTasksIndirectCommandEXT
// (3 u32 = 12 bytes). The meshlet region of the global indirect command
// buffer sits immediately after the four traditional buckets
// (globalTraditionalCommandsCount * 20 bytes offset) and is read as a
// VkDrawMeshTasksIndirectBuffer with a 12-byte stride.
//
// VK_DRAW_INDEXED_CMD_STRIDE_BYTES and VK_DRAW_MESH_TASKS_CMD_STRIDE_BYTES
// are pinned to the host-side sizes so any future host-side drift shows
// up as a compile-time mismatch in tests.
//
// VK_GLOBAL_TRADITIONAL_CMD_STRIDE_BYTES / VK_GLOBAL_MESHLET_CMD_STRIDE_BYTES
// duplicate the per-region strides in a single identifier so the reset
// and culling shaders can name the regions uniformly without dragging
// in the host-side constants through more #defines.
// ---------------------------------------------------------------------------

#define VK_DRAW_INDEXED_CMD_STRIDE_BYTES      20u
#define VK_DRAW_MESH_TASKS_CMD_STRIDE_BYTES   12u
#define VK_GLOBAL_TRADITIONAL_CMD_STRIDE_BYTES 20u
#define VK_GLOBAL_MESHLET_CMD_STRIDE_BYTES     12u

struct VkDispatchIndirectCommand {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

struct VkDrawIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct VkDrawIndexedIndirectCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;
};

struct VkDrawMeshTasksIndirectCommandEXT {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

layout(buffer_reference, std430) restrict buffer VkDispatchIndirectBuffer         { VkDispatchIndirectCommand data; };
layout(buffer_reference, std430) restrict buffer VkDrawIndirectBuffer             { VkDrawIndirectCommand data[]; };
layout(buffer_reference, std430) restrict buffer VkDrawIndexedIndirectBuffer      { VkDrawIndexedIndirectCommand data[]; };
layout(buffer_reference, std430) restrict buffer VkDrawMeshTasksIndirectBuffer     { VkDrawMeshTasksIndirectCommandEXT data[]; };

#define GET_VK_DISPATCH_CMD(addr)              VkDispatchIndirectBuffer(addr).data
#define GET_VK_DRAW_CMD(addr, idx)             VkDrawIndirectBuffer(addr).data[idx]
#define GET_VK_DRAW_INDEXED_CMD(addr, idx)     VkDrawIndexedIndirectBuffer(addr).data[idx]
#define GET_VK_MESH_TASKS_CMD(addr, idx)       VkDrawMeshTasksIndirectBuffer(addr).data[idx]

// meshlet_indirect_base returns the byte address of the meshlet region
// of the global indirect command buffer. The traditional region holds
// (globalTraditionalCommandsCount) VkDrawIndexedIndirectCommand slots at
// VK_GLOBAL_TRADITIONAL_CMD_STRIDE_BYTES per slot; the meshlet region
// starts immediately after that and is indexed separately.
uint64_t meshlet_indirect_base(uint64_t globalAddr, uint traditionalCount) {
    return globalAddr + uint64_t(traditionalCount) * uint64_t(VK_GLOBAL_TRADITIONAL_CMD_STRIDE_BYTES);
}

#endif