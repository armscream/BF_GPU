#ifndef BF_GPU_INCLUDES_UTILS_CULLING_UTILS_GLSL
#define BF_GPU_INCLUDES_UTILS_CULLING_UTILS_GLSL

#include "../Common/Material.glsl"

uint GetMaterialRenderType(Material mat) {
    bool isTransparent = IS_TRANSPARENT(mat);
    bool isAlphaTested = IS_ALPHA_TESTED(mat);
    bool isDoubleSided = IS_DOUBLE_SIDED(mat);

    if (isTransparent) {
        if (isAlphaTested) return isDoubleSided ? 7 : 6;
        else               return isDoubleSided ? 5 : 4;
    } else {
        if (isAlphaTested) return isDoubleSided ? 3 : 2;
        else               return isDoubleSided ? 1 : 0;
    }
}

// CalculateLodFromScreenSize mirrors the host-side
// lod_select_from_screen_size in BF_GPU/Gpu_Types.odin: the GPU LOD
// selector uses three descending thresholds (512, 256, 128 pixels) on
// the projected screen footprint to pick one of four buckets.
//
// `lodBias` is a quality knob, positive values bias toward higher
// detail (smaller LOD index); the function multiplies the footprint
// by 2^lodBias before threshold comparison. `lodCount` clamps the
// returned bucket to the asset-baked range, so an asset that only
// baked 2 LODs never sees a slot the culling pipeline cannot render.
uint CalculateLodFromScreenSize(float screenSizePixels, float lodBias, uint lodCount) {
    float effectiveSize = screenSizePixels;
    if (lodBias != 0.0) {
        effectiveSize = screenSizePixels * exp2(lodBias);
    }
    uint lod = 3u;
    if (effectiveSize > 512.0)      lod = 0u;
    else if (effectiveSize > 256.0) lod = 1u;
    else if (effectiveSize > 128.0) lod = 2u;

    uint count = lodCount;
    if (count == 0u) count = 1u;
    return min(lod, count - 1u);
}

bool IsModelSimpleEnoughForFastPath(uint meshCount, uint vertexCount) {
    const uint MAX_FAST_PATH_MESHES = 8u;
    const uint MAX_FAST_PATH_VERTICES = 4096u;
    
    return (meshCount <= MAX_FAST_PATH_MESHES && vertexCount < MAX_FAST_PATH_VERTICES);
}

#endif