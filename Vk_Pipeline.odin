package BF_GPU

import vk "vendor:vulkan"

GPU_Pass_Type :: enum {
    Compue,
    Graphics,
}
GPU_Pass :: struct {
    name: string,
    type: GPU_Pass_Type,
    pipeline: vk.Pipeline,
    layout: vk.PipelineLayout,
    shader: vk.ShaderModule,
    enabled: bool,
}

renderer_build_default_passes :: proc(r: ^Vulkan_Renderer){
    renderer_register_compute_pass(
        r, 
        "HiZ.LinearizeDepth",
        shader_path("Hiz/LinearizeDepth.comp")
    )
    renderer_register_compute_pass(
        r, 
        "HiZ.Downsample",
        shader_path("Hiz/Downsample.comp")
    )
    renderer_register_compute_pass(
        r, 
        "Culling.CommandReset",
        shader_path("Culling/CommandReset.comp")
    )
    renderer_register_compute_pass(
        r, 
        "Culling.StaticChunk",
        shader_path("Culling/StaticChunk.comp")
    )
    renderer_register_compute_pass(
        r, 
        "Culling.StaticModel",
        shader_path("Culling/StaticModelCulling.comp")
    )
    renderer_ep_register_compute_pass(
        r,
        "Culling.Mesh",
        shader_path("Culling/MeshCulling.comp")
    )
    renderer_register_compute_pass(
        r, 
        "Shading.Traditional",
        shader_path("Shading/Traditional.vert"),
        shader_path("Shading/Traditional.frag")
    )
}