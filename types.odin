//BF_Renderer/types.odin
package BF_Renderer

import hm "core:container/handle_map"
import ECS "../BF_ECS"
import "../../Core"

// Culling ordering
// ModelCull -> MeshCull -> MeshletCull
// Chunk(spatial)(CPU) -> Frustum -> Zero-Pixel -> Hi-Z -> Cone (meshlet)
// LOD is selected per Simple Large bound box and screen space % then indexed. 

GPU_Model_ID :: distinct u32
GPU_Mesh_ID :: distinct u32
GPU_Material_ID :: distinct u32
GPU_Texture_ID :: distinct u32

GPU_Model :: struct {
	mesh_id: GPU_Mesh_ID,
	material_id: GPU_Material_ID,
}
GPU_Mesh :: struct {
//	vertex_buffer: GPU_Buffer_ID,
//	index_buffer: GPU_Buffer_ID,
	index_count: u32,
}
GPU_Material :: struct {
	texture_id: GPU_Texture_ID,
}
GPU_Texture :: struct {
	texture_id: GPU_Texture_ID,
}

Asset_GPU_Map :: struct {
	models: hm.Handle64,
	meshes: hm.Handle64,
	materials: hm.Handle64,
	textures: hm.Handle64,
}

Render_Scene :: struct {
	instances: Render_Instance_Store,
	lights: Render_Light_Store,
	particles: Render_Particle_Store,
	// Render-side spatial metadata
	spatial: Render_Spatial_Store,
}

Render_Instance_Store :: struct {
	instances: []Render_Instance,
}
Render_Light_Store :: struct {
	lights: []Render_Light,
}
Render_Particle_Store :: struct {
	particles: []Render_Particle,
}
Render_Spatial_Store :: struct {
	spatial: []Render_Spatial,
}

Render_Instance :: struct {
    entity: ECS.Entity,
    model: Core.Asset_Ref,
    transform_index: u32,
    material_override_index: u32,
    flags: Render_Instance_Flags,
}
Render_Instance_Flags :: bit_set[Render_Instance_Flag]
Render_Instance_Flag :: enum u8 {
    None,
}

Render_Light :: struct {
    entity: ECS.Entity,
    transform_index: u32,
    light_index: u32,
}
Render_Particle :: struct {

}
Render_Spatial :: struct {

}

Render_Bucket_Key :: struct {
    domain: Render_Domain,
    pipeline: Render_Pipeline,
    material_class: Render_Material_Class,
}

Render_Domain :: enum { 
    Geometry,
    Particle,
    Decal,
    Terrain,
    Water,
    UI,
    Shadow,
    Post_Process,
}

Render_Bucket :: enum u8 {
    Traditional_Opaque_1Sided,
    Traditional_Opaque_2Sided,
    Traditional_Transparent_1Sided,
    Traditional_Transparent_2Sided,
    Mesh_Opaque_1Sided,
    Mesh_Opaque_2Sided,
    Mesh_Transparent_1Sided,
    Mesh_Transparent_2Sided,
}

// Render_Instance
// Render_Light
// Render_Camera
// Render_Particle
// Material_Override

// GPU_Instance
// GPU_Model
// GPU_Mesh
// GPU_Material
// GPU_Texture
// Visible_Instance
// Indirect_DrawS