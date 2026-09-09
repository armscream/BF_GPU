//BF_Renderer/types.odin
package BF_GPU

import "../../Core"
import mth "../../Core/BF_Math"
import ECS "../BF_ECS"
import "base:runtime"

// Culling ordering
// ModelCull -> MeshCull -> MeshletCull
// Chunk(spatial)(CPU) -> Frustum -> Zero-Pixel -> Hi-Z -> Cone (meshlet)
// LOD is selected per Simple Large bound box and screen space % then indexed.

Asset_ID :: Core.Asset_ID
Asset_Ref :: Core.Asset_Ref
Entity :: ECS.Entity
Chunk_ID :: ECS.Chunk_ID
Light_Type :: ECS.Light_Type
Camera_Projection :: ECS.Camera_Projection

GPU_Model_ID :: distinct u32
GPU_Mesh_ID :: distinct u32
GPU_Material_ID :: distinct u32
GPU_Texture_ID :: distinct u32

GPU_MODEL_INVALID :: GPU_Model_ID(0)
GPU_MESH_INVALID :: GPU_Mesh_ID(0)
GPU_MATERIAL_INVALID :: GPU_Material_ID(0)
GPU_TEXTURE_INVALID :: GPU_Texture_ID(0)

GPU_Model :: struct {
	meshes_offset: u32,
	meshes_count:  u32,
	bounds:        mth.AABB,
	flags:         u32,
}
GPU_Mesh :: struct {
	vertex_buffer:  u32, //	vertex_buffer: GPU_Buffer_ID,
	index_buffer:   u32, //	index_buffer: GPU_Buffer_ID,
	vertex_offset:  u32,
	index_offset:   u32,
	index_count:    u32,
	meshlet_offset: u32,
	meshlet_count:  u32,
	bounds:         mth.AABB,
	material:       GPU_Material_ID,
}
GPU_Material :: struct {
	shader_id:           u32,
	texture_base_colour: GPU_Texture_ID,
	texture_normal:      GPU_Texture_ID,
	texture_orm:         GPU_Texture_ID,
	texture_emissive:    GPU_Texture_ID,
	flags:               u32,
}
GPU_Texture :: struct {
	image_id:   u32,
	sampler_id: u32,
	width:      u32,
	height:     u32,
	mip_count:  u32,
	flags:      u32,
}

//* Renderer resource tables
GPU_Resource_Store :: struct {
	models:            [dynamic]GPU_Model,
	meshes:            [dynamic]GPU_Mesh,
	materials:         [dynamic]GPU_Material,
	textures:          [dynamic]GPU_Texture,
	model_by_asset:    map[ECS.Asset_ID]GPU_Model_ID,
	mesh_by_asset:     map[ECS.Asset_ID]GPU_Mesh_ID,
	material_by_asset: map[ECS.Asset_ID]GPU_Material_ID,
	texture_by_asset:  map[ECS.Asset_ID]GPU_Texture_ID,
}

//* Render_Scene
Render_Instance_ID :: distinct u32
RENDER_INSTANCE_INVALID :: Render_Instance_ID(0)

Render_Instance :: struct {
	entity:                  Entity,
	model:                   Asset_Ref,
	gpu_model:               GPU_Model_ID,
	transform_index:         u32,
	material_override_index: u32,
	spatial_index:           u32,
	flags:                   Render_Instance_Flags,
}
Render_Instance_Flags :: bit_set[Render_Instance_Flag]
Render_Instance_Flag :: enum u8 {
	None,
}
//Later this can become GPU-packed: world mat, prev mat, norm/quat/scale
Render_Transform :: struct {
	world:          [16]f32,
	previous_world: [16]f32,
	normal:         [9]f32,
}
Render_Spatial_Metadata :: struct {
	bounds:   mth.AABB,
	chunk:    Chunk_ID,
	distance: f32,
	lod:      u16,
	flags:    Render_Spatial_Flags,
}
Render_Spatial_Flags :: bit_set[Render_Spatial_Flag]
Render_Spatial_Flag :: enum u8 {
	Visible,
	Culled,
	Static,
	Dynamic,
	Cast_Shadow,
	Receive_Shadow,
}
// CPU Render_Scene
Render_Scene :: struct {
	allocator:          runtime.Allocator,
	instances:          [dynamic]Render_Instance,
	transforms:         [dynamic]Render_Transform,
	spatial:            [dynamic]Render_Spatial_Metadata,
	lights:             [dynamic]Render_Light,
	cameras:            [dynamic]Render_Camera,
	particles:          [dynamic]Render_Particle,
	entity_to_instance: map[ECS.Entity]Render_Instance_ID,
	frame_index:        u64,
}

//* Render Objects
Render_Light :: struct {
	entity:           Entity,
	transform_index:  u32,
	type:             Light_Type,
	colour:           mth.Vec3,
	intensity:        f32,
	range:            f32,
	inner_cone:       f32,
	outer_cone:       f32,
	shadow_map_index: u32,
}
Render_Camera :: struct {
	entity:                 Entity,
	transform_index:        u32,
	projection:             Camera_Projection,
	view:                   [16]f32,
	projection_matrix:      [16]f32,
	view_projection_matrix: [16]f32,
	near_plane:             f32,
	far_plane:              f32,
	width:                  u32,
	height:                 u32,
}
Render_Particle :: struct {
	entity:          Entity,
	system:          Asset_Ref,
	transform_index: u32,
	emitter_index:   u32,
	flags:           u32,
}
Render_Spatial :: struct {}

//* RENDER PIPELINE SETTINGS
// Pipelines: GEOMETRY_BUCKET_COUNT :: 8 -- This is if we dont seperate btw
// Mesh shader mode and traditional, then you divide by 2. We dont need to support both
// At the same time really.
// Traditional * Opaque * Back
// Traditional * Opaque * None
// Traditional * Transparent * Back
// Traditional * Transparent * None
// Mesh * Opaque * Back
// Mesh * Opaque * None
// Mesh * Transparent * Back
// Mesh * Transparent * None
GEOMETRY_BUCKET_COUNT :: 8

Render_Bucket_Key :: struct {
	domain:         Render_Domain,
	pipeline:       Render_Pipeline,
	material_class: Render_Material_Class,
	cull_mode:      Render_Cull_Mode,
}

Render_Domain :: enum u8 {} //TODO: add fields, verify if necessary


Render_Pipeline :: enum u8 {
	Traditional,
	Mesh,
}

Render_Material_Class :: enum u8 {
	Opaque,
	Transparent,
}

Render_Cull_Mode :: enum u8 {
	Back,
	None,
}

Render_Bucket :: struct {
	key:             Render_Bucket_Key,
	instances:       [dynamic]Render_Instance_ID,
	visible_count:   u32,
	indirect_offset: u32,
}
Render_Bucket_Set :: struct {
	buckets: [dynamic]Render_Bucket,
}
// GEOMETRY -> 8 default buckets (all instances organized into their buckets - one draw call per bucket)
// PARTICLE -> particle-specific buckets, and same for shadow/terrain/water etc.

//* GPU SCENE
GPU_Instance :: struct {
	model:                   GPU_Model_ID,
	transform_index:         u32,
	material_override_index: u32,
	spatial_index:           u32,
	flags:                   u32,
}

GPU_Scene :: struct {
	instances:         []GPU_Instance,
	transforms:        []Render_Transform,
	lights:            []Render_Light,
	cameras:           []Render_Camera,
	culling:           []GPU_Culling_Instance,
	visible_instances: []u32,
	indirect_commands: []GPU_Indirect_Command,
	bucket_counts:     []u32,
}
// Later the mesh-shader path gets a seperate command representation.
GPU_Indirect_Command :: struct {
	index_count:    u32,
	instance_count: u32,
	first_index:    u32,
	vertex_offset:  i32,
	first_instance: u32,
}

//* CULLING DATA
Culling_Bounds :: struct {
	centre: mth.Vec3,
	radius: f32,
	min:    mth.Vec3,
	max:    mth.Vec3,
}
Culling_Result :: enum u8 {
	Unknown,
	Visible,
	Frustum_Culled,
	Zero_Pixel_Culled,
	HiZ_Culled,
	Cone_Culled,
}
GPU_Culling_Instance :: struct {
	instance: u32,
	bounds:   Culling_Bounds,
	lod:      u16,
	result:   Culling_Result,
	flags:    u32,
}

//* LOD
LOD_Level :: struct {
	screen_size:     f32,
	mesh:            Asset_Ref,
	geometric_error: f32,
}
LOD_Set :: struct {
	levels:     []LOD_Level,
	hysteresis: f32,
}
GPU_LOD_Result :: struct {
	lod:  u16,
	mesh: GPU_Mesh_ID,
}
