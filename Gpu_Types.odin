// BF_GPU/Gpu_Types.odin
//
// GPU-side data layout types. Every struct here mirrors a GLSL struct
// under Engine/src/Modules/BF_GPU/Shaders/Includes/Common/*.glsl byte for
// byte (std430 layout). These are the values that live in GPU buffers;
// they are not what game code constructs directly. Game code produces
// Render_Transform / Render_Camera / etc. (Types.odin) which the renderer
// packs into the Gpu_* pools in Scene.odin.
//
// All multi-component math fields use BF_Math types (Mat4, Vec3, Vec4) so
// the matrices stay 16-wide / column-major, matching the shader side.

package BF_GPU

import mth "../../Core/BF_Math"

// ---------------------------------------------------------------------------
// Layout constants shared with the shader side.
// ---------------------------------------------------------------------------

PIPELINE_TRADITIONAL :: 0
PIPELINE_MESHLET     :: 1
PIPELINE_COUNT       :: 2

MATERIAL_RENDER_COUNT :: 8

INVALID_INDEX          :: u32(0xFFFFFFFF)
VIS_PIPELINE_BIT       :: 31

CHUNK_SIZE :: 32

// LOD bin thresholds, copied from Shaders/Includes/Utils/CullingUtils.glsl.
LOD_THRESHOLD_0 :: 512.0
LOD_THRESHOLD_1 :: 256.0
LOD_THRESHOLD_2 :: 128.0

// Fast-path mesh culling thresholds. Mirrors IsModelSimpleEnoughForFastPath.
MAX_FAST_PATH_MESHES    :: 8
MAX_FAST_PATH_VERTICES  :: 4096

// ---------------------------------------------------------------------------
// Visibility buffer packing. R32G32_UINT, exactly the layout the GLSL
// Visibility.glsl macros produce.
// ---------------------------------------------------------------------------

VIS_MASK_PRIMITIVE_DATA :: u32(0x000FFFFF) // 20 bits
VIS_MASK_LOD            :: u32(0x3)        //  2 bits
VIS_MASK_MESH           :: u32(0x3FF)      // 10 bits
VIS_MASK_MS_TRIANGLE    :: u32(0x7F)       //  7 bits
VIS_MASK_MS_MESHLET     :: u32(0x1FFF)     // 13 bits

VIS_SHIFT_PRIMITIVE_DATA :: 0
VIS_SHIFT_LOD            :: 20
VIS_SHIFT_MESH           :: 22
VIS_SHIFT_MS_TRIANGLE    :: 0
VIS_SHIFT_MS_MESHLET     :: 7

// pack_visibility_entity packs (pipeline_bit | entity_id) into a u32.
pack_visibility_entity :: #force_inline proc(entity_id, pipeline_flag: u32) -> u32 {
	return entity_id | ((pipeline_flag & 0x1) << VIS_PIPELINE_BIT)
}

// pack_partial_traditional packs (lod, mesh) into the partial payload.
pack_partial_traditional :: #force_inline proc(lod, mesh: u32) -> u32 {
	return ((lod & VIS_MASK_LOD) << VIS_SHIFT_LOD) | ((mesh & VIS_MASK_MESH) << VIS_SHIFT_MESH)
}

// pack_partial_mesh_shader packs (meshlet, lod, mesh) into the partial payload.
pack_partial_mesh_shader :: #force_inline proc(meshlet, lod, mesh: u32) -> u32 {
	return ((meshlet & VIS_MASK_MS_MESHLET) << VIS_SHIFT_MS_MESHLET) |
	       ((lod & VIS_MASK_LOD) << VIS_SHIFT_LOD) |
	       ((mesh & VIS_MASK_MESH) << VIS_SHIFT_MESH)
}

// finalize_vis_traditional stamps the primitive id into the partial payload.
finalize_vis_traditional :: #force_inline proc(partial, primitive_id: u32) -> u32 {
	return partial | ((primitive_id & VIS_MASK_PRIMITIVE_DATA) << VIS_SHIFT_PRIMITIVE_DATA)
}

// finalize_vis_ms stamps the meshlet triangle index into the partial payload.
finalize_vis_ms :: #force_inline proc(partial, tri_idx: u32) -> u32 {
	return partial | ((tri_idx & VIS_MASK_MS_TRIANGLE) << VIS_SHIFT_MS_TRIANGLE)
}

// ---------------------------------------------------------------------------
// Component GLSL mirrors.
// ---------------------------------------------------------------------------

// Gpu_Transform_Component mirrors Shaders/Includes/Common/Transform.glsl::TransformComponent.
Gpu_Transform_Component :: struct {
	transform:    mth.Mat4, // 64
	transform_it: mth.Mat4, // 64
}

// Gpu_Node_Transform mirrors Shaders/Includes/Common/Transform.glsl::GpuNodeTransform.
Gpu_Node_Transform :: struct {
	global_transform:    mth.Mat4,
	global_transform_it: mth.Mat4,
}

// Gpu_Transform_Model_Link mirrors Shaders/Includes/Common/Transform.glsl::TransformModelLink.
Gpu_Transform_Model_Link :: struct {
	entity_index:      u32,
	model_dense_index: u32,
}

// Gpu_Model_Component mirrors Shaders/Includes/Common/Model.glsl::ModelComponent.
// Matches std430 scalar layout (note the `#extension GL_EXT_scalar_block_layout`).
Gpu_Model_Component :: struct {
	entity_index:     u32,
	model_index:      u32,
	flags:            u32,
	material_offset:  u32,
	pipeline_offset:  u32,
}

// Gpu_Camera mirrors Shaders/Includes/Common/Camera.glsl::CameraComponent.
// 6 frustum planes packed after the matrices. All view/proj are column-major.
Gpu_Camera :: struct {
	view:              mth.Mat4,
	view_inv:          mth.Mat4,
	proj:              mth.Mat4,
	proj_inv:          mth.Mat4,
	proj_vulkan:       mth.Mat4,
	proj_vulkan_inv:   mth.Mat4,
	view_proj:         mth.Mat4,
	view_proj_inv:     mth.Mat4,
	view_proj_vulkan:  mth.Mat4,
	view_proj_vulkan_inv: mth.Mat4,
	eye:               mth.Vec4,
	params:            mth.Vec4, // (near, far, _, _)
	frustum:           [6]mth.Vec4,
}

// Gpu_Vertex_Position mirrors Shaders/Includes/Common/Mesh.glsl::GpuVertexPosition.
Gpu_Vertex_Position :: struct {
	position:     mth.Vec3,
	packed_index: u32,
}

// Gpu_Vertex_Attributes mirrors Shaders/Includes/Common/Mesh.glsl::GpuVertexAttributes.
Gpu_Vertex_Attributes :: struct {
	normal:  mth.Vec3,
	uv_x:    f32,
	tangent: mth.Vec3,
	uv_y:    f32,
}

// Gpu_Mesh_Collider mirrors Shaders/Includes/Common/Mesh.glsl::GpuMeshCollider.
Gpu_Mesh_Collider :: struct {
	center:    mth.Vec3,
	radius:    f32,
	aabb_min:  mth.Vec3,
	padding0:  f32,
	aabb_max:  mth.Vec3,
	padding1:  f32,
}

// Gpu_Meshlet_Collider mirrors Shaders/Includes/Common/Mesh.glsl::GpuMeshletCollider.
Gpu_Meshlet_Collider :: struct {
	center:    mth.Vec3,
	radius:    f32,
	aabb_min:  mth.Vec3,
	padding0:  f32,
	aabb_max:  mth.Vec3,
	padding1:  f32,
	apex:      mth.Vec3,
	cutoff:    f32,
	axis:      mth.Vec3,
	padding2:  f32,
}

// Gpu_Model_Addresses mirrors Shaders/Includes/Common/Mesh.glsl::GpuModelAddresses.
// All sub-buffers are accessed by buffer_device_address (uint64_t); they
// do not need to live in a single allocation but their addresses must be
// written into this struct so the culling and shading shaders can fetch
// them through the FrameGlobalContext.
Gpu_Model_Addresses :: struct {
	vertex_positions:         u64,
	vertex_attributes:        u64,
	indices:                  u64,
	mesh_material_indices:    u64,
	mesh_descriptors:         u64,
	mesh_colliders:           u64,
	lod_descriptors:          u64,
	meshlet_vertex_indices:   u64,
	meshlet_triangle_indices: u64,
	meshlet_descriptors:      u64,
	meshlet_draw_descriptors: u64,
	meshlet_colliders:        u64,
	node_transforms:          u64,

	is_ready:             u32,
	vertex_count:         u32,
	index_count:          u32,
	average_lod_idx_count: u32,
	mesh_count:           u32,
	padding:              u32,

	global_collider: Gpu_Mesh_Collider,
}

// Gpu_Mesh_Draw_Descriptor mirrors Shaders/Includes/Common/Mesh.glsl::MeshDrawDescriptor.
Gpu_Mesh_Draw_Descriptor :: struct {
	model_index:      u32,
	mesh_index:       u32,
	lod_index:        u32,
	instance_offset:  u32,
	max_instances:    u32,
	indirect_index:   u32,
	is_meshlet_pipe:  u32,
	padding:          u32,
}

// Gpu_Meshlet_Draw_Descriptor mirrors Shaders/Includes/Common/Mesh.glsl::GpuMeshletDrawDescriptor.
Gpu_Meshlet_Draw_Descriptor :: struct {
	meshlet_offset: u32,
	meshlet_count:  u32,
	material_index: u32,
	padding:        u32,
}

// Gpu_Meshlet_Descriptor mirrors Shaders/Includes/Common/Mesh.glsl::GpuMeshletDescriptor.
Gpu_Meshlet_Descriptor :: struct {
	vertex_indices_offset:   u32,
	vertex_count:            u32,
	triangle_indices_offset: u32,
	triangle_count:          u32,
}

// ---------------------------------------------------------------------------
// Culling pool types.
// ---------------------------------------------------------------------------

// Gpu_Visible_Model mirrors Shaders/Includes/Common/Culling.glsl::VisibleModelData.
Gpu_Visible_Model :: struct {
	entity_id:  u32,
	model_index: u32,
}

// Gpu_Model_Allocation mirrors Shaders/Includes/Common/Culling.glsl::ModelAllocationInfo.
Gpu_Model_Allocation :: struct {
	max_instances:        u32,
	mesh_alloc_offset:    u32,
	mesh_alloc_count:     u32,
	padding:              u32,
}

// Gpu_Mesh_Allocation mirrors Shaders/Includes/Common/Culling.glsl::MeshAllocationInfo.
// 4 LODs per (model, mesh) slot; 2 pipelines x 8 material render types.
Gpu_Mesh_Allocation :: struct {
	descriptor_index:    u32,
	padding:             [3]u32,
	indirect_indices:    [PIPELINE_COUNT][MATERIAL_RENDER_COUNT]u32,
	instance_offsets:    [PIPELINE_COUNT][MATERIAL_RENDER_COUNT]u32,
	active_types:        [PIPELINE_COUNT][MATERIAL_RENDER_COUNT]u32,
}

// ---------------------------------------------------------------------------
// Chunk types.
// ---------------------------------------------------------------------------

// Gpu_Static_Chunk mirrors Shaders/Includes/Common/StaticChunk.glsl::StaticChunk.
// 32 entities per chunk; bounds are world-space AABB.
Gpu_Static_Chunk :: struct {
	min_bounds:          mth.Vec3,
	first_entity_index:  u32,
	max_bounds:          mth.Vec3,
	entity_count:        u32,
}

// Gpu_Scene_AABB mirrors Shaders/Includes/Common/StaticChunk.glsl::SceneAABB.
// Stores sortable-uint packed floats (see AtomicFloatUtils.glsl).
Gpu_Scene_AABB :: struct {
	min_x: u32, min_y: u32, min_z: u32,
	max_x: u32, max_y: u32, max_z: u32,
}

// ---------------------------------------------------------------------------
// Animation types.
// ---------------------------------------------------------------------------

// Gpu_Vertex_Skin_Data mirrors Shaders/Includes/Common/Animation.glsl::GpuVertexSkinData.
Gpu_Vertex_Skin_Data :: struct {
	bone_indices: [4]u32,
	bone_weights: [4]f32,
}

// Gpu_Animation_Component mirrors Shaders/Includes/Common/Animation.glsl::AnimationComponent.
Gpu_Animation_Component :: struct {
	animation_index: u32,
	frame_index:     u32,
	padding0:        u32,
	padding1:        u32,
}

// Gpu_Animation_Descriptor mirrors Shaders/Includes/Common/Animation.glsl::GpuAnimationDescriptor.
Gpu_Animation_Descriptor :: struct {
	frame_count:          u32,
	node_count:           u32,
	global_vertex_count:  u32,
	global_mesh_count:    u32,
	global_meshlet_count: u32,
	duration_in_seconds:  f32,
	sample_rate:          f32,
	padding:              f32,
}

// Gpu_Animation_Addresses mirrors Shaders/Includes/Common/Animation.glsl::GpuAnimationAddresses.
Gpu_Animation_Addresses :: struct {
	is_ready:                 u32,
	padding:                  u32,
	vertex_skin_data:         u64,
	node_transforms:          u64,
	frame_global_colliders:   u64,
	frame_mesh_colliders:     u64,
	frame_meshlet_colliders:  u64,
	descriptor:               Gpu_Animation_Descriptor,
	global_collider:          Gpu_Mesh_Collider,
}

// ---------------------------------------------------------------------------
// Material type. Mirrors Shaders/Includes/Common/Material.glsl::Material.
// ---------------------------------------------------------------------------

// Gpu_Material mirrors the GLSL Material struct.
// 15 packed texture/sampler pairs, each a u32 (textureId in low 24 bits,
// samplerId in high 8 bits on the shader side; here we keep them as u32).
Gpu_Material :: struct {
	color:                 mth.Vec4,
	emissive_color:        mth.Vec3,
	emissive_intensity:    f32,
	uv_scale:              mth.Vec2,
	metalness:             f32,
	roughness:             f32,
	ao_strength:           f32,
	packed_flags:          u32, // bit 0 double-sided, bit 1 transparent, bit 2 alpha-tested
	clearcoat_factor:      f32,
	clearcoat_roughness:   f32,
	specular_color:        mth.Vec3,
	specular_factor:       f32,
	ior:                   f32,
	albedo_texture:        u32,
	normal_texture:        u32,
	metalness_texture:     u32,
	roughness_texture:     u32,
	metallic_roughness_texture: u32,
	emissive_texture:      u32,
	ambient_occlusion_texture: u32,
	opacity_texture:       u32,
	clearcoat_texture:     u32,
	clearcoat_roughness_texture: u32,
	clearcoat_normal_texture: u32,
	specular_texture:      u32,
	specular_color_texture: u32,
	video_texture:         u32,
	padding1:              u32,
}

// Material flag bits (packed_flags).
MATERIAL_FLAG_DOUBLE_SIDED :: 1 << 0
MATERIAL_FLAG_TRANSPARENT  :: 1 << 1
MATERIAL_FLAG_ALPHA_TESTED :: 1 << 2

// material_render_type maps the (transparent, alpha_tested, double_sided)
// tuple into the 0..7 shader-side slot.
material_render_type :: #force_inline proc(packed_flags: u32) -> u32 {
	transparent  := (packed_flags & MATERIAL_FLAG_TRANSPARENT)  != 0
	alpha_tested := (packed_flags & MATERIAL_FLAG_ALPHA_TESTED) != 0
	double_sided := (packed_flags & MATERIAL_FLAG_DOUBLE_SIDED) != 0
	switch {
	case transparent && alpha_tested: return double_sided ? 7 : 6
	case transparent:                  return double_sided ? 5 : 4
	case alpha_tested:                 return double_sided ? 3 : 2
	case:                              return double_sided ? 1 : 0
	}
}

// ---------------------------------------------------------------------------
// FrameGlobalContext. Mirrors Shaders/Includes/Common/FrameGlobalContext.glsl.
// This is the single SSBO the culling / HiZ / Morton / shading shaders
// read every frame; it contains every persistent buffer device address
// they need plus the per-frame scalar state.
//
// The host-side mirror below holds raw u64 addresses that the GPU backend
// fills from buffer creation handles. The struct layout is std430, so
// padding matches the GLSL declaration exactly.
// ---------------------------------------------------------------------------

Gpu_Frame_Global_Context :: struct {
	// --- texture / environment ---
	texture_metadata_buffer_addr: u64,
	environment_buffer_addr:      u64,

	// --- geometry dispatch / instance / draw ---
	global_draw_count_buffer_addr:           u64,
	global_instance_index_buffer_addr:       u64,
	global_indirect_command_buffer_addr:     u64,
	global_indirect_command_desc_buffer_addr: u64,
	global_model_allocation_buffer_addr:     u64,
	global_mesh_allocation_buffer_addr:      u64,

	// --- camera ---
	camera_visible_index_buffer_addr: u64,
	camera_buffer_addr:               u64,
	camera_sparse_map_buffer_addr:    u64,

	// --- tag ---
	tag_sparse_map_buffer_addr: u64,
	tag_data_buffer_addr:       u64,

	// --- transform ---
	transform_buffer_addr:           u64,
	transform_sparse_map_buffer_addr: u64,
	transform_model_link_buffer_addr: u64,

	// --- static chunks ---
	static_chunk_data_buffer_addr:          u64,
	static_chunk_visible_index_buffer_addr: u64,
	static_chunk_count_buffer_addr:         u64,

	// --- model ---
	model_address_buffer_addr:        u64,
	model_buffer_addr:                u64,
	model_sparse_map_buffer_addr:     u64,
	model_count_buffer_addr:          u64,
	model_visible_index_buffer_addr:  u64,

	// --- animation ---
	animation_address_buffer_addr:    u64,
	animation_buffer_addr:            u64,
	animation_sparse_map_buffer_addr: u64,

	// --- material / pipeline lookup ---
	material_lookup_buffer_addr: u64,
	material_buffer_addr:        u64,
	pipeline_lookup_buffer_addr:  u64,

	// --- scene-level buffers ---
	scene_aabb_buffer_addr:                u64,
	morton_keys_buffer_addr:               u64,
	morton_values_buffer_addr:             u64,
	morton_chunk_data_buffer_addr:         u64,
	morton_chunk_indirect_dispatch_addr:   u64,
	morton_chunk_indirect_draw_addr:       u64,
	morton_chunk_visible_indirect_dispatch_addr: u64,
	morton_chunk_visible_index_buffer_addr: u64,
	morton_chunk_transforms_index_buffer_addr: u64,

	// --- scalar state ---
	screen_width:    f32,
	screen_height:   f32,
	ambient_strength: f32,
	emissive_strength: f32,
	alpha_limit_discard: f32,

	// --- enable flags (1 = on, 0 = off) ---
	enable_meshlet_cone_culling:    u32,
	enable_chunk_frustum_culling:   u32,
	enable_model_frustum_culling:   u32,
	enable_mesh_frustum_culling:    u32,
	enable_meshlet_frustum_culling: u32,
	enable_chunk_occlusion_culling: u32,
	enable_model_occlusion_culling: u32,
	enable_mesh_occlusion_culling:  u32,
	enable_meshlet_occlusion_culling: u32,

	global_indirect_command_count:    u32,
	global_traditional_commands_count: u32,
	global_meshlet_commands_count:     u32,

	main_camera_entity:  u32,
	active_camera_entity: u32,

	static_chunk_count: u32,
	model_count:        u32,

	all_transform_count:     u32,
	static_transform_count:  u32,
	dynamic_transform_count: u32,
	stream_transform_count:  u32,
	non_static_transform_count: u32,

	tile_size:        u32,
	tile_count_x:     u32,
	tile_count_y:     u32,
	hiz_mip_level:    f32,
	slice_scale_factor: f32,

	active_environment_index: u32,
	brdf_lut_texture_index:   u32,
}

// ---------------------------------------------------------------------------
// Task payload. Mirrors Shaders/Includes/Payload/TaskPayload.glsl.
// Carried through task/mesh shader stages when MeshShaders == true.
// ---------------------------------------------------------------------------

Gpu_Task_Payload :: struct {
	draw_id:           u32,
	entity_id:         u32,
	transform_dense_idx: u32,
	active_camera_dense_idx: u32,
	model_dense_index:  u32,
	meshlet_indices:    [32]u32,
}

// ---------------------------------------------------------------------------
// Push constants. Mirrors Shaders/Includes/PushConstants/*.glsl structs.
// These are uploaded via vkCmdPushConstants at pass-record time; the host
// side writes the FrameGlobalContext device address (and a few scalar
// fields) into them right before each dispatch.
// ---------------------------------------------------------------------------

PC_Frame_Context_Only :: struct {
	frame_global_context_buffer_addr: u64,
}

PC_Hiz_Linearize_Depth :: struct {
	frame_global_context_buffer_addr: u64,
	out_image_size:                   mth.Vec2,
}

PC_Hiz_Down_Sample :: struct {
	in_image_size:  mth.Vec2,
	out_image_size: mth.Vec2,
}

PC_Traditional_Meshlet_Pass :: struct {
	frame_global_context_buffer_addr: u64,
	base_descriptor_offset:           u32,
	material_render_type:             u32,
	disable_cone_culling:             u32,
}

// ---------------------------------------------------------------------------
// Frame-level settings consumed at startup.
//
// `mesh_shaders` is NOT a project setting - it is derived at init time
// from whether the BF_GPU_Meshlet extension has registered a meshlet
// pipeline with the extension point. Renderer_Settings has no
// MeshShaders field; mesh-shader support is governed entirely by the
// project's [[extensions]] table.
// ---------------------------------------------------------------------------

GPU_Runtime_Settings :: struct {
	mesh_shaders:        bool, // auto-detected from BF_GPU_Meshlet extension presence
	hiz_occlusion:       bool,
	frustum_culling:     bool,
	cone_culling:        bool,
	async_compute_cull:  bool,
	gpu_sort_extension:  GPU_Sort_Extension,
	lod_count:           u32,
}

GPU_Sort_Extension :: enum u8 {
	None,
	VK_KHR_Shader_Subgroup_Sort,
	AMD_Shader_Subgroup_Sort,
}

// ---------------------------------------------------------------------------
// Pass + material descriptors extensions hand to BF_GPU's
// Renderer_Extension_Point. These are the public ABI for extensions;
// BF_GPU reads them at attach time and ignores any contribution whose
// descriptor it does not understand.
// ---------------------------------------------------------------------------

Write_Target :: enum u8 {
	None,
	Visibility_Buffer,
	GBuffer_Albedo_Metal_Rough_AO,
	GBuffer_Normal,
	GBuffer_Emissive,
	GBuffer_Motion,
	HDR_Color,
	Depth,
	Shadow_Atlas,
	Custom, // extensions may add custom targets with their own descriptors
}

Graphics_Pass_Descriptor :: struct {
	name:              cstring,
	vertex_shader:     cstring, // path to .vert / .task / nil
	fragment_shader:   cstring, // path to .frag / nil
	mesh_shader:       cstring, // path to .mesh / nil (for meshlet paths)
	indirect_buffer:   Gpu_Buffer_Kind,
	slot_count:        u32,
	write_target:      Write_Target,
	reads_depth:       bool,
}

Compute_Pass_Descriptor :: struct {
	name:           cstring,
	shader:         cstring, // path to .comp
	dispatch_buffer: Gpu_Buffer_Kind,
	reads:          bit_set[Compute_Read_Target],
	writes:         bit_set[Compute_Write_Target],
}
Compute_Read_Target :: enum u8 {
	Depth,
	HiZ,
	Visibility_Buffer,
	Indirect_Commands,
	Visible_Lists,
}
Compute_Write_Target :: enum u8 {
	HiZ,
	Indirect_Commands,
	Visible_Lists,
	Morton_Keys,
	Scene_AABB,
}

Meshlet_Pipeline_Descriptor :: struct {
	name:                cstring,
	task_shader:         cstring, // .task path
	mesh_shader:         cstring, // .mesh path
	max_meshlets_per_wg: u32,
	supports_cull:       bool,
}

Material_Descriptor :: struct {
	name:               cstring,
	vertex_shader:      cstring,
	fragment_shader:    cstring,
	domain:             u32, // 0 geometry, 1 post-process, ...
	reads_visibility:   bool,
}

Material_Instance_Descriptor :: struct {
	name:          cstring,
	parent:        cstring,
	overrides_ptr: rawptr, // backend-defined override blob
}
