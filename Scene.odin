// BF_GPU/Scene.odin
//
// Render_Scene -> GPU_Scene.
//
// GPU_Scene is the *derived* GPU representation of Render_Scene. It owns the
// host-side mirror of every buffer the culling pipeline reads: dense SoA pools
// (uploaded verbatim by the GPU backend) and the (entity -> dense index)
// sparse maps the shaders translate through. It never owns ECS state and never
// re-derives anything from the ECS - Extraction.odin already did that.
//
// The pipeline is:
//
//   BF_ECS -> Render_Scene -> GPU_Scene -> GPU culling -> indirect rendering
//
// Slot parity: GPU_Scene dense index == Render_Scene slot index. That keeps
// the update path a direct scatter (no remapping table) and lets per-slot GPU
// state such as previous-frame transforms stay valid while an entity keeps
// contributing.
//
// The update path is incremental: it consumes the added / updated / removed
// change sets Render_Scene produced for the frame instead of rebuilding the
// pools. Only the per-frame scalars, camera pool and bucket counts are
// recomputed wholesale, because they are frame-global by nature.

package BF_GPU

import "base:runtime"
import mth "../../Core/BF_Math"

// ---------------------------------------------------------------------------
// Dense pools. One SoA slab per component kind; the GPU backend uploads
// these each frame as the matching GLSL buffer.
// ---------------------------------------------------------------------------

Transform_Pool :: struct {
	data:       [dynamic]Gpu_Transform_Component,
	dense_used: u32,
}

Model_Pool :: struct {
	data:       [dynamic]Gpu_Model_Component,
	dense_used: u32,
}

Camera_Pool :: struct {
	data:       [dynamic]Gpu_Camera,
	dense_used: u32,
}

Material_Pool :: struct {
	data:       [dynamic]Gpu_Material,
	dense_used: u32,
}

// Per-entity render state the shaders read through the tag sparse map. This
// replaces the placeholder "tag" component the renderer used to invent: the
// data is the packed Render_Instance_Flags, which are themselves derived from
// Render_Model + Spatial_State.
Tag_Pool :: struct {
	data:       [dynamic]u32,
	dense_used: u32,
}

// Material lookup table: per (model, mesh) -> dense material index.
// Lives in its own small buffer so the culling shaders can fetch by
// materialOffset + meshIndex without indirection into the material pool.
Material_Lookup :: struct {
	data:       [dynamic]u32,
	dense_used: u32,
}

// Pipeline lookup table: per (model, mesh) -> 0 (traditional) or 1 (meshlet).
// When MeshShaders == false all entries are forced to 0 at startup.
Pipeline_Lookup :: struct {
	data:       [dynamic]u32,
	dense_used: u32,
}

// Mesh allocation info: per (model, mesh, lod) -> the
// Gpu_Mesh_Allocation table. The culling shaders index this via
// (meshAllocOffset + mesh * 4 + lod).
Model_Allocation_Pool :: struct {
	data:       [dynamic]Gpu_Model_Allocation,
	dense_used: u32,
}

Mesh_Allocation_Pool :: struct {
	data:       [dynamic]Gpu_Mesh_Allocation,
	dense_used: u32,
}

// MeshDrawDescriptor pool. Each entry is the per-draw-call constant data
// that Traditional.vert / Meshlet.task read via gl_DrawID.
Mesh_Draw_Descriptor_Pool :: struct {
	data:       [dynamic]Gpu_Mesh_Draw_Descriptor,
	dense_used: u32,
}

// Model address pool: per GPU_Model_ID -> Gpu_Model_Addresses.
Model_Address_Pool :: struct {
	data:       [dynamic]Gpu_Model_Addresses,
	dense_used: u32,
}

// ---------------------------------------------------------------------------
// Sparse maps. entity_id -> dense_index (or INVALID_INDEX). Maintained by the
// GPU_Scene update path from Render_Scene's change sets; queried by the
// culling shaders via the matching FrameGlobalContext sparse_map field.
//
// Implemented as two parallel arrays so the GPU can read them as a
// uint32[] buffer with no per-entry stride or padding.
// ---------------------------------------------------------------------------

Sparse_Map :: struct {
	entity_to_dense:   [dynamic]u32, // entity index -> dense index
	dense_to_entity:   [dynamic]u32, // dense index -> entity index (CPU only)
	live_entity_count: u32,
}

Scene_Sparse_Maps :: struct {
	transforms: Sparse_Map,
	models:     Sparse_Map,
	cameras:    Sparse_Map,
	tags:       Sparse_Map,
}

// ---------------------------------------------------------------------------
// GPU_Scene. One per renderer; lives for the module lifetime.
// ---------------------------------------------------------------------------

GPU_Scene :: struct {
	allocator:                   runtime.Allocator,
	settings:                    GPU_Runtime_Settings,

	// Slot-parallel pools (dense index == Render_Scene slot).
	transforms:                  Transform_Pool,
	models:                      Model_Pool,
	tags:                        Tag_Pool,
	instances:                   [dynamic]GPU_Instance,
	culling:                     [dynamic]GPU_Culling_Instance,

	// Frame-global pools.
	cameras:                     Camera_Pool,
	materials:                   Material_Pool,
	material_lookup:             Material_Lookup,
	pipeline_lookup:             Pipeline_Lookup,
	model_allocations:           Model_Allocation_Pool,
	mesh_allocations:            Mesh_Allocation_Pool,
	draw_descriptors:            Mesh_Draw_Descriptor_Pool,
	model_addresses:             Model_Address_Pool,
	visible_instances:           [dynamic]u32,
	indirect_commands:           [dynamic]GPU_Indirect_Command,

	// Chunk-scoped instance ranges, mirroring Render_Scene.chunks. The GPU
	// side consumes them as the static-chunk buffer; the CPU side keeps the
	// flattened instance list they index.
	static_chunks:               [dynamic]Gpu_Static_Chunk,
	chunk_instances:             [dynamic]u32,

	sparse:                      Scene_Sparse_Maps,

	// Default geometry bucket set: (pipeline, material class, cull mode).
	bucket_counts:               [GEOMETRY_BUCKET_COUNT]u32,

	// Per-frame counts surfaced to the FrameGlobalContext scalar fields.
	frame_main_camera:           u32,
	frame_active_camera:         u32,
	frame_traditional_cmd_count: u32,
	frame_meshlet_cmd_count:     u32,
	frame_indirect_cmd_count:    u32,
	frame_static_chunk_count:    u32,
	frame_static_transforms:     u32,
	frame_all_transforms:        u32,
	frame_model_count:           u32,
	frame_pending_assets:        u32,

	// Frame number (debugging only).
	frame_idx:                   u64,
}

//* Lifecycle.

DEFAULT_POOL_CAPACITY :: 1024
DEFAULT_SPARSE_CAPACITY :: 65_536

gpu_scene_init :: proc(
	gpu: ^GPU_Scene,
	settings: GPU_Runtime_Settings,
	allocator := context.allocator,
) {
	if gpu == nil do return
	gpu.allocator = allocator
	gpu.settings = settings
	gpu.frame_idx = 0
	gpu.frame_main_camera = INVALID_INDEX
	gpu.frame_active_camera = INVALID_INDEX

	gpu.transforms.data = make([dynamic]Gpu_Transform_Component, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.models.data = make([dynamic]Gpu_Model_Component, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.tags.data = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.instances = make([dynamic]GPU_Instance, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.culling = make([dynamic]GPU_Culling_Instance, 0, DEFAULT_POOL_CAPACITY, allocator)

	gpu.cameras.data = make([dynamic]Gpu_Camera, 0, 16, allocator)
	gpu.materials.data = make([dynamic]Gpu_Material, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.material_lookup.data = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.pipeline_lookup.data = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.model_allocations.data = make([dynamic]Gpu_Model_Allocation, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.mesh_allocations.data = make([dynamic]Gpu_Mesh_Allocation, 0, DEFAULT_POOL_CAPACITY * 4, allocator)
	gpu.draw_descriptors.data = make([dynamic]Gpu_Mesh_Draw_Descriptor, 0, DEFAULT_POOL_CAPACITY * 4, allocator)
	gpu.model_addresses.data = make([dynamic]Gpu_Model_Addresses, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.visible_instances = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.indirect_commands = make([dynamic]GPU_Indirect_Command, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.static_chunks = make([dynamic]Gpu_Static_Chunk, 0, 64, allocator)
	gpu.chunk_instances = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)

	gpu.sparse.transforms.entity_to_dense = make([dynamic]u32, 0, DEFAULT_SPARSE_CAPACITY, allocator)
	gpu.sparse.transforms.dense_to_entity = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.sparse.models.entity_to_dense = make([dynamic]u32, 0, DEFAULT_SPARSE_CAPACITY, allocator)
	gpu.sparse.models.dense_to_entity = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	gpu.sparse.cameras.entity_to_dense = make([dynamic]u32, 0, DEFAULT_SPARSE_CAPACITY, allocator)
	gpu.sparse.cameras.dense_to_entity = make([dynamic]u32, 0, 16, allocator)
	gpu.sparse.tags.entity_to_dense = make([dynamic]u32, 0, DEFAULT_SPARSE_CAPACITY, allocator)
	gpu.sparse.tags.dense_to_entity = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
}

gpu_scene_destroy :: proc(gpu: ^GPU_Scene) {
	if gpu == nil || gpu.allocator.procedure == nil do return
	delete(gpu.transforms.data)
	delete(gpu.models.data)
	delete(gpu.tags.data)
	delete(gpu.instances)
	delete(gpu.culling)

	delete(gpu.cameras.data)
	delete(gpu.materials.data)
	delete(gpu.material_lookup.data)
	delete(gpu.pipeline_lookup.data)
	delete(gpu.model_allocations.data)
	delete(gpu.mesh_allocations.data)
	delete(gpu.draw_descriptors.data)
	delete(gpu.model_addresses.data)
	delete(gpu.visible_instances)
	delete(gpu.indirect_commands)
	delete(gpu.static_chunks)
	delete(gpu.chunk_instances)

	delete(gpu.sparse.transforms.entity_to_dense)
	delete(gpu.sparse.transforms.dense_to_entity)
	delete(gpu.sparse.models.entity_to_dense)
	delete(gpu.sparse.models.dense_to_entity)
	delete(gpu.sparse.cameras.entity_to_dense)
	delete(gpu.sparse.cameras.dense_to_entity)
	delete(gpu.sparse.tags.entity_to_dense)
	delete(gpu.sparse.tags.dense_to_entity)
	gpu^ = {}
}

gpu_scene_is_valid :: #force_inline proc(gpu: ^GPU_Scene) -> bool {
	return gpu != nil && gpu.allocator.procedure != nil
}

// ---------------------------------------------------------------------------
// Sparse map maintenance.
// ---------------------------------------------------------------------------

sparse_map_clear :: proc(m: ^Sparse_Map) {
	for i in 0 ..< len(m.entity_to_dense) {
		m.entity_to_dense[i] = INVALID_INDEX
	}
	for i in 0 ..< len(m.dense_to_entity) {
		m.dense_to_entity[i] = INVALID_INDEX
	}
	m.live_entity_count = 0
}

sparse_map_reserve_entity :: proc(m: ^Sparse_Map, entity: u32) {
	for len(m.entity_to_dense) <= int(entity) {
		append(&m.entity_to_dense, INVALID_INDEX)
	}
}

sparse_map_grow_dense :: proc(m: ^Sparse_Map, dense: u32, entity: u32) {
	for len(m.dense_to_entity) <= int(dense) {
		append(&m.dense_to_entity, INVALID_INDEX)
	}
	m.dense_to_entity[dense] = entity
}

// Bind entity <-> dense in both directions.
sparse_map_assign :: proc(m: ^Sparse_Map, entity, dense: u32) {
	sparse_map_reserve_entity(m, entity)
	if m.entity_to_dense[entity] == INVALID_INDEX {
		m.live_entity_count += 1
	}
	m.entity_to_dense[entity] = dense
	sparse_map_grow_dense(m, dense, entity)
}

// Release a (entity, dense) pair. The dense guard keeps a stale removal from
// clobbering a slot that has already been handed to another entity.
sparse_map_release :: proc(m: ^Sparse_Map, entity, dense: u32) {
	if int(dense) < len(m.dense_to_entity) {
		if m.dense_to_entity[dense] != entity do return
		m.dense_to_entity[dense] = INVALID_INDEX
	}
	if int(entity) < len(m.entity_to_dense) && m.entity_to_dense[entity] != INVALID_INDEX {
		m.entity_to_dense[entity] = INVALID_INDEX
		if m.live_entity_count > 0 do m.live_entity_count -= 1
	}
}

sparse_map_dense :: #force_inline proc(m: ^Sparse_Map, entity: u32) -> u32 {
	if int(entity) >= len(m.entity_to_dense) do return INVALID_INDEX
	return m.entity_to_dense[entity]
}

// ---------------------------------------------------------------------------
// Render_Scene -> GPU_Scene update.
// ---------------------------------------------------------------------------

// gpu_scene_update applies one frame of Render_Scene change sets. `store` is
// needed only to classify buckets (the material a slot draws with); the GPU
// resource ids themselves were already resolved during extraction.
gpu_scene_update :: proc(
	gpu: ^GPU_Scene,
	scene: ^Render_Scene,
	store: ^GPU_Resource_Store,
) -> bool {
	if !gpu_scene_is_valid(gpu) || !render_scene_is_valid(scene) do return false

	gpu_scene_reserve(gpu, len(scene.instances))

	// Removals first: a slot retired this frame cannot be reused before the
	// next extraction, so clearing before writing is always safe and keeps a
	// stale entity out of the sparse maps.
	for removal in scene.removed {
		gpu_scene_clear_slot(gpu, removal)
	}
	for id in scene.added {
		gpu_scene_write_slot(gpu, scene, id)
	}
	for id in scene.updated {
		gpu_scene_write_slot(gpu, scene, id)
	}

	gpu_scene_update_chunks(gpu, scene)
	gpu_scene_update_cameras(gpu, scene)
	gpu_scene_update_counts(gpu, scene, store)

	gpu.frame_idx = scene.frame_index
	return true
}

// Grows every slot-parallel pool so that `slots` entries exist. New entries
// start out empty (model_index == INVALID_INDEX) so the shaders skip them.
gpu_scene_reserve :: proc(gpu: ^GPU_Scene, slots: int) {
	for len(gpu.transforms.data) < slots {
		append(&gpu.transforms.data, Gpu_Transform_Component{})
	}
	for len(gpu.models.data) < slots {
		append(&gpu.models.data, gpu_model_component_empty())
	}
	for len(gpu.tags.data) < slots {
		append(&gpu.tags.data, 0)
	}
	for len(gpu.instances) < slots {
		append(&gpu.instances, GPU_Instance{})
	}
	for len(gpu.culling) < slots {
		append(&gpu.culling, GPU_Culling_Instance{})
	}
	gpu.transforms.dense_used = u32(slots)
	gpu.models.dense_used = u32(slots)
	gpu.tags.dense_used = u32(slots)
}

@(private = "file")
gpu_model_component_empty :: proc() -> Gpu_Model_Component {
	return Gpu_Model_Component {
		entity_index = INVALID_INDEX,
		model_index = INVALID_INDEX,
		flags = 0,
		material_offset = INVALID_INDEX,
		pipeline_offset = 0,
	}
}

@(private = "file")
gpu_scene_write_slot :: proc(gpu: ^GPU_Scene, scene: ^Render_Scene, id: Render_Instance_ID) {
	slot := render_instance_index(id)
	if int(slot) >= len(scene.instances) || int(slot) >= len(gpu.instances) do return

	instance := scene.instances[slot]
	if .Live not_in instance.flags do return

	transform := scene.transforms[slot]
	spatial := scene.spatial[slot]
	entity_index := u32(instance.entity.ix)
	packed := gpu_pack_instance_flags(instance.flags)

	gpu.transforms.data[slot] = Gpu_Transform_Component {
		transform    = transform.world,
		transform_it = mat4_from_mat3(transform.normal),
	}
	gpu.models.data[slot] = Gpu_Model_Component {
		entity_index    = entity_index,
		model_index     = u32(instance.gpu_model),
		flags           = packed,
		material_offset = instance.material_override_index,
		pipeline_offset = gpu.settings.mesh_shaders ? PIPELINE_MESHLET : PIPELINE_TRADITIONAL,
	}
	gpu.tags.data[slot] = packed
	gpu.instances[slot] = GPU_Instance {
		model                   = instance.gpu_model,
		transform_index         = slot,
		material_override_index = instance.material_override_index,
		spatial_index           = slot,
		flags                   = packed,
	}
	gpu.culling[slot] = GPU_Culling_Instance {
		instance = slot,
		bounds   = aabb_to_culling_bounds(spatial.bounds),
		lod      = spatial.lod,
		result   = .Unknown,
		flags    = packed,
	}

	sparse_map_assign(&gpu.sparse.transforms, entity_index, slot)
	sparse_map_assign(&gpu.sparse.models, entity_index, slot)
	sparse_map_assign(&gpu.sparse.tags, entity_index, slot)
}

@(private = "file")
gpu_scene_clear_slot :: proc(gpu: ^GPU_Scene, removal: Render_Removal) {
	slot := render_instance_index(removal.instance)
	if int(slot) >= len(gpu.instances) do return
	entity_index := u32(removal.entity.ix)

	gpu.transforms.data[slot] = {}
	gpu.models.data[slot] = gpu_model_component_empty()
	gpu.tags.data[slot] = 0
	gpu.instances[slot] = {}
	gpu.culling[slot] = {}

	sparse_map_release(&gpu.sparse.transforms, entity_index, slot)
	sparse_map_release(&gpu.sparse.models, entity_index, slot)
	sparse_map_release(&gpu.sparse.tags, entity_index, slot)
}

// The chunk set the renderer received this frame, mirrored into the GPU
// static-chunk buffer. `chunk_instances` is the flattened per-chunk instance
// list the chunk entries index into.
@(private = "file")
gpu_scene_update_chunks :: proc(gpu: ^GPU_Scene, scene: ^Render_Scene) {
	clear(&gpu.static_chunks)
	clear(&gpu.chunk_instances)

	for instance_id in scene.chunk_instances {
		append(&gpu.chunk_instances, render_instance_index(instance_id))
	}
	for chunk in scene.chunks {
		append(
			&gpu.static_chunks,
			Gpu_Static_Chunk {
				min_bounds = chunk.bounds.min,
				first_entity_index = chunk.first_instance,
				max_bounds = chunk.bounds.max,
				entity_count = chunk.instance_count,
			},
		)
	}
	gpu.frame_static_chunk_count = u32(len(gpu.static_chunks))
}

@(private = "file")
gpu_scene_update_cameras :: proc(gpu: ^GPU_Scene, scene: ^Render_Scene) {
	clear(&gpu.cameras.data)
	sparse_map_clear(&gpu.sparse.cameras)

	for camera, index in scene.cameras {
		append(&gpu.cameras.data, gpu_camera_from_render_camera(camera))
		sparse_map_assign(&gpu.sparse.cameras, u32(camera.entity.ix), u32(index))
	}
	gpu.cameras.dense_used = u32(len(gpu.cameras.data))
	gpu.frame_main_camera = scene.main_camera
	gpu.frame_active_camera = scene.active_camera
}

@(private = "file")
gpu_camera_from_render_camera :: proc(camera: Render_Camera) -> Gpu_Camera {
	out: Gpu_Camera
	out.view = camera.view
	out.proj = camera.projection_matrix
	out.view_proj = camera.view_projection_matrix
	out.proj_vulkan = mat4_y_flip(camera.projection_matrix)
	out.view_proj_vulkan = mat4_mul(out.proj_vulkan, camera.view)

	if inverse, ok := mat4_affine_inverse(camera.view); ok {
		out.view_inv = inverse
		out.eye = mth.Vec4{inverse[12], inverse[13], inverse[14], 1}
	}
	out.params = mth.Vec4{camera.near_plane, camera.far_plane, 0, 0}
	out.frustum = frustum_planes_from_view_proj(camera.view_projection_matrix)
	return out
}

// Frame-global counters and the default geometry bucket set. Both are whole
// -scene properties, so they are recomputed from the live slots rather than
// tracked incrementally.
@(private = "file")
gpu_scene_update_counts :: proc(
	gpu: ^GPU_Scene,
	scene: ^Render_Scene,
	store: ^GPU_Resource_Store,
) {
	for i in 0 ..< GEOMETRY_BUCKET_COUNT {
		gpu.bucket_counts[i] = 0
	}

	traditional: u32 = 0
	meshlet: u32 = 0
	for instance in scene.instances {
		if .Live not_in instance.flags do continue
		if .Pending_Asset in instance.flags do continue
		key := gpu_scene_bucket_key(gpu, store, instance)
		index := render_bucket_index(key)
		if int(index) < GEOMETRY_BUCKET_COUNT {
			gpu.bucket_counts[index] += 1
		}
		if key.pipeline == .Mesh {
			meshlet += 1
		} else {
			traditional += 1
		}
	}

	gpu.frame_traditional_cmd_count = traditional
	gpu.frame_meshlet_cmd_count = meshlet
	gpu.frame_indirect_cmd_count = traditional + meshlet
	gpu.frame_model_count = scene.live_count
	gpu.frame_all_transforms = scene.live_count
	gpu.frame_static_transforms = scene.static_count
	gpu.frame_pending_assets = scene.pending_asset_count
}

// The bucket an instance lands in. Material class and cull mode come from the
// resolved GPU material's flags; the pipeline comes from the renderer's
// mesh-shader configuration.
gpu_scene_bucket_key :: proc(
	gpu: ^GPU_Scene,
	store: ^GPU_Resource_Store,
	instance: Render_Instance,
) -> Render_Bucket_Key {
	key := Render_Bucket_Key {
		domain         = .Geometry,
		pipeline       = gpu.settings.mesh_shaders ? .Mesh : .Traditional,
		material_class = .Opaque,
		cull_mode      = .Back,
	}
	material_id := gpu_instance_material(store, instance)
	material := gpu_material_get(store, material_id)
	if material != nil {
		if material.flags & MATERIAL_FLAG_TRANSPARENT != 0 {
			key.material_class = .Transparent
		}
		if material.flags & MATERIAL_FLAG_DOUBLE_SIDED != 0 {
			key.cull_mode = .None
		}
	}
	return key
}

gpu_pack_instance_flags :: proc(flags: Render_Instance_Flags) -> u32 {
	packed: u32 = 0
	if .Live in flags do packed |= GPU_INSTANCE_FLAG_LIVE
	if .Cast_Shadow in flags do packed |= GPU_INSTANCE_FLAG_CAST_SHADOW
	if .Receive_Shadow in flags do packed |= GPU_INSTANCE_FLAG_RECEIVE_SHADOW
	if .Static in flags do packed |= GPU_INSTANCE_FLAG_STATIC
	if .Dynamic in flags do packed |= GPU_INSTANCE_FLAG_DYNAMIC
	if .Pending_Asset in flags do packed |= GPU_INSTANCE_FLAG_PENDING_ASSET
	return packed
}

// ---------------------------------------------------------------------------
// Asset upload helpers.
//
// Called when a model / material / mesh is loaded or unloaded. These
// append to the matching pool, update the sparse map, and (for meshes)
// emit per-(model, mesh, lod) Gpu_Mesh_Allocation entries with the
// correct active_types bits per the MeshShaders flag.
// ---------------------------------------------------------------------------

// register_gpu_model_addresses reserves the address-pool slot for a
// GPU_Model_ID. The caller fills in the addresses afterwards (the sub-buffer
// addresses aren't known until the GPU backend creates them).
register_gpu_model_addresses :: proc(gpu: ^GPU_Scene, model: GPU_Model_ID) -> ^Gpu_Model_Addresses {
	if !gpu_scene_is_valid(gpu) || model == GPU_MODEL_INVALID do return nil
	for len(gpu.model_addresses.data) <= int(model) {
		append(&gpu.model_addresses.data, Gpu_Model_Addresses{})
	}
	gpu.model_addresses.dense_used = u32(len(gpu.model_addresses.data))
	return &gpu.model_addresses.data[u32(model)]
}

// register_gpu_material appends a material and returns the dense index.
register_gpu_material :: proc(gpu: ^GPU_Scene, material: Gpu_Material) -> u32 {
	append(&gpu.materials.data, material)
	gpu.materials.dense_used = u32(len(gpu.materials.data))
	return gpu.materials.dense_used - 1
}

// register_mesh_allocation appends one Gpu_Mesh_Allocation slot. The
// active_types bits are zeroed for the meshlet pipeline when MeshShaders
// is disabled, so the culling shaders skip the meshlet slots.
register_mesh_allocation :: proc(gpu: ^GPU_Scene, base: Gpu_Mesh_Allocation) -> u32 {
	alloc := base
	if !gpu.settings.mesh_shaders {
		// Zero the meshlet row of active_types so culling shaders do not
		// atomicAdd into meshlet slots.
		for k in 0 ..< MATERIAL_RENDER_COUNT {
			alloc.active_types[PIPELINE_MESHLET][k] = 0
		}
	}
	append(&gpu.mesh_allocations.data, alloc)
	gpu.mesh_allocations.dense_used = u32(len(gpu.mesh_allocations.data))
	return gpu.mesh_allocations.dense_used - 1
}

// register_draw_descriptor appends a Gpu_Mesh_Draw_Descriptor and
// returns the assigned descriptor index (which becomes gl_DrawID at
// shading time).
register_draw_descriptor :: proc(gpu: ^GPU_Scene, descriptor: Gpu_Mesh_Draw_Descriptor) -> u32 {
	append(&gpu.draw_descriptors.data, descriptor)
	gpu.draw_descriptors.dense_used = u32(len(gpu.draw_descriptors.data))
	return gpu.draw_descriptors.dense_used - 1
}

// register_pipeline_lookup writes the pipeline index for a (model, mesh)
// slot: 0 traditional, 1 meshlet. When MeshShaders is false the entry
// is forced to 0 regardless of the input.
register_pipeline_lookup :: proc(gpu: ^GPU_Scene, mesh_index: u32, pipeline: u32) {
	for len(gpu.pipeline_lookup.data) <= int(mesh_index) {
		append(&gpu.pipeline_lookup.data, 0)
	}
	gpu.pipeline_lookup.data[mesh_index] = gpu.settings.mesh_shaders ? pipeline : 0
	gpu.pipeline_lookup.dense_used = u32(len(gpu.pipeline_lookup.data))
}

// register_material_lookup writes the material index for a (model, mesh)
// slot. The culling shaders read materialOffset + meshIndex into this
// table to resolve which material the mesh uses.
register_material_lookup :: proc(gpu: ^GPU_Scene, mesh_index, material_index: u32) {
	for len(gpu.material_lookup.data) <= int(mesh_index) {
		append(&gpu.material_lookup.data, 0)
	}
	gpu.material_lookup.data[mesh_index] = material_index
	gpu.material_lookup.dense_used = u32(len(gpu.material_lookup.data))
}
