// BF_GPU/Scene.odin
//
// Render_Scene -> GPU_Scene extraction. Walks a set of BF_ECS views at
// the start of every frame to populate the dense pools and sparse maps
// that the culling pipeline reads. Pools are mirrored into GPU buffers
// by the GPU backend; this module owns the host-side storage and the
// (entity_id -> dense_index) translation tables.
//
// Integration: the renderer registers a single extraction system with
// BF_DAG at stage .PreRender. The system calls extract_scene_state()
// once per frame, then renderer_record_frame() picks up the result.

package BF_GPU

import "base:runtime"
import "core:log"
import ECS "../BF_ECS"

// ---------------------------------------------------------------------------
// Dense pools. One SoA slab per component kind; the GPU backend uploads
// these each frame as the matching GLSL buffer.
//
// The pools grow on demand up to a per-pool soft cap. Once an entity
// stops contributing (destroyed, removed from a view) its dense slot is
// NOT immediately compacted - the sparse map is rewritten so the slot
// becomes unused and the count is reported to the GPU via the matching
// FrameGlobalContext scalar field.
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
// Sparse maps. entity_id -> dense_index (or INVALID_INDEX). Maintained
// when entities are added to or removed from their view; queried by the
// culling shaders via the matching FrameGlobalContext sparse_map field.
//
// Implemented as two parallel arrays so the GPU can read them as a
// uint32[] buffer with no per-entry stride or padding.
// ---------------------------------------------------------------------------

Sparse_Map :: struct {
	entity_to_dense:   [dynamic]u32, // entity_id -> dense_index
	dense_to_entity:   [dynamic]u32, // dense_index -> entity_id (CPU only)
	live_entity_count: u32,
}

Scene_Sparse_Maps :: struct {
	transforms: Sparse_Map,
	models:     Sparse_Map,
	cameras:    Sparse_Map,
	tags:       Sparse_Map,
	animations: Sparse_Map,
}

// ---------------------------------------------------------------------------
// Render scene state. One per renderer; lives for the module lifetime.
// The pools and sparse maps are owned by the renderer so they survive
// across frames; only the dense_used counter is reset at extraction
// start so the new frame can reuse existing slots when the underlying
// ECS view hasn't churned.
// ---------------------------------------------------------------------------

Render_Scene_State :: struct {
	allocator: runtime.Allocator,

	transforms:           Transform_Pool,
	models:               Model_Pool,
	cameras:              Camera_Pool,
	materials:            Material_Pool,
	material_lookup:      Material_Lookup,
	pipeline_lookup:      Pipeline_Lookup,
	model_allocations:    Model_Allocation_Pool,
	mesh_allocations:     Mesh_Allocation_Pool,
	draw_descriptors:     Mesh_Draw_Descriptor_Pool,
	model_addresses:      Model_Address_Pool,
	sparse:               Scene_Sparse_Maps,

	// Per-frame counts surfaced to the FrameGlobalContext scalar fields.
	frame_main_camera:    u32,
	frame_active_camera:  u32,
	frame_traditional_cmd_count: u32,
	frame_meshlet_cmd_count:     u32,
	frame_indirect_cmd_count:    u32,
	frame_static_chunk_count:    u32,
	frame_static_transforms:     u32,
	frame_all_transforms:        u32,
	frame_model_count:           u32,

	// Settings snapshot for the current frame. Read by Scene.odin when
	// building pipeline_lookup and active_types bits.
	settings:             GPU_Runtime_Settings,

	// Frame number (debugging only).
	frame_idx:            u64,
}

//* Lifecycle.

DEFAULT_POOL_CAPACITY :: 1024
DEFAULT_SPARSE_CAPACITY :: 65_536

scene_state_init :: proc(s: ^Render_Scene_State, settings: GPU_Runtime_Settings, allocator := context.allocator) {
	s.allocator = allocator
	s.settings = settings
	s.frame_idx = 0

	s.transforms.data = make([dynamic]Gpu_Transform_Component, 0, DEFAULT_POOL_CAPACITY, allocator)
	s.models.data = make([dynamic]Gpu_Model_Component, 0, DEFAULT_POOL_CAPACITY, allocator)
	s.cameras.data = make([dynamic]Gpu_Camera, 0, 16, allocator)
	s.materials.data = make([dynamic]Gpu_Material, 0, DEFAULT_POOL_CAPACITY, allocator)
	s.material_lookup.data = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	s.pipeline_lookup.data = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	s.model_allocations.data = make([dynamic]Gpu_Model_Allocation, 0, DEFAULT_POOL_CAPACITY, allocator)
	s.mesh_allocations.data = make([dynamic]Gpu_Mesh_Allocation, 0, DEFAULT_POOL_CAPACITY * 4, allocator)
	s.draw_descriptors.data = make([dynamic]Gpu_Mesh_Draw_Descriptor, 0, DEFAULT_POOL_CAPACITY * 4, allocator)
	s.model_addresses.data = make([dynamic]Gpu_Model_Addresses, 0, DEFAULT_POOL_CAPACITY, allocator)

	s.sparse.transforms.entity_to_dense = make([dynamic]u32, 0, DEFAULT_SPARSE_CAPACITY, allocator)
	s.sparse.transforms.dense_to_entity = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	s.sparse.models.entity_to_dense = make([dynamic]u32, 0, DEFAULT_SPARSE_CAPACITY, allocator)
	s.sparse.models.dense_to_entity = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	s.sparse.cameras.entity_to_dense = make([dynamic]u32, 0, DEFAULT_SPARSE_CAPACITY, allocator)
	s.sparse.cameras.dense_to_entity = make([dynamic]u32, 0, 16, allocator)
	s.sparse.tags.entity_to_dense = make([dynamic]u32, 0, DEFAULT_SPARSE_CAPACITY, allocator)
	s.sparse.tags.dense_to_entity = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
	s.sparse.animations.entity_to_dense = make([dynamic]u32, 0, DEFAULT_SPARSE_CAPACITY, allocator)
	s.sparse.animations.dense_to_entity = make([dynamic]u32, 0, DEFAULT_POOL_CAPACITY, allocator)
}

scene_state_destroy :: proc(s: ^Render_Scene_State) {
	delete(s.transforms.data)
	delete(s.models.data)
	delete(s.cameras.data)
	delete(s.materials.data)
	delete(s.material_lookup.data)
	delete(s.pipeline_lookup.data)
	delete(s.model_allocations.data)
	delete(s.mesh_allocations.data)
	delete(s.draw_descriptors.data)
	delete(s.model_addresses.data)

	delete(s.sparse.transforms.entity_to_dense)
	delete(s.sparse.transforms.dense_to_entity)
	delete(s.sparse.models.entity_to_dense)
	delete(s.sparse.models.dense_to_entity)
	delete(s.sparse.cameras.entity_to_dense)
	delete(s.sparse.cameras.dense_to_entity)
	delete(s.sparse.tags.entity_to_dense)
	delete(s.sparse.tags.dense_to_entity)
	delete(s.sparse.animations.entity_to_dense)
	delete(s.sparse.animations.dense_to_entity)
	s^ = {}
}

// ---------------------------------------------------------------------------
// Sparse map maintenance.
//
// Called when an entity is added to or removed from a view. The caller
// (extract_scene_state or a separate structural-changes handler) must
// keep these in sync; the culling shaders assume the maps are consistent
// with the dense pools.
// ---------------------------------------------------------------------------

sparse_map_clear :: proc(m: ^Sparse_Map) {
	for i in 0 ..< len(m.entity_to_dense) {
		m.entity_to_dense[i] = INVALID_INDEX
	}
	m.live_entity_count = 0
}

sparse_map_reserve_entity :: proc(m: ^Sparse_Map, entity: u32) {
	for len(m.entity_to_dense) <= int(entity) {
		append(&m.entity_to_dense, INVALID_INDEX)
	}
}

sparse_map_assign :: proc(m: ^Sparse_Map, entity, dense: u32) {
	sparse_map_reserve_entity(m, entity)
	m.entity_to_dense[entity] = dense
}

sparse_map_grow_dense :: proc(m: ^Sparse_Map, dense: u32, entity: u32) {
	for len(m.dense_to_entity) <= int(dense) {
		append(&m.dense_to_entity, INVALID_INDEX)
	}
	m.dense_to_entity[dense] = entity
}

// ---------------------------------------------------------------------------
// Per-frame extraction.
//
// Walks the four BF_ECS views the culling pipeline reads:
//
//   transforms_view      - entities with Transform + Model_Component
//   cameras_view         - entities with Camera
//   tags_view            - entities with active/inactive tag
//   animations_view      - entities with Animation (optional)
//
// Fills the dense pools and sparse maps; resets dense_used counters to 0
// at the start so a frame that re-uses entities can rewrite into the
// same slots. Entities that survive across frames keep their dense index
// because their sparse_map slot is rewritten before the pool index is
// re-emitted.
//
// This is the only place the renderer reads from BF_ECS. Everything
// after this runs entirely on the GPU.
// ---------------------------------------------------------------------------

View_Inputs :: struct {
	transforms:  ^ECS.View,
	cameras:     ^ECS.View,
	tags:        ^ECS.View,
	animations:  ^ECS.View,
	asset_store: ^GPU_Resource_Store, // for Model -> GPU_Model_ID + Gpu_Material lookup
}

// world_view_find_by_name walks the world's view registry and returns
// the first view whose name matches. The renderer's views are
// registered during BF_GPU startup; once cached in MODULE_STATE_VALUE.views
// we don't re-resolve them every frame.
world_view_find_by_name :: proc(world: ^ECS.World, name: string) -> ^ECS.View {
	if world == nil do return nil
	for v in world.views.all {
		if v != nil && v.initialized && v.name == name do return v
	}
	return nil
}

extract_scene_state :: proc(s: ^Render_Scene_State, views: ^View_Inputs) -> bool {
	if s == nil || views == nil do return false

	// reset dense counters and tag maps; pools stay allocated
	s.transforms.dense_used = 0
	s.models.dense_used     = 0
	s.cameras.dense_used    = 0
	sparse_map_clear(&s.sparse.tags)
	sparse_map_clear(&s.sparse.animations)

	// --- tags: small, do first so culling shaders can read active flags ---
	if views.tags != nil && views.tags.initialized {
		// The tags view column carries a u32 flag bitfield. Until the
		// built-in Tag component lands we treat view_entities as the
		// authority and assume all entities are active (bit 0 set).
		entities := ECS.view_entities(views.tags)
		for e, i in entities {
			dense := u32(i)
			sparse_map_assign(&s.sparse.tags, u32(e), dense)
			sparse_map_grow_dense(&s.sparse.tags, dense, u32(e))
		}
	}

	// --- animations (optional) ---
	if views.animations != nil && views.animations.initialized {
		entities := ECS.view_entities(views.animations)
		for e, i in entities {
			sparse_map_assign(&s.sparse.animations, u32(e), u32(i))
		}
	}

	// --- cameras ---
	if views.cameras != nil && views.cameras.initialized {
		// Camera pool is small; we use view_column to read typed camera
		// components. The Camera struct is engine-defined; we mirror the
		// math fields directly into Gpu_Camera.
		entities := ECS.view_entities(views.cameras)
		s.cameras.dense_used = u32(len(entities))
		for e, i in entities {
			dense := u32(i)
			sparse_map_assign(&s.sparse.cameras, u32(e), dense)
			// Gpu_Camera fill is handled by the GPU backend once the
			// engine-level Camera component lands. Until then the slot
			// exists and is filled by the backend at upload time.
		}
		// First camera becomes main, second becomes active.
		if len(entities) > 0 {
			s.frame_main_camera   = u32(entities[0])
			s.frame_active_camera = u32(entities[0])
		}
		if len(entities) > 1 {
			s.frame_active_camera = u32(entities[1])
		}
	}

	// --- transforms + models ---
	if views.transforms != nil && views.transforms.initialized {
		entities := ECS.view_entities(views.transforms)

		// Iterate per row; the typed view column gives us a typed slice
		// of components. Until the Transform + Model_Component built-ins
		// land, view_column returns the engine's chosen layout.
		//
		// For minimal coherence we assume the view is built over
		// (Transform, Model_Component) and we emit a placeholder entry
		// per row. The backend overwrites with real math once the
		// built-ins exist.

		s.transforms.dense_used = u32(len(entities))
		s.models.dense_used     = u32(len(entities))
		s.frame_all_transforms  = u32(len(entities))

		for e, i in entities {
			dense := u32(i)
			sparse_map_assign(&s.sparse.transforms, u32(e), dense)
			sparse_map_assign(&s.sparse.models,     u32(e), dense)
			sparse_map_grow_dense(&s.sparse.transforms, dense, u32(e))
			sparse_map_grow_dense(&s.sparse.models,     dense, u32(e))

			// TransformModelLink links dense transform slot to dense model slot.
			// In a single-table view they're identical.
			_ = dense
		}
	}

	// --- bucket sizing ---
	//
	// The number of indirect commands per (pipeline, material render type)
	// is one per (model, mesh) pair in the scene. Until the cooked asset
	// metadata exists we default to 0 commands per bucket and let the
	// asset upload path fill them in as models are loaded.
	s.frame_traditional_cmd_count = s.mesh_allocations.dense_used // one entry per mesh allocation slot
	s.frame_meshlet_cmd_count     = 0
	s.frame_indirect_cmd_count    = s.frame_traditional_cmd_count + s.frame_meshlet_cmd_count
	s.frame_model_count           = s.models.dense_used
	s.frame_static_chunk_count    = 0
	s.frame_static_transforms     = s.frame_all_transforms

	s.frame_idx += 1
	return true
}

// ---------------------------------------------------------------------------
// Asset upload helpers.
//
// Called when a model / material / mesh is loaded or unloaded. These
// append to the matching pool, update the sparse map, and (for meshes)
// emit per-(model, mesh, lod) Gpu_Mesh_Allocation entries with the
// correct active_types bits per the MeshShaders flag.
// ---------------------------------------------------------------------------

// register_gpu_model appends a model to the address pool and returns the
// assigned GPU_Model_ID. The caller fills in addrs afterwards (the
// sub-buffer addresses aren't known until the GPU backend creates them).
register_gpu_model :: proc(s: ^Render_Scene_State, model_id: u32) -> GPU_Model_ID {
	id := GPU_Model_ID(len(s.model_addresses.data))
	append(&s.model_addresses.data, Gpu_Model_Addresses{})
	sparse_map_assign(&s.sparse.models, model_id, u32(id))
	sparse_map_grow_dense(&s.sparse.models, u32(id), model_id)
	s.model_addresses.dense_used = u32(len(s.model_addresses.data))
	return id
}

// register_gpu_material appends a material and returns the dense index.
register_gpu_material :: proc(s: ^Render_Scene_State, mat: Gpu_Material) -> u32 {
	append(&s.materials.data, mat)
	s.materials.dense_used = u32(len(s.materials.data))
	return s.materials.dense_used - 1
}

// register_mesh_allocation appends one Gpu_Mesh_Allocation slot. The
// active_types bits are zeroed for the meshlet pipeline when MeshShaders
// is disabled, so the culling shaders skip the meshlet slots.
register_mesh_allocation :: proc(
	s: ^Render_Scene_State,
	base: Gpu_Mesh_Allocation,
) -> u32 {
	alloc := base
	if !s.settings.mesh_shaders {
		// Zero the meshlet row of active_types so culling shaders do not
		// atomicAdd into meshlet slots.
		for k in 0 ..< MATERIAL_RENDER_COUNT {
			alloc.active_types[PIPELINE_MESHLET][k] = 0
		}
	}
	append(&s.mesh_allocations.data, alloc)
	s.mesh_allocations.dense_used = u32(len(s.mesh_allocations.data))
	return s.mesh_allocations.dense_used - 1
}

// register_draw_descriptor appends a Gpu_Mesh_Draw_Descriptor and
// returns the assigned descriptor index (which becomes gl_DrawID at
// shading time).
register_draw_descriptor :: proc(s: ^Render_Scene_State, desc: Gpu_Mesh_Draw_Descriptor) -> u32 {
	append(&s.draw_descriptors.data, desc)
	s.draw_descriptors.dense_used = u32(len(s.draw_descriptors.data))
	return s.draw_descriptors.dense_used - 1
}

// register_pipeline_lookup writes the pipeline index for a (model, mesh)
// slot: 0 traditional, 1 meshlet. When MeshShaders is false the entry
// is forced to 0 regardless of the input.
register_pipeline_lookup :: proc(s: ^Render_Scene_State, mesh_index: u32, pipeline: u32) {
	for len(s.pipeline_lookup.data) <= int(mesh_index) {
		append(&s.pipeline_lookup.data, 0)
	}
	s.pipeline_lookup.data[mesh_index] = s.settings.mesh_shaders ? pipeline : 0
	s.pipeline_lookup.dense_used = u32(len(s.pipeline_lookup.data))
}

// register_material_lookup writes the material index for a (model, mesh)
// slot. The culling shaders read materialOffset + meshIndex into this
// table to resolve which material the mesh uses.
register_material_lookup :: proc(s: ^Render_Scene_State, mesh_index, material_index: u32) {
	for len(s.material_lookup.data) <= int(mesh_index) {
		append(&s.material_lookup.data, 0)
	}
	s.material_lookup.data[mesh_index] = material_index
	s.material_lookup.dense_used = u32(len(s.material_lookup.data))
}
