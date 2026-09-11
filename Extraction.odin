// BF_GPU/Extraction.odin
//
// The ECS -> Render_Scene extraction contract.
//
// This is the ONLY place BF_GPU reads BF_ECS. Everything downstream of it
// (GPU_Scene, culling, indirect rendering) is derived from Render_Scene and
// never touches the ECS again.
//
// ---------------------------------------------------------------------------
// The contract
// ---------------------------------------------------------------------------
//
// Required components (an entity without both is not renderable):
//
//   Transform                 world / previous_world matrices, dirty flag
//   Render_Model              Asset_Ref of the model + Render_Instance_Flags
//
// Gates (an entity that fails any of these is skipped for the frame):
//
//   Render_Model.flags        must contain .Visible and must not contain .Hidden
//   Spatial_State.flags       when the component exists it must contain .Visible
//
// Optional components:
//
//   Render_Material_Override  per-slot material Asset_Ref
//   Chunk_Membership          authoritative chunk for the entity
//   Spatial_Bounds            world-space AABB; derived from the transform
//                             translation when absent
//
// Asset identity stays Asset_ID / Asset_Ref on the ECS side. The renderer
// resolves it into GPU_Model_ID / GPU_Material_ID through GPU_Resource_Store
// (Resources.odin) and stores the result on the Render_Instance. An asset
// that has no GPU resource yet is extracted and flagged .Pending_Asset; it is
// re-resolved on later frames and starts drawing once the upload lands.
//
// ---------------------------------------------------------------------------
// Scope: chunks, not the whole world
// ---------------------------------------------------------------------------
//
// Extraction never scans the ECS world. It walks the BF_ECS runtime chunk
// index and visits only the entities of the chunks flagged .Visible (or
// .Active) - the same world chunks server-side interest management consumes.
// `Explicit_Entities` exists for editor / test paths that already know the
// entity set.
//
// ---------------------------------------------------------------------------
// Lifetime handling
// ---------------------------------------------------------------------------
//
// Each contributing entity owns a stable renderer slot for as long as it keeps
// contributing, so per-slot GPU state (previous-frame transforms, sparse maps)
// stays valid across frames. Every frame produces three change sets:
//
//   added    slots that started contributing this frame
//   updated  slots whose extracted state changed this frame
//   removed  slots retired this frame (entity destroyed, component removed,
//            gate failed, or the chunk left the visible set)
//
// GPU_Scene consumes exactly those three lists; see Scene.odin.

package BF_GPU

import "core:log"
import mth "../../Core/BF_Math"
import ECS "../BF_ECS"

//* COMPONENT ACCESS
// A resolved Component_Binding: the concrete ODE table plus the type-erased
// accessors the BF_ECS registry publishes for it. Resolving through the
// registry (instead of importing ODE here) keeps the component -> storage
// mapping owned by BF_ECS.
Component_Access :: struct {
	table: rawptr,
	get:   ECS.Component_Get_Proc,
	has:   ECS.Component_Has_Proc,
}

component_access_valid :: #force_inline proc(access: Component_Access) -> bool {
	return access.table != nil && access.get != nil
}

component_access_get :: #force_inline proc(
	$T: typeid,
	access: Component_Access,
	entity: ECS.Entity,
) -> ^T {
	if access.table == nil || access.get == nil do return nil
	raw := access.get(access.table, entity)
	if raw == nil do return nil
	return cast(^T)raw
}

// Databases searched, in order, when resolving a component binding. The first
// database that holds a binding for the type wins, so a project may keep the
// render components on Gameplay and the spatial ones on Spatial without the
// renderer hard-coding either choice.
EXTRACTION_DATABASE_ORDER :: []ECS.Database_Kind{.Gameplay, .Spatial, .Editor, .Custom}

component_access_bind :: proc(
	$T: typeid,
	world: ^ECS.World,
	order: []ECS.Database_Kind = EXTRACTION_DATABASE_ORDER,
) -> Component_Access {
	if world == nil do return {}
	descriptor := ECS.component_find_by_type(T, &world.registry)
	if descriptor == nil do return {}
	for kind in order {
		binding := ECS.component_binding_find(descriptor, kind)
		if binding != nil && binding.table != nil {
			return Component_Access{table = binding.table, get = binding.get, has = binding.has}
		}
	}
	return {}
}

//* BINDINGS
Extraction_Bindings :: struct {
	transform:         Component_Access,
	render_model:      Component_Access,
	material_override: Component_Access,
	chunk_membership:  Component_Access,
	spatial_bounds:    Component_Access,
	spatial_state:     Component_Access,
	camera:            Component_Access,
	camera_active:     Component_Access,
	bound:             bool,
}

// Resolves every component the contract mentions. Fails only when one of the
// two required components is missing from the registry, because without them
// nothing is renderable.
extraction_bindings_resolve :: proc(bindings: ^Extraction_Bindings, world: ^ECS.World) -> bool {
	if bindings == nil || world == nil do return false

	bindings.transform = component_access_bind(ECS.Transform, world)
	bindings.render_model = component_access_bind(ECS.Render_Model, world)
	bindings.material_override = component_access_bind(ECS.Render_Material_Override, world)
	bindings.chunk_membership = component_access_bind(ECS.Chunk_Membership, world)
	bindings.spatial_bounds = component_access_bind(ECS.Spatial_Bounds, world)
	bindings.spatial_state = component_access_bind(ECS.Spatial_State, world)
	bindings.camera = component_access_bind(ECS.Camera, world)
	bindings.camera_active = component_access_bind(ECS.Camera_Active, world)

	bindings.bound =
		component_access_valid(bindings.transform) && component_access_valid(bindings.render_model)
	return bindings.bound
}

//* SOURCE
Extraction_Scope :: enum u8 {
	// Chunks flagged .Visible in the BF_ECS chunk index (default).
	Visible_Chunks,
	// Chunks flagged .Active; a superset used by editor / debug views.
	Active_Chunks,
	// Caller-provided entity list; no chunk walk.
	Explicit_Entities,
}

Extraction_Source :: struct {
	world:         ^ECS.World,
	store:         ^GPU_Resource_Store,
	bindings:      Extraction_Bindings,
	scope:         Extraction_Scope,
	// Used by Extraction_Scope.Explicit_Entities.
	entities:      []ECS.Entity,
	// Cameras are never chunk-derived: the renderer owns the camera set and
	// the active-camera choice. See Renderer.odin.
	cameras:       []ECS.Entity,
	main_camera:   ECS.Entity,
	active_camera: ECS.Entity,
	viewport_w:    f32,
	viewport_h:    f32,
}

// A zero-valued Entity is a *valid* entity (index 0, generation 0), so the
// camera designations must be explicitly cleared before use. Always build a
// source through this instead of a bare struct literal.
extraction_source_init :: proc(src: ^Extraction_Source, world: ^ECS.World, store: ^GPU_Resource_Store) {
	if src == nil do return
	src.world = world
	src.store = store
	src.scope = .Visible_Chunks
	src.main_camera = ECS.ENTITY_INVALID
	src.active_camera = ECS.ENTITY_INVALID
}

//* STATS
Extraction_Stats :: struct {
	chunks_visited:     u32,
	entities_visited:   u32,
	instances_added:    u32,
	instances_updated:  u32,
	instances_retained: u32,
	instances_removed:  u32,
	cameras:            u32,
	skipped_not_renderable: u32,
	skipped_hidden:     u32,
	pending_assets:     u32,
}

//* LIFECYCLE
RENDER_SCENE_DEFAULT_CAPACITY :: 1024

render_scene_init :: proc(
	scene: ^Render_Scene,
	allocator := context.allocator,
	capacity: int = RENDER_SCENE_DEFAULT_CAPACITY,
) {
	if scene == nil do return
	scene.allocator = allocator
	scene.instances = make([dynamic]Render_Instance, 0, capacity, allocator)
	scene.transforms = make([dynamic]Render_Transform, 0, capacity, allocator)
	scene.spatial = make([dynamic]Render_Spatial_Metadata, 0, capacity, allocator)
	scene.lights = make([dynamic]Render_Light, 0, 64, allocator)
	scene.cameras = make([dynamic]Render_Camera, 0, 8, allocator)
	scene.particles = make([dynamic]Render_Particle, 0, 16, allocator)
	scene.chunks = make([dynamic]Render_Chunk, 0, 64, allocator)
	scene.chunk_instances = make([dynamic]Render_Instance_ID, 0, capacity, allocator)
	scene.entity_to_instance = make(map[ECS.Entity]Render_Instance_ID, allocator)
	scene.free_slots = make([dynamic]u32, 0, 64, allocator)
	scene.added = make([dynamic]Render_Instance_ID, 0, 64, allocator)
	scene.updated = make([dynamic]Render_Instance_ID, 0, capacity, allocator)
	scene.removed = make([dynamic]Render_Removal, 0, 64, allocator)
	scene.main_camera = INVALID_INDEX
	scene.active_camera = INVALID_INDEX
	scene.main_camera_entity = ECS.ENTITY_INVALID
	scene.active_camera_entity = ECS.ENTITY_INVALID
}

render_scene_destroy :: proc(scene: ^Render_Scene) {
	if scene == nil do return
	delete(scene.instances)
	delete(scene.transforms)
	delete(scene.spatial)
	delete(scene.lights)
	delete(scene.cameras)
	delete(scene.particles)
	delete(scene.chunks)
	delete(scene.chunk_instances)
	delete(scene.entity_to_instance)
	delete(scene.free_slots)
	delete(scene.added)
	delete(scene.updated)
	delete(scene.removed)
	scene^ = {}
}

render_scene_is_valid :: #force_inline proc(scene: ^Render_Scene) -> bool {
	return scene != nil && scene.allocator.procedure != nil
}

// Retires every slot without producing removal records. Used when the world
// goes away entirely; the GPU representation is expected to be reset too.
render_scene_reset :: proc(scene: ^Render_Scene) {
	if !render_scene_is_valid(scene) do return
	clear(&scene.instances)
	clear(&scene.transforms)
	clear(&scene.spatial)
	clear(&scene.lights)
	clear(&scene.cameras)
	clear(&scene.particles)
	clear(&scene.chunks)
	clear(&scene.chunk_instances)
	clear(&scene.entity_to_instance)
	clear(&scene.free_slots)
	clear(&scene.added)
	clear(&scene.updated)
	clear(&scene.removed)
	scene.live_count = 0
	scene.static_count = 0
	scene.pending_asset_count = 0
	scene.main_camera = INVALID_INDEX
	scene.active_camera = INVALID_INDEX
	scene.main_camera_entity = ECS.ENTITY_INVALID
	scene.active_camera_entity = ECS.ENTITY_INVALID
}

render_scene_instance :: #force_inline proc(
	scene: ^Render_Scene,
	id: Render_Instance_ID,
) -> ^Render_Instance {
	if !render_scene_is_valid(scene) || id == RENDER_INSTANCE_INVALID do return nil
	index := render_instance_index(id)
	if int(index) >= len(scene.instances) do return nil
	return &scene.instances[index]
}

render_scene_find :: #force_inline proc(
	scene: ^Render_Scene,
	entity: ECS.Entity,
) -> Render_Instance_ID {
	if !render_scene_is_valid(scene) do return RENDER_INSTANCE_INVALID
	if id, ok := scene.entity_to_instance[entity]; ok do return id
	return RENDER_INSTANCE_INVALID
}

//* EXTRACTION
// Rebuilds this frame's view of the ECS into `scene`. Returns the per-frame
// statistics and whether the extraction ran at all.
render_scene_extract :: proc(
	scene: ^Render_Scene,
	src: ^Extraction_Source,
) -> (
	stats: Extraction_Stats,
	ok: bool,
) {
	if !render_scene_is_valid(scene) do return stats, false
	if src == nil || src.world == nil do return stats, false
	if !src.bindings.bound {
		if !extraction_bindings_resolve(&src.bindings, src.world) {
			return stats, false
		}
	}

	scene.frame_index += 1
	clear(&scene.added)
	clear(&scene.updated)
	clear(&scene.removed)
	clear(&scene.chunks)
	clear(&scene.chunk_instances)
	clear(&scene.lights)
	scene.pending_asset_count = 0

	switch src.scope {
	case .Visible_Chunks:
		extract_chunk_state(scene, src, .Visible, &stats)
	case .Active_Chunks:
		extract_chunk_state(scene, src, .Active, &stats)
	case .Explicit_Entities:
		for entity in src.entities {
			extract_entity(scene, src, entity, ECS.CHUNK_INVALID, &stats)
		}
	}

	stats.instances_removed = render_scene_sweep(scene)
	extract_cameras(scene, src, &stats)

	// Static-instance accounting is a whole-slot property; recompute it once
	// per frame rather than trying to keep a delta in sync with the sweep.
	scene.static_count = 0
	for &inst in scene.instances {
		if .Live in inst.flags && .Static in inst.flags {
			scene.static_count += 1
		}
	}
	return stats, true
}

//* CHUNK WALK
@(private = "file")
Chunk_Walk :: struct {
	scene: ^Render_Scene,
	src:   ^Extraction_Source,
	stats: ^Extraction_Stats,
	index: ^ECS.Chunk_Index,
}

@(private = "file")
extract_chunk_state :: proc(
	scene: ^Render_Scene,
	src: ^Extraction_Source,
	state: ECS.Chunk_State,
	stats: ^Extraction_Stats,
) {
	index := ECS.world_chunk_index(src.world)
	if index == nil do return
	walk := Chunk_Walk {
		scene = scene,
		src   = src,
		stats = stats,
		index = index,
	}
	ECS.chunk_index_for_each_state(index, state, chunk_walk_visit, cast(rawptr)&walk)
}

@(private = "file")
chunk_walk_visit :: proc(id: ECS.Chunk_ID, chunk: ^ECS.Chunk_Runtime, user_data: rawptr) -> bool {
	walk := cast(^Chunk_Walk)user_data
	if walk == nil do return false

	walk.stats.chunks_visited += 1
	first := u32(len(walk.scene.chunk_instances))

	entities := ECS.chunk_index_chunk_entities(walk.index, id)
	for entity in entities {
		instance := extract_entity(walk.scene, walk.src, entity, id, walk.stats)
		if instance != RENDER_INSTANCE_INVALID {
			append(&walk.scene.chunk_instances, instance)
		}
	}

	append(
		&walk.scene.chunks,
		Render_Chunk {
			id = id,
			bounds = chunk.bounds,
			first_instance = first,
			instance_count = u32(len(walk.scene.chunk_instances)) - first,
		},
	)
	return true
}

//* ENTITY EXTRACTION
@(private = "file")
extract_entity :: proc(
	scene: ^Render_Scene,
	src: ^Extraction_Source,
	entity: ECS.Entity,
	walk_chunk: ECS.Chunk_ID,
	stats: ^Extraction_Stats,
) -> Render_Instance_ID {
	if entity == ECS.ENTITY_INVALID do return RENDER_INSTANCE_INVALID
	stats.entities_visited += 1
	bindings := &src.bindings

	transform := component_access_get(ECS.Transform, bindings.transform, entity)
	if transform == nil {
		stats.skipped_not_renderable += 1
		return RENDER_INSTANCE_INVALID
	}
	model := component_access_get(ECS.Render_Model, bindings.render_model, entity)
	if model == nil {
		stats.skipped_not_renderable += 1
		return RENDER_INSTANCE_INVALID
	}
	if .Hidden in model.flags || .Visible not_in model.flags {
		stats.skipped_hidden += 1
		return RENDER_INSTANCE_INVALID
	}
	// Spatial_State is the engine-wide per-entity state; when the entity
	// carries it, it is authoritative over renderer visibility. Entities
	// without it default to visible.
	if state := component_access_get(ECS.Spatial_State, bindings.spatial_state, entity);
	   state != nil {
		if .Visible not_in state.flags {
			stats.skipped_hidden += 1
			return RENDER_INSTANCE_INVALID
		}
	}

	id, is_new := render_scene_acquire(scene, entity)
	slot := render_instance_index(id)
	instance := &scene.instances[slot]

	previous_flags := instance.flags
	previous_model := instance.model
	previous_chunk := instance.chunk

	flags: Render_Instance_Flags = {.Live}
	if .Cast_Shadow in model.flags do flags |= {.Cast_Shadow}
	if .Receive_Shadow in model.flags do flags |= {.Receive_Shadow}
	if .Static in model.flags do flags |= {.Static}
	if .Dynamic in model.flags do flags |= {.Dynamic}

	//* ASSET RESOLUTION
	// Re-resolve when the slot is new, when the Asset_Ref changed, or while
	// the asset is still unresolved (the upload may have landed since).
	asset_changed := !is_new && previous_model.id != model.model.id
	if is_new || asset_changed || instance.gpu_model == GPU_MODEL_INVALID {
		instance.gpu_model = gpu_model_find(src.store, model.model.id)
	}
	if asset_changed do flags |= {.Asset_Dirty}
	instance.model = model.model
	if instance.gpu_model == GPU_MODEL_INVALID {
		flags |= {.Pending_Asset}
		stats.pending_assets += 1
		scene.pending_asset_count += 1
	}

	//* MATERIAL OVERRIDE
	override := component_access_get(
		ECS.Render_Material_Override,
		bindings.material_override,
		entity,
	)
	if override != nil {
		if is_new ||
		   instance.material_override.id != override.material.id ||
		   instance.gpu_material_override == GPU_MATERIAL_INVALID {
			if instance.material_override.id != override.material.id && !is_new {
				flags |= {.Asset_Dirty}
			}
			instance.gpu_material_override = gpu_material_find(src.store, override.material.id)
		}
		instance.material_override = override.material
		instance.material_override_slot = override.slot
		instance.material_override_index = instance.gpu_material_override == GPU_MATERIAL_INVALID \
			? INVALID_INDEX \
			: u32(instance.gpu_material_override)
	} else {
		if !is_new && instance.material_override.id != ECS.Asset_ID(0) {
			flags |= {.Asset_Dirty}
		}
		instance.material_override = {}
		instance.gpu_material_override = GPU_MATERIAL_INVALID
		instance.material_override_index = INVALID_INDEX
	}

	//* CHUNK
	// The Chunk_Membership component is authoritative; the walked chunk is
	// only a fallback for entities the caller supplied explicitly.
	chunk := walk_chunk
	if membership := component_access_get(ECS.Chunk_Membership, bindings.chunk_membership, entity);
	   membership != nil {
		chunk = membership.chunk
	}
	instance.chunk = chunk

	//* TRANSFORM
	extracted := Render_Transform {
		world          = transform.world,
		previous_world = transform.previous_world,
		normal         = mat3_normal_from_mat4(transform.world),
	}
	stored := &scene.transforms[slot]
	if is_new || stored.world != extracted.world || transform.dirty {
		flags |= {.Transform_Dirty}
	}
	stored^ = extracted

	//* SPATIAL METADATA
	bounds: mth.AABB
	if spatial_bounds := component_access_get(
		ECS.Spatial_Bounds,
		bindings.spatial_bounds,
		entity,
	); spatial_bounds != nil {
		bounds = spatial_bounds.world
	} else {
		// No authored bounds: fall back to a degenerate box at the world
		// origin of the transform. GPU culling treats it as a point.
		origin := mth.Vec3{transform.world[12], transform.world[13], transform.world[14]}
		bounds = mth.AABB {
			min = origin,
			max = origin,
		}
	}
	spatial := &scene.spatial[slot]
	spatial.bounds = bounds
	spatial.chunk = chunk
	spatial.distance = 0
	spatial.lod = 0
	spatial.flags = render_spatial_flags(flags)

	instance.entity = entity
	instance.transform_index = slot
	instance.spatial_index = slot
	instance.flags = flags
	instance.last_seen_frame = scene.frame_index

	if is_new {
		append(&scene.added, id)
		stats.instances_added += 1
	} else if (flags & RENDER_INSTANCE_PERSISTENT_FLAGS) !=
		   (previous_flags & RENDER_INSTANCE_PERSISTENT_FLAGS) ||
	   chunk != previous_chunk ||
	   .Transform_Dirty in flags ||
	   .Asset_Dirty in flags {
		append(&scene.updated, id)
		stats.instances_updated += 1
	} else {
		stats.instances_retained += 1
	}
	return id
}

// The flags that describe lasting instance state. Transform_Dirty and
// Asset_Dirty are per-frame signals, so they are excluded from the
// "did anything change?" comparison - otherwise every instance would report a
// change on the frame after it was added.
@(private = "file")
RENDER_INSTANCE_PERSISTENT_FLAGS :: Render_Instance_Flags {
	.Live,
	.Cast_Shadow,
	.Receive_Shadow,
	.Static,
	.Dynamic,
	.Pending_Asset,
}

@(private = "file")
render_spatial_flags :: proc(flags: Render_Instance_Flags) -> Render_Spatial_Flags {
	out: Render_Spatial_Flags = {.Visible}
	if .Cast_Shadow in flags do out |= {.Cast_Shadow}
	if .Receive_Shadow in flags do out |= {.Receive_Shadow}
	if .Static in flags do out |= {.Static}
	if .Dynamic in flags do out |= {.Dynamic}
	return out
}

//* SLOT ALLOCATION
@(private = "file")
render_scene_acquire :: proc(
	scene: ^Render_Scene,
	entity: ECS.Entity,
) -> (
	id: Render_Instance_ID,
	is_new: bool,
) {
	if existing, ok := scene.entity_to_instance[entity]; ok {
		return existing, false
	}

	slot: u32
	if count := len(scene.free_slots); count > 0 {
		slot = scene.free_slots[count - 1]
		pop(&scene.free_slots)
	} else {
		slot = u32(len(scene.instances))
		append(&scene.instances, Render_Instance{})
		append(&scene.transforms, Render_Transform{})
		append(&scene.spatial, Render_Spatial_Metadata{})
	}

	id = render_instance_id(slot)
	scene.entity_to_instance[entity] = id
	scene.live_count += 1
	return id, true
}

//* SLOT RETIREMENT
// Any live slot that this frame's walk did not touch stops contributing:
// the entity was destroyed, lost a required component, failed a gate, or its
// chunk left the visible set. The slot is recorded in `removed` so the GPU
// representation can clear it, then returned to the free list.
@(private = "file")
render_scene_sweep :: proc(scene: ^Render_Scene) -> u32 {
	removed: u32 = 0
	for &instance, index in scene.instances {
		if .Live not_in instance.flags do continue
		if instance.last_seen_frame == scene.frame_index do continue

		id := render_instance_id(u32(index))
		append(&scene.removed, Render_Removal{instance = id, entity = instance.entity})
		delete_key(&scene.entity_to_instance, instance.entity)
		append(&scene.free_slots, u32(index))

		instance = {}
		scene.transforms[index] = {}
		scene.spatial[index] = {}
		if scene.live_count > 0 do scene.live_count -= 1
		removed += 1
	}
	return removed
}

//* CAMERAS
// Explicit active-camera semantics. There is no "first camera wins" rule:
//
//   1. The renderer's designated active camera entity, when it is registered
//      and still carries a Camera component.
//   2. Otherwise the single entity carrying Camera_Active{value = true}. More
//      than one is a project error; the extraction reports it and leaves the
//      active camera unresolved.
//   3. Otherwise unresolved (INVALID_INDEX) - the frame renders no view.
//
// The main camera follows the same rule with its own designation and falls
// back to the active camera.
@(private = "file")
extract_cameras :: proc(scene: ^Render_Scene, src: ^Extraction_Source, stats: ^Extraction_Stats) {
	clear(&scene.cameras)
	scene.main_camera = INVALID_INDEX
	scene.active_camera = INVALID_INDEX
	scene.main_camera_entity = ECS.ENTITY_INVALID
	scene.active_camera_entity = ECS.ENTITY_INVALID

	bindings := &src.bindings
	if !component_access_valid(bindings.camera) do return

	flagged_index := INVALID_INDEX
	flagged_entity := ECS.ENTITY_INVALID
	flagged_count := 0

	for entity in src.cameras {
		if entity == ECS.ENTITY_INVALID do continue
		if !ECS.world_entity_is_alive(src.world, entity) do continue
		camera := component_access_get(ECS.Camera, bindings.camera, entity)
		if camera == nil do continue

		dense := u32(len(scene.cameras))
		append(&scene.cameras, build_render_camera(src, entity, camera))
		stats.cameras += 1

		if active := component_access_get(ECS.Camera_Active, bindings.camera_active, entity);
		   active != nil && active.value {
			flagged_count += 1
			if flagged_index == INVALID_INDEX {
				flagged_index = dense
				flagged_entity = entity
			}
		}
		if entity == src.active_camera && src.active_camera != ECS.ENTITY_INVALID {
			scene.active_camera = dense
			scene.active_camera_entity = entity
		}
		if entity == src.main_camera && src.main_camera != ECS.ENTITY_INVALID {
			scene.main_camera = dense
			scene.main_camera_entity = entity
		}
	}

	if scene.active_camera == INVALID_INDEX {
		if flagged_count > 1 {
			log.warnf(
				"[BF_GPU] %d entities carry Camera_Active; active camera left unresolved",
				flagged_count,
			)
		} else if flagged_count == 1 {
			scene.active_camera = flagged_index
			scene.active_camera_entity = flagged_entity
		}
	}
	if scene.main_camera == INVALID_INDEX {
		scene.main_camera = scene.active_camera
		scene.main_camera_entity = scene.active_camera_entity
	}
}

@(private = "file")
build_render_camera :: proc(
	src: ^Extraction_Source,
	entity: ECS.Entity,
	camera: ^ECS.Camera,
) -> Render_Camera {
	view := MAT4_IDENTITY
	if transform := component_access_get(ECS.Transform, src.bindings.transform, entity);
	   transform != nil {
		if inverse, ok := mat4_affine_inverse(transform.world); ok {
			view = inverse
		}
	}

	aspect := camera.aspect
	if aspect <= 0 {
		aspect = src.viewport_h > 0 ? src.viewport_w / src.viewport_h : 1.0
	}
	near := camera.near_plane > 0 ? camera.near_plane : 0.1
	far := camera.far_plane > near ? camera.far_plane : near + 1000.0

	projection: mth.Mat4
	switch camera.projection {
	case .Perspective:
		projection = mat4_perspective(camera.fov_y, aspect, near, far)
	case .Orthographic:
		projection = mat4_orthographic(camera.ortho_height, aspect, near, far)
	}

	return Render_Camera {
		entity = entity,
		// Cameras do not own a Render_Scene slot; their matrices are carried
		// here directly.
		transform_index = INVALID_INDEX,
		projection = camera.projection,
		view = view,
		projection_matrix = projection,
		view_projection_matrix = mat4_mul(projection, view),
		near_plane = near,
		far_plane = far,
		width = u32(max(src.viewport_w, 1)),
		height = u32(max(src.viewport_h, 1)),
	}
}
