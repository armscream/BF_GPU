// BF_GPU/Renderer.odin
//
// Public renderer module surface. Owns the host-side state
// (Frame_Context_State + Render_Scene_State) and exposes the per-frame
// hooks the engine calls:
//
//   renderer_init(settings)
//       one-time setup; allocates persistent GPU buffers through the
//       registered backend, refreshes FrameGlobalContext addresses.
//
// renderer_register_systems()
//       registers the four BF_DAG systems that drive a frame:
//           BF_GPU.SceneExtract  stage .Render_Extract
//           BF_GPU.Render_Upload stage .Render_Upload
//           BF_GPU.Render_Submit stage .Render_Submit
//           BF_GPU.FramePresent  stage .PostRender
//
//       Stages form a strict chain: extraction -> upload -> submit, with
//       the optional present step after submit completes. The stage
//       barrier in BF_DAG/dag.odin guarantees this order without
//       requiring explicit System_Dependency entries from the module.
//
//   renderer_shutdown()
//       destroys host state and tells the backend to release resources.
//
// The actual GPU backend calls (Vulkan / MoltenVK) live behind a
// GPU_Backend interface that the Vulkan module wires in during its own
// module load. This file never imports Vulkan directly.

package BF_GPU  

import "base:runtime"
import "core:log"
import "../../Core"
import ECS "../BF_ECS"

Module_State :: struct {
	allocator:   runtime.Allocator,
	frame_ctx:   Frame_Context_State,
	// CPU-side extracted representation of the ECS (Extraction.odin).
	scene:       Render_Scene,
	// GPU-side derived representation of `scene` (Scene.odin).
	gpu:         GPU_Scene,
	// Asset_ID -> GPU resource identity (Resources.odin).
	store:       GPU_Resource_Store,
	// Per-frame extraction inputs; component bindings are resolved once.
	source:      Extraction_Source,
	// Explicit camera registry. The renderer owns which entities are cameras
	// and which one is active; extraction never guesses.
	cameras:     [dynamic]ECS.Entity,
	stats:       Extraction_Stats,
	settings:    GPU_Runtime_Settings,
	initialized: bool,
	world:       ^ECS.World,
	backend:     ^GPU_Backend,
	viewport_w:  f32,
	viewport_h:  f32,
	// Cached BF_DAG scheduler service handle. Resolved once at
	// renderer_init so per-frame code (render_submit_step, the
	// GPU completion pre-frame hook) does not pay a service-lookup
	// every frame. Nil when BF_DAG is unavailable.
	scheduler:   ^Core.Scheduler_Service,
	// Persistent external node that represents GPU completion for
	// the most recent frame's submission. Created in renderer_init,
	// destroyed in renderer_shutdown. The pre-frame hook attached
	// in renderer_init re-arms a wait on it for the FramePresent
	// DAG node every frame, and render_submit_step signals it after
	// the backend reports a successful record_frame.
	gpu_complete: Core.External_Node_Handle,
}

@(private)
MODULE_STATE_VALUE: Module_State

// ===========================================================================
// Backend abstraction. The Vulkan backend implements this interface and
// registers itself via register_gpu_backend() during its own module load.
// All GPU-side calls go through here so this file never imports Vulkan.
// ===========================================================================

GPU_Backend :: struct {
	name:                string,
	compile_shader:      proc(path, stage: cstring) -> rawptr,
	create_pipeline:     proc(shader: rawptr, descriptor: rawptr) -> rawptr,
	create_buffer:       proc(kind: Gpu_Buffer_Kind, size: u64, stride: u32) -> Gpu_Buffer_Handle,
	get_device_address:  proc(handle: Gpu_Buffer_Handle) -> u64,
	upload_buffer:       proc(handle: Gpu_Buffer_Handle, data: rawptr, size: u64) -> bool,
	record_frame:        proc(gpu: ^GPU_Scene, ctx: ^Frame_Context_State) -> bool,
	destroy_buffer:      proc(handle: Gpu_Buffer_Handle),
	destroy_pipeline:    proc(p: rawptr),
	destroy_shader:      proc(s: rawptr),
}

register_gpu_backend :: proc(backend: ^GPU_Backend) {
	if backend == nil do return
	MODULE_STATE_VALUE.backend = backend
	log.infof("[BF_GPU] GPU backend registered: %s", backend.name)
}

unregister_gpu_backend :: proc() {
	MODULE_STATE_VALUE.backend = nil
}

////////////////////////////////////////////////////////////////////////////////////////////
//* Public lifecycle.

renderer_init :: proc(settings: GPU_Runtime_Settings, allocator := context.allocator) -> bool {
	s := &MODULE_STATE_VALUE

	if s.initialized {
		log.warn("[BF_GPU] renderer_init called twice; ignoring")
		return true
	}

	s.allocator = allocator
	s.settings  = settings
	if s.settings.lod_count == 0 {
		s.settings.lod_count = 4
	}
	s.viewport_w = 1920.0
	s.viewport_h = 1080.0

	frame_context_init(&s.frame_ctx, allocator)
	gpu_store_init(&s.store, allocator)
	render_scene_init(&s.scene, allocator)
	gpu_scene_init(&s.gpu, s.settings, allocator)
	s.cameras = make([dynamic]ECS.Entity, 0, 8, allocator)

	// Look up the BF_ECS world service so scene extraction has something
	// to walk. If BF_ECS is not loaded the renderer still initialises
	// but the extraction system becomes a no-op.
	if reg := core_service_registry(); reg != nil {
		handle, found := Core.service_find(reg, ECS.ECS_WORLD_SERVICE_NAME)
		if found {
			instance := Core.service_get(reg, handle)
			if instance != nil {
				s.world = cast(^ECS.World)instance
			}
		}
	}
	if s.world == nil {
		log.warn("[BF_GPU] BF_ECS world unavailable; scene extraction will be a no-op")
	} else if !extraction_bindings_resolve(&s.source.bindings, s.world) {
		log.warn(
			"[BF_GPU] Transform / Render_Model are not registered with BF_ECS; extraction will retry each frame",
		)
	}
	extraction_source_init(&s.source, s.world, &s.store)

	if !vulkan_init() {
		log.error("[BF_GPU] Failed to initialize Vulkan")
		return false
	}

	// Persistent buffer creation is the backend's job. Once the backend
	// calls create_buffer() for every Gpu_Buffer_Kind, refresh_frame_addresses()
	// copies the resulting device addresses into the FrameGlobalContext
	// mirror so the culling shaders can read them.

	// Wire the GPU completion external node + pre-frame hook on the
	// BF_DAG scheduler service. Failure is non-fatal (the renderer
	// still works; only the GPU completion gate is a no-op).
	if reg := core_service_registry(); reg != nil {
		handle, found := Core.service_find(reg, Core.BF_DAG_SCHEDULER_SERVICE_NAME)
		if found {
			instance := Core.service_get(reg, handle)
			if instance != nil {
				s.scheduler = cast(^Core.Scheduler_Service)instance
			}
		}
	}
	if s.scheduler == nil {
		log.warn("[BF_GPU] BF_DAG scheduler service unavailable; GPU completion external node disabled")
	} else if !gpu_completion_setup() {
		log.warn("[BF_GPU] GPU completion external node setup failed; rendering will not gate on GPU completion")
	}

	s.initialized = true
	log.infof(
		"[BF_GPU] renderer initialized (meshShaders=%v hiz=%v frustum=%v cone=%v sort=%v)",
		s.settings.mesh_shaders,
		s.settings.hiz_occlusion,
		s.settings.frustum_culling,
		s.settings.cone_culling,
		s.settings.gpu_sort_extension,
	)
	return true
}

renderer_shutdown :: proc() {
	s := &MODULE_STATE_VALUE
	if !s.initialized do return

	gpu_completion_teardown()
	s.scheduler = nil
	delete(s.cameras)
	gpu_scene_destroy(&s.gpu)
	render_scene_destroy(&s.scene)
	gpu_store_destroy(&s.store)
	frame_context_destroy(&s.frame_ctx)
	s^ = {}
	log.info("[BF_GPU] renderer shutdown")
}

// ===========================================================================
//* GPU completion external node.
//
// The intended dependency chain (prompt_plan §32-34):
//
//   ECS/Spatial -> SceneExtract -> Render_Upload -> Render_Submit
//                                       ^
//                                       |
//                              Render_Upload (or FramePresent)
//                              waits on the GPU completion
//                              external node created here.
//
// The pre-frame hook re-attaches a wait on this node for FramePresent
// every frame. After render_submit_step succeeds, the renderer signals
// the node. In v1 the signal happens synchronously (the backend has
// already completed the submission by the time record_frame returns);
// once a real GPU fence is wired in the backend can defer the signal
// until the fence completes and the same wiring still gates
// FramePresent on real GPU completion.

// gpu_completion_pre_frame_hook is the pre-frame callback the
// scheduler invokes at the top of every begin_frame. It re-attaches
// a wait on the GPU completion external node to the FramePresent
// system, so FramePresent only fires once the previous frame's GPU
// work has signalled completion.
//
// Package-visible (no @(private)) so tests_gpu_completion.odin can
// invoke it directly to simulate a frame's begin_frame without
// spinning up the full service registry.
gpu_completion_pre_frame_hook :: proc(runtime_raw: rawptr, user_data: rawptr) {
	_ = runtime_raw
	state := cast(^Module_State)user_data
	if state == nil || state.scheduler == nil do return

	handle := state.gpu_complete
	if handle == Core.EXTERNAL_NODE_HANDLE_INVALID do return

	ok := state.scheduler.external_wait_for_system_name(
		state.scheduler,
		handle,
		"BF_GPU.FramePresent",
	)
	if !ok {
		// The frame's node reset has already happened by the time
		// hooks run, so a single failure here usually means the
		// FramePresent system isn't compiled into the DAG yet (an
		// early-frame race during module activation). Swallow it —
		// FramePresent will simply run without the external gate.
		log.debug("[BF_GPU] GPU completion wait could not be attached for FramePresent")
	}
}

// gpu_completion_setup wires the GPU completion external node +
// pre-frame hook on the BF_DAG scheduler service. Returns true on
// success; the renderer treats a false return as "BF_DAG unavailable,
// the renderer still works but the GPU gate is a no-op".
//
// Package-visible (no @(private)) so tests_gpu_completion.odin can
// exercise the wire-up against a stub Scheduler_Service.
gpu_completion_setup :: proc() -> bool {
	s := &MODULE_STATE_VALUE
	if s.scheduler == nil do return false
	if s.scheduler.external_create == nil do return false
	if s.scheduler.register_pre_frame_hook == nil do return false

	s.gpu_complete = s.scheduler.external_create(s.scheduler)
	if s.gpu_complete == Core.EXTERNAL_NODE_HANDLE_INVALID {
		log.warn("[BF_GPU] external_create returned invalid handle for GPU completion")
		return false
	}

	if !s.scheduler.register_pre_frame_hook(
		s.scheduler,
		gpu_completion_pre_frame_hook,
		rawptr(s),
	) {
		log.warn("[BF_GPU] failed to register GPU completion pre-frame hook")
		if s.scheduler.external_destroy != nil {
			s.scheduler.external_destroy(s.scheduler, s.gpu_complete)
		}
		s.gpu_complete = Core.EXTERNAL_NODE_HANDLE_INVALID
		return false
	}

	log.info("[BF_GPU] GPU completion external node wired into BF_DAG")
	return true
}

// gpu_completion_teardown removes the GPU completion external node +
// pre-frame hook. Called from renderer_shutdown.
//
// Package-visible so tests_gpu_completion.odin can clean up.
gpu_completion_teardown :: proc() {
	s := &MODULE_STATE_VALUE
	if s.scheduler == nil do return
	if s.gpu_complete != Core.EXTERNAL_NODE_HANDLE_INVALID &&
	   s.scheduler.external_destroy != nil {
		s.scheduler.external_destroy(s.scheduler, s.gpu_complete)
	}
	s.gpu_complete = Core.EXTERNAL_NODE_HANDLE_INVALID
}

// gpu_completion_signal is called from render_submit_step after the
// backend reports a successful record_frame. In v1 the submission is
// synchronous so the signal is immediate; a real GPU fence will defer
// this call until the fence fires. Either way the next frame's
// FramePresent gate releases.
//
// We also reset the node right after signaling so the subsequent
// pre-frame hook can re-arm a fresh wait. Reset clears the signaled
// flag and the waiter list (any leftover waiter from a previous
// frame would silently block the new frame's DAG node).
//
// Package-visible so tests_gpu_completion.odin can drive the
// signal/reset cycle.
gpu_completion_signal :: proc() {
	s := &MODULE_STATE_VALUE
	if s.scheduler == nil do return
	if s.gpu_complete == Core.EXTERNAL_NODE_HANDLE_INVALID do return

	if s.scheduler.external_signal != nil {
		s.scheduler.external_signal(s.scheduler, s.gpu_complete)
	}
	if s.scheduler.external_reset != nil {
		s.scheduler.external_reset(s.scheduler, s.gpu_complete)
	}
}



// ===========================================================================
//* Camera registry.
//
// Explicit active-camera semantics (prompt_plan §93): the renderer owns the
// camera set and the designation. Extraction never picks "the first camera it
// found". A registered entity only becomes a Render_Camera while it carries a
// BF_ECS Camera component.

renderer_register_camera :: proc(entity: ECS.Entity) -> bool {
	s := &MODULE_STATE_VALUE
	if !s.initialized || entity == ECS.ENTITY_INVALID do return false
	for existing in s.cameras {
		if existing == entity do return true
	}
	append(&s.cameras, entity)
	return true
}

renderer_unregister_camera :: proc(entity: ECS.Entity) -> bool {
	s := &MODULE_STATE_VALUE
	if !s.initialized do return false
	for existing, index in s.cameras {
		if existing == entity {
			ordered_remove(&s.cameras, index)
			if s.source.active_camera == entity do s.source.active_camera = ECS.ENTITY_INVALID
			if s.source.main_camera == entity do s.source.main_camera = ECS.ENTITY_INVALID
			return true
		}
	}
	return false
}

// The camera the frame is rendered from. Registers the entity if needed.
renderer_set_active_camera :: proc(entity: ECS.Entity) -> bool {
	s := &MODULE_STATE_VALUE
	if !s.initialized do return false
	if entity == ECS.ENTITY_INVALID {
		s.source.active_camera = ECS.ENTITY_INVALID
		return true
	}
	if !renderer_register_camera(entity) do return false
	s.source.active_camera = entity
	return true
}

// The camera culling and streaming are driven from. Defaults to the active
// camera when never set.
renderer_set_main_camera :: proc(entity: ECS.Entity) -> bool {
	s := &MODULE_STATE_VALUE
	if !s.initialized do return false
	if entity == ECS.ENTITY_INVALID {
		s.source.main_camera = ECS.ENTITY_INVALID
		return true
	}
	if !renderer_register_camera(entity) do return false
	s.source.main_camera = entity
	return true
}

renderer_active_camera :: proc() -> ECS.Entity {
	return MODULE_STATE_VALUE.scene.active_camera_entity
}

// Which chunks extraction walks. Defaults to the visible set.
renderer_set_extraction_scope :: proc(scope: Extraction_Scope) {
	MODULE_STATE_VALUE.source.scope = scope
}

renderer_extraction_stats :: proc() -> Extraction_Stats {
	return MODULE_STATE_VALUE.stats
}

renderer_render_scene :: proc() -> ^Render_Scene {
	return &MODULE_STATE_VALUE.scene
}

renderer_gpu_scene :: proc() -> ^GPU_Scene {
	return &MODULE_STATE_VALUE.gpu
}

renderer_resource_store :: proc() -> ^GPU_Resource_Store {
	return &MODULE_STATE_VALUE.store
}

// ===========================================================================
//* Per-frame hooks (called by the BF_DAG systems below).

// scene_extract_step runs at .PreRender. Walks the active/visible chunk set
// of the BF_ECS world and refreshes the renderer-owned Render_Scene.
scene_extract_step :: proc(raw_ctx: rawptr) {
	_ = raw_ctx
	s := &MODULE_STATE_VALUE
	if !s.initialized do return
	if s.world == nil do return

	s.source.world = s.world
	s.source.store = &s.store
	s.source.cameras = s.cameras[:]
	s.source.viewport_w = s.viewport_w
	s.source.viewport_h = s.viewport_h

	stats, ok := render_scene_extract(&s.scene, &s.source)
	if !ok {
		log.warn("[BF_GPU] scene extraction did not run (missing world or component bindings)")
		return
	}
	s.stats = stats
}

// render_upload_step runs at .Render_Upload. Walks the freshly extracted
// Render_Scene and produces the GPU_Scene mirror, refreshes the per-frame
// scalar + address mirror in Frame_Context_State, and leaves everything in
// place for the submit step to consume. No GPU backend calls happen here —
// the stage is purely CPU staging of GPU-side state.
render_upload_step :: proc(raw_ctx: rawptr) {
	_ = raw_ctx
	s := &MODULE_STATE_VALUE
	if !s.initialized do return

	s.gpu.settings = s.settings
	if !gpu_scene_update(&s.gpu, &s.scene, &s.store) {
		log.warn("[BF_GPU] GPU scene update failed")
		return
	}

	update_frame_scalar_state(
		&s.frame_ctx,
		&s.settings,
		s.gpu.frame_main_camera,
		s.gpu.frame_active_camera,
		s.gpu.frame_traditional_cmd_count,
		s.gpu.frame_meshlet_cmd_count,
		s.gpu.frame_indirect_cmd_count,
		s.gpu.frame_static_chunk_count,
		s.gpu.frame_model_count,
		s.gpu.frame_all_transforms,
		s.gpu.frame_static_transforms,
		s.viewport_w,
		s.viewport_h,
	)

	refresh_frame_addresses(&s.frame_ctx)
}

// render_submit_step runs at .Render_Submit. Hands the staged GPU_Scene
// to the backend for command buffer recording and advances the per-frame
// counter once the submission is accepted. On success it also signals
// the GPU completion external node (BF_DAG) so the next frame's
// FramePresent pre-frame wait releases. In v1 submission is
// synchronous and the signal fires immediately; a real GPU fence will
// defer the signal until the fence completes — the rest of the
// dependency chain stays the same.
render_submit_step :: proc(raw_ctx: rawptr) {
	_ = raw_ctx
	s := &MODULE_STATE_VALUE
	if !s.initialized do return

	if s.backend == nil do return
	if !s.backend.record_frame(&s.gpu, &s.frame_ctx) {
		log.warn("[BF_GPU] backend record_frame failed")
		return
	}

	frame_context_advance(&s.frame_ctx)

	// Mark the frame's GPU work as complete. Pre-frame hook for the
	// NEXT frame has already attached a wait on this node to that
	// frame's FramePresent; the wait auto-completes once the node
	// is signaled.
	gpu_completion_signal()
}

// frame_present_step runs at .PostRender. Reserved for editor overlays
// and the swapchain present hook. The actual GPU work is already
// submitted by the .Render_Submit step; this node is a no-op in v1 and
// fires after the GPU completion external node wired in by task 9.
frame_present_step :: proc(raw_ctx: rawptr) {
	_ = raw_ctx
}

// ===========================================================================
// System registration. Called from module_register once the component
// registration interface is available.
// ===========================================================================

renderer_register_systems :: proc() -> bool {
	// Order matters: each system uses a distinct stage, so the stage
	// barrier in BF_DAG/dag.odin enforces
	//   SceneExtract -> Render_Upload -> Render_Submit -> FramePresent
	// without explicit System_Dependency entries.
	if !Core.engine_register_system(
		"BF_GPU.SceneExtract",
		scene_extract_step,
		Core.System_Info{stage = .Render_Extract},
	) {
		log.warn("[BF_GPU] failed to register SceneExtract system")
	}

	if !Core.engine_register_system(
		"BF_GPU.Render_Upload",
		render_upload_step,
		Core.System_Info{stage = .Render_Upload},
	) {
		log.warn("[BF_GPU] failed to register Render_Upload system")
	}

	if !Core.engine_register_system(
		"BF_GPU.Render_Submit",
		render_submit_step,
		Core.System_Info{stage = .Render_Submit},
	) {
		log.warn("[BF_GPU] failed to register Render_Submit system")
	}

	if !Core.engine_register_system(
		"BF_GPU.FramePresent",
		frame_present_step,
		Core.System_Info{stage = .PostRender},
	) {
		log.warn("[BF_GPU] failed to register FramePresent system")
	}

	return true
}

//* Settings accessor for the editor / project-settings panel.

renderer_get_settings :: proc() -> ^GPU_Runtime_Settings {
	return &MODULE_STATE_VALUE.settings
}

renderer_apply_settings :: proc(settings: GPU_Runtime_Settings) {
	// MeshShaders cannot be flipped at runtime because the meshlet
	// pipeline objects either exist or don't; the editor must trigger
	// a renderer_recreate() call to pick up the change.
	was_mesh := MODULE_STATE_VALUE.settings.mesh_shaders
	MODULE_STATE_VALUE.settings = settings
	if was_mesh != settings.mesh_shaders {
		log.warnf(
			"[BF_GPU] MeshShaders changed (%v -> %v); restart renderer to take effect",
			was_mesh,
			settings.mesh_shaders,
		)
	}
}

renderer_set_viewport :: proc(w, h: f32) {
	s := &MODULE_STATE_VALUE
	s.viewport_w = max(w, 1.0)
	s.viewport_h = max(h, 1.0)
}

//* Internal helpers.

@(private = "file")
CORE_SERVICE_REGISTRY_PTR: ^Core.Service_Registry
CORE_LIB_CTX_PTR:          ^Core.Lib_Context

core_service_registry :: proc() -> ^Core.Service_Registry {
	return CORE_SERVICE_REGISTRY_PTR
}

renderer_set_lib_context :: proc(reg: ^Core.Service_Registry, ctx: ^Core.Lib_Context) {
	CORE_SERVICE_REGISTRY_PTR = reg
	CORE_LIB_CTX_PTR = ctx
}
