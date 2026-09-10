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
//   renderer_register_systems()
//       registers three BF_DAG systems:
//           BF_GPU.SceneExtract  stage .PreRender
//           BF_GPU.FrameRecord   stage .Render
//           BF_GPU.FramePresent  stage .PostRender
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
	scene:       Render_Scene_State,
	settings:    GPU_Runtime_Settings,
	views:       View_Inputs,
	initialized: bool,
	world:       ^ECS.World,
	backend:     ^GPU_Backend,
	viewport_w:  f32,
	viewport_h:  f32,
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
	record_frame:        proc(scene: ^Render_Scene_State, ctx: ^Frame_Context_State) -> bool,
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
	scene_state_init(&s.scene, s.settings, allocator)

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
	}

	// Persistent buffer creation is the backend's job. Once the backend
	// calls create_buffer() for every Gpu_Buffer_Kind, refresh_frame_addresses()
	// copies the resulting device addresses into the FrameGlobalContext
	// mirror so the culling shaders can read them.

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

	scene_state_destroy(&s.scene)
	frame_context_destroy(&s.frame_ctx)
	s^ = {}
	log.info("[BF_GPU] renderer shutdown")
}

// ===========================================================================
//* Per-frame hooks (called by the BF_DAG systems below).

// scene_extract_step runs at .PreRender. Walks the cached view pointers
// in MODULE_STATE_VALUE.views and refills the dense pools / sparse maps.
// Idempotent within a frame.
scene_extract_step :: proc(raw_ctx: rawptr) {
	_ = raw_ctx
	s := &MODULE_STATE_VALUE
	if !s.initialized do return
	if s.world == nil do return

	if s.views.transforms == nil {
		// Prefer the typed view pointer the world already exposes.
		if s.world.views.transforms != nil {
			s.views.transforms = s.world.views.transforms
		} else {
			s.views.transforms = world_view_find_by_name(s.world, "Renderer.Transforms")
		}
	}
	if s.views.cameras == nil {
		s.views.cameras = world_view_find_by_name(s.world, "Renderer.Cameras")
	}
	if s.views.tags == nil {
		s.views.tags = world_view_find_by_name(s.world, "Renderer.Tags")
	}
	if s.views.animations == nil {
		s.views.animations = world_view_find_by_name(s.world, "Renderer.Animations")
	}

	if !extract_scene_state(&s.scene, &s.views) {
		log.warn("[BF_GPU] scene extraction returned false")
	}
}

// frame_record_step runs at .Render. Updates the FrameGlobalContext
// mirror from the freshly extracted scene state and hands control to the
// backend for actual GPU command recording.
frame_record_step :: proc(raw_ctx: rawptr) {
	_ = raw_ctx
	s := &MODULE_STATE_VALUE
	if !s.initialized do return
	if s.backend == nil do return

	update_frame_scalar_state(
		&s.frame_ctx,
		&s.settings,
		s.scene.frame_main_camera,
		s.scene.frame_active_camera,
		s.scene.frame_traditional_cmd_count,
		s.scene.frame_meshlet_cmd_count,
		s.scene.frame_indirect_cmd_count,
		s.scene.frame_static_chunk_count,
		s.scene.frame_model_count,
		s.scene.frame_all_transforms,
		s.scene.frame_static_transforms,
		s.viewport_w,
		s.viewport_h,
	)

	refresh_frame_addresses(&s.frame_ctx)

	if !s.backend.record_frame(&s.scene, &s.frame_ctx) {
		log.warn("[BF_GPU] backend record_frame failed")
	}

	frame_context_advance(&s.frame_ctx)
}

// frame_present_step runs at .PostRender. Reserved for editor overlays;
// the backend's swapchain present handles actual submission.
frame_present_step :: proc(raw_ctx: rawptr) {
	_ = raw_ctx
}

// ===========================================================================
// System registration. Called from module_register once the component
// registration interface is available.
// ===========================================================================

renderer_register_systems :: proc() -> bool {
	if !Core.engine_register_system(
		"BF_GPU.SceneExtract",
		scene_extract_step,
		Core.System_Info{stage = .PreRender},
	) {
		log.warn("[BF_GPU] failed to register SceneExtract system")
	}

	if !Core.engine_register_system(
		"BF_GPU.FrameRecord",
		frame_record_step,
		Core.System_Info{stage = .Render},
	) {
		log.warn("[BF_GPU] failed to register FrameRecord system")
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
