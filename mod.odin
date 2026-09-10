// Engine/src/Modules/Bifrost_Renderer/mod.odin
//
// PBR forward+ renderer module. This module OWNS its types and the
// Renderer.ExtensionPoint ABI; extensions targeting us import this
// package (or, more accurately, the Bifrost_Renderer package symbol set
// they need). See the ABI section below.
//
// The bifrost_lib_get_api entry point is gated on BUILDING_BF_GPU_DLL
// so that an extension DLL building against this package does NOT pull a
// duplicate `@(export) bifrost_lib_get_api` into its own DLL. rbs
// automatically passes that flag when building THIS DLL and omits it
// when building other DLLs (see Project/rbs/rbs.odin::component_build_flag).
package BF_GPU

import "core:log"
import "../../Core"

// TODO: Pull from persistent settings file that is game-facing and not project settings.
WINDOW_WIDTH := Core.GLOBAL_PROJECT_SETTINGS.renderer_settings.Window_Width
WINDOW_HEIGHT := Core.GLOBAL_PROJECT_SETTINGS.renderer_settings.Window_Height

// ============================================================================
//* RENDERER EXTENSION POINT (public ABI)
// ============================================================================
//
// Renderer_Extension_Point is the struct extensions receive when they
// call service_find("Renderer.ExtensionPoint"). Bifrost_Renderer owns
// this type; extensions that want to attach import this package so both
// sides agree on the layout.
//
// Extensions can register four kinds of contributions:
//   - compute_pass     (HiZ downsample, culling, Morton sort, ...)
//   - graphics_pass    (visibility buffer read, deferred material, ...)
//   - meshlet_pipeline (task + mesh shader pair for one meshlet slot)
//   - material         (PBR, NPR, stylized, ...; consumed by materials the
//                       material_instance registrations reference)
//
// Layout note: every method takes ^Renderer_Extension_Point as the
// first argument so the renderer can store per-instance state later
// (Vulkan device, descriptor sets, etc.) without breaking ABI.
Renderer_Extension_Point :: struct {
	attach:                proc(ep: ^Renderer_Extension_Point, extension_name: cstring),
	detach:                proc(ep: ^Renderer_Extension_Point, extension_name: cstring),
	register_compute_pass: proc(
		ep: ^Renderer_Extension_Point,
		pass_name: cstring,
		pass_descriptor: rawptr,
	) -> bool,
	register_graphics_pass: proc(
		ep: ^Renderer_Extension_Point,
		pass_name: cstring,
		pass_descriptor: rawptr,
	) -> bool,
	register_meshlet_pipeline: proc(
		ep: ^Renderer_Extension_Point,
		pipeline_name: cstring,
		pipeline_descriptor: rawptr,
	) -> bool,
	register_material: proc(
		ep: ^Renderer_Extension_Point,
		material_name: cstring,
		material_descriptor: rawptr,
	) -> bool,
	register_material_instance: proc(
		ep: ^Renderer_Extension_Point,
		instance_name: cstring,
		parent_material: cstring,
		overrides: rawptr,
	) -> bool,
}

// The single, engine-wide name of the service that exposes
// Renderer_Extension_Point. Extensions look this up via the SDK.
RENDERER_EXTENSION_POINT_SERVICE_NAME :: "Renderer.ExtensionPoint"

//* STATIC IMPLEMENTATIONS (stubs in v1)
@(private)
renderer_ep_attach :: proc(ep: ^Renderer_Extension_Point, extension_name: cstring) {
	_ = ep
	log.infof("[Renderer] Extension attached: %s", extension_name)
}

@(private)
renderer_ep_detach :: proc(ep: ^Renderer_Extension_Point, extension_name: cstring) {
	_ = ep
	log.infof("[Renderer] Extension detached: %s", extension_name)
}

@(private)
renderer_ep_register_compute_pass :: proc(
	ep: ^Renderer_Extension_Point,
	pass_name: cstring,
	pass_descriptor: rawptr,
) -> bool {
	_ = ep
	_ = pass_descriptor
	log.infof("[Renderer] Compute pass registered: %s", pass_name)
	return true
}

@(private)
renderer_ep_register_graphics_pass :: proc(
	ep: ^Renderer_Extension_Point,
	pass_name: cstring,
	pass_descriptor: rawptr,
) -> bool {
	_ = ep
	_ = pass_descriptor
	log.infof("[Renderer] Graphics pass registered: %s", pass_name)
	return true
}

@(private)
renderer_ep_register_meshlet_pipeline :: proc(
	ep: ^Renderer_Extension_Point,
	pipeline_name: cstring,
	pipeline_descriptor: rawptr,
) -> bool {
	_ = ep
	_ = pipeline_descriptor
	log.infof("[Renderer] Meshlet pipeline registered: %s", pipeline_name)
	return true
}

@(private)
renderer_ep_register_material :: proc(
	ep: ^Renderer_Extension_Point,
	material_name: cstring,
	material_descriptor: rawptr,
) -> bool {
	_ = ep
	_ = material_descriptor
	log.infof("[Renderer] Material registered: %s", material_name)
	return true
}

@(private)
renderer_ep_register_material_instance :: proc(
	ep: ^Renderer_Extension_Point,
	instance_name: cstring,
	parent_material: cstring,
	overrides: rawptr,
) -> bool {
	_ = ep
	_ = overrides
	log.infof("[Renderer] Material instance '%s' of '%s' registered", instance_name, parent_material)
	return true
}

// new_renderer_extension_point allocates a Renderer_Extension_Point
// instance from the given allocator with the static stub
// implementations. The caller registers the pointer as a Core service
// in the renderer's register() callback.
new_renderer_extension_point :: proc(allocator := context.allocator) -> ^Renderer_Extension_Point {
	ep := new(Renderer_Extension_Point, allocator)
	ep.attach                    = renderer_ep_attach
	ep.detach                    = renderer_ep_detach
	ep.register_compute_pass     = renderer_ep_register_compute_pass
	ep.register_graphics_pass    = renderer_ep_register_graphics_pass
	ep.register_meshlet_pipeline = renderer_ep_register_meshlet_pipeline
	ep.register_material         = renderer_ep_register_material
	ep.register_material_instance = renderer_ep_register_material_instance
	return ep
}

@(private)
destroy_renderer_extension_point :: proc(instance: rawptr) {
	if instance == nil do return
	free(cast(^Renderer_Extension_Point)instance, context.allocator)
}


// === MODULE_IDENTITY (parsed by rbs) ===
IDENTITY :: Core.Lib_Descriptor {
	api_version    = Core.LIB_API_VERSION,
	name           = "BF_GPU",
	version        = Core.Version{0, 0, 1},
	author         = "armscream",
	description    = "GPU-driven visibility-buffer renderer with Vulkan and MoltenVK backends.",
	component_kind = .Module,
	type           = .Renderer,
	flags          = {.Runtime},
	capabilities   = {.Renderer, .GPU, .Materials, .Textures},
	dependencies   = {
		{
			name            = "BF_DAG",
			min_version     = Core.Version{0, 0, 1},
			max_version     = Core.Version{9, 9, 9},
			has_max_version = true,
			has_min_version = true,
			optional        = false,
		},
		{
			name            = "BF_ECS",
			min_version     = Core.Version{0, 0, 1},
			max_version     = Core.Version{9, 9, 9},
			has_max_version = true,
			has_min_version = true,
			optional        = false,
		},
		{
			name            = "BF_Input",
			min_version     = Core.Version{0, 0, 1},
			max_version     = Core.Version{9, 9, 9},
			has_max_version = true,
			has_min_version = true,
			optional        = true,
		},
	},
	dependency_count = 3,
}
// === END MODULE_IDENTITY ===

MODULE_API := Core.LIB_API {
	descriptor = IDENTITY,
	load       = module_load,
	register   = module_register,
	activate   = module_activate,
	deactivate = module_deactivate,
	unload     = module_unload,
}


// `#config(NAME, default)` is the canonical Odin way to query a build-time define
when #config(BUILDING_BF_GPU_DLL, false) {
	@(export)
	bifrost_lib_get_api :: proc() -> ^Core.LIB_API {
		return &MODULE_API
	}
}

module_load :: proc(ctx: ^Core.Lib_Context) -> bool {
	_ = ctx
	// Each DLL has its own copy of `context.logger` (Odin duplicates
	// package globals per DLL). Create one so BF_GPU's log calls
	// write to the engine's stdout console alongside every other
	// module.
	context.logger = log.create_console_logger()
	log.info("[Renderer] loaded")
	return true
}

module_register :: proc(ctx: ^Core.Lib_Context) -> bool {
	// Service registry handle for downstream lookups (BF_ECS.World,
	// future Vulkan backend service, etc.). Stashed on Renderer.odin
	// state so renderer_init and the per-frame systems can resolve it
	// without taking an import dependency on Core package globals.
	reg_query := Core.lib_context_query(
		ctx,
		Core.CORE_LIB_INTERFACE_SERVICE_REGISTRY,
		Core.SERVICE_REGISTRY_API_VERSION,
	)
	reg := cast(^Core.Service_Registry)reg_query
	renderer_set_lib_context(reg, ctx)
	window_set_lib_context(reg, ctx)

	// Allocate the Renderer_Extension_Point service instance and hand it
	// to the Core service registry under
	// "Renderer.ExtensionPoint". Extensions targeting Bifrost_Renderer
	// look this service up and call attach() to wire themselves in.
	ep := new_renderer_extension_point()
	if ep == nil {
		log.error("[Renderer] Failed to allocate extension point.")
		return false
	}

	api_raw := Core.lib_context_query(
		ctx,
		Core.CORE_LIB_INTERFACE_COMPONENT_REGISTRATION,
		Core.COMPONENT_REGISTRATION_API_VERSION,
	)
	if api_raw == nil {
		log.error("[Renderer] component_registration interface unavailable.")
		destroy_renderer_extension_point(cast(rawptr)ep)
		return false
	}
	api := cast(^Core.Component_Registration_API)api_raw

	sreg := Core.Service_Registration {
		name     = RENDERER_EXTENSION_POINT_SERVICE_NAME,
		instance = cast(rawptr)ep,
		destroy  = destroy_renderer_extension_point,
	}
	if !api.add_service(ctx, sreg) {
		log.error("[Renderer] Failed to register extension point service.")
		destroy_renderer_extension_point(cast(rawptr)ep)
		return false
	}

	// Create the SDL3 window + register the Input_Backend service.
	// The actual SDL_CreateWindow call is wired in Window.odin; this
	// registers a service so BF_Input can find it.
	win_init := window_init(
		&WINDOW_STATE_VALUE,
		title = "Bifrost Engine",
		w     = WINDOW_WIDTH,
		h     = WINDOW_HEIGHT,
	)
	if !win_init {
		log.warn("[Renderer] window_init failed; running headless")
	}

	// Register the three DAG systems (SceneExtract, FrameRecord,
	// FramePresent). engine_register_system lives on the SDK; the
	// engine merges game-side and module-side registrations during
	// scheduler_build.
	renderer_register_systems()

	log.info("[Renderer] Extension point service + Input_Backend + 3 DAG systems registered.")
	return true
}

module_activate :: proc(ctx: ^Core.Lib_Context) -> bool {
	// IMPORTANT: must go through lib_context_query. Core's package globals
	// are duplicated into every DLL that imports Core; reading them
	// directly (`Core.renderer_settings_get()`) would return the DLL's
	// own zero copy. `renderer_settings_from_lib` walks the engine-owned
	// user_data pointer the engine passes to each DLL.
	settings := Core.renderer_settings_from_lib(ctx)
	if settings == nil {
		log.warn("[Renderer] project_settings interface unavailable; using safe defaults.")
		fallback := Core.Renderer_Settings{
			texture_compression = .BC,
			max_texture_size    = 4096,
			lod_count           = 4,
			lod_simplification  = 0.5,
			index_buffer_format = .U32,
		}
		settings = &fallback
	}

	// Translate the engine's Renderer_Settings into the GPU-side
	// GPU_Runtime_Settings. Mesh-shader support is NOT a project
	// setting; it is auto-detected from whether the BF_GPU_Meshlet
	// extension has attached (see detect_mesh_shaders_extension).
	gpu_settings := GPU_Runtime_Settings {
		mesh_shaders       = detect_mesh_shaders_extension(ctx),
		hiz_occlusion      = true,
		frustum_culling    = true,
		cone_culling       = true,
		async_compute_cull = false,
		gpu_sort_extension = .None,
		lod_count          = max(settings.lod_count, 1),
	}
	renderer_apply_settings(gpu_settings)

	if !renderer_init(gpu_settings) {
		log.error("[Renderer] renderer_init failed")
		return false
	}

	log.infof(
		"[Renderer] activating — tex=%v mips=%v maxTex=%d colour=%v LODs=%d (simpl=%.2f) idxFmt=%v octN=%v anim[rot=%v,tra=%v,scl=%v] meshShaders=%v",
		settings.texture_compression,
		settings.generate_mips,
		settings.max_texture_size,
		settings.colour_space,
		settings.lod_count,
		settings.lod_simplification,
		settings.index_buffer_format,
		settings.oct_encoded_normals,
		settings.animation_quantization.rotation,
		settings.animation_quantization.translation,
		settings.animation_quantization.scale,
		gpu_settings.mesh_shaders,
	)
	return true
}

module_deactivate :: proc(ctx: ^Core.Lib_Context) {
	_ = ctx
	renderer_shutdown()
}

module_unload :: proc(ctx: ^Core.Lib_Context) {
	_ = ctx
	window_destroy(&WINDOW_STATE_VALUE)
	unregister_gpu_backend()
	log.info("[Renderer] unloaded")
}

// detect_mesh_shaders_extension returns true if the BF_GPU_Meshlet
// extension has attached to the extension point at startup time. The
// renderer queries its own extension registry, not a project setting.
detect_mesh_shaders_extension :: proc(ctx: ^Core.Lib_Context) -> bool {
	if ctx == nil do return false
	raw := Core.lib_context_query(
		ctx,
		Core.CORE_LIB_INTERFACE_SERVICE_REGISTRY,
		Core.SERVICE_REGISTRY_API_VERSION,
	)
	if raw == nil do return false
	reg := cast(^Core.Service_Registry)raw

	// The extension point service exists; presence of BF_GPU_Meshlet
	// in [[extensions]] is reflected by an attached entry on the
	// extension point. For now we check via a separate flag the
	// extension writes when it calls ep.attach().
	handle, found := Core.service_find(reg, RENDERER_EXTENSION_POINT_SERVICE_NAME)
	if !found do return false
	ep_raw := Core.service_get(reg, handle)
	if ep_raw == nil do return false
	ep := cast(^Renderer_Extension_Point)ep_raw

	// Look up the meshlet pipeline in the extension's registry.
	// The extension writes its name into the extension point's
	// attached_extensions list when attach() is called; the renderer
	// reads that list. Until the extension point tracks attached
	// extensions by name, the lookup is a single service query.
	return meshlet_extension_attached(ep)
}

// meshlet_extension_attached is a stub. The real implementation walks
// the extension point's attached_extensions list and looks for one
// whose descriptor.name == "BF_GPU_Meshlet.Pipeline".
@(private)
meshlet_extension_attached :: proc(_: ^Renderer_Extension_Point) -> bool {
	// Until the extension point tracks attached extensions by name,
	// this returns false (no meshlet). When BF_GPU_Meshlet is wired
	// up properly, this becomes a lookup against an attached list.
	return false
}
