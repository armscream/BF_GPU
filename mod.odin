// Engine/src/Modules/Bifrost_Renderer/mod.odin
//
// PBR forward+ renderer module. This module OWNS its types and the
// Renderer.ExtensionPoint ABI; extensions targeting us import this
// package (or, more accurately, the Bifrost_Renderer package symbol set
// they need). See the ABI section below.
//
// The bifrost_lib_get_api entry point is gated on BUILDING_BF_RENDERER_DLL
// so that an extension DLL building against this package does NOT pull a
// duplicate `@(export) bifrost_lib_get_api` into its own DLL. rbs
// automatically passes that flag when building THIS DLL and omits it
// when building other DLLs (see Project/rbs/rbs.odin::component_build_flag).
package BF_Renderer

import "core:log"
import "../../Core"

// ============================================================================
//* RENDERER EXTENSION POINT (public ABI)
// ============================================================================
//
// Renderer_Extension_Point is the struct extensions receive when they
// call service_find("Renderer.ExtensionPoint"). Bifrost_Renderer owns
// this type; extensions that want to attach import this package so both
// sides agree on the layout.
//
// Layout note: every method takes ^Renderer_Extension_Point as the
// first argument so the renderer can store per-instance state later
// (Vulkan device, descriptor sets, etc.) without breaking ABI.
Renderer_Extension_Point :: struct {
	attach:           proc(ep: ^Renderer_Extension_Point, extension_name: cstring),
	detach:           proc(ep: ^Renderer_Extension_Point, extension_name: cstring),
	register_pass:    proc(ep: ^Renderer_Extension_Point, pass_name: cstring, pass_descriptor: rawptr) -> bool,
	register_material: proc(ep: ^Renderer_Extension_Point, material_name: cstring, material_descriptor: rawptr) -> bool,
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
renderer_ep_register_pass :: proc(
	ep: ^Renderer_Extension_Point,
	pass_name: cstring,
	pass_descriptor: rawptr,
) -> bool {
	_ = ep
	_ = pass_descriptor
	log.infof("[Renderer] Pass registered: %s", pass_name)
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

// new_renderer_extension_point allocates a Renderer_Extension_Point
// instance from the given allocator with the static stub
// implementations. The caller registers the pointer as a Core service
// in the renderer's register() callback.
new_renderer_extension_point :: proc(allocator := context.allocator) -> ^Renderer_Extension_Point {
	ep := new(Renderer_Extension_Point, allocator)
	ep.attach            = renderer_ep_attach
	ep.detach            = renderer_ep_detach
	ep.register_pass     = renderer_ep_register_pass
	ep.register_material = renderer_ep_register_material
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
	name           = "BF_Renderer",
	version        = Core.Version{0, 0, 1},
	author         = "armscream",
	description    = "PBR forward+ renderer with Vulkan and MoltenVK backends.",
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
	},
	dependency_count = 1,
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
when #config(BUILDING_BF_RENDERER_DLL, false) {
	@(export)
	bifrost_lib_get_api :: proc() -> ^Core.LIB_API {
		return &MODULE_API
	}
}

module_load :: proc(ctx: ^Core.Lib_Context) -> bool {
	_ = ctx
	log.info("[Renderer] loaded")
	return true
}

module_register :: proc(ctx: ^Core.Lib_Context) -> bool {
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

	log.info("[Renderer] Extension point service registered.")
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
		// Shouldn't happen — the engine always passes project_settings —
		// but if it does, fall back to safe values rather than dereferencing nil.
		log.warn("[Renderer] project_settings interface unavailable; using safe defaults.")
		fallback := Core.Renderer_Settings{
			texture_compression = .BC,
			max_texture_size    = 4096,
			lod_count           = 4,
			lod_simplification  = 0.5,
			index_buffer_format = .U32,
		}
		log.infof(
			"[Renderer] activating (fallback) — tex=%v maxTex=%d LODs=%d (simpl=%.2f) idxFmt=%v",
			fallback.texture_compression, fallback.max_texture_size,
			fallback.lod_count, fallback.lod_simplification, fallback.index_buffer_format,
		)
		return true
	}
	log.infof(
		"[Renderer] activating — tex=%v mips=%v maxTex=%d colour=%v LODs=%d (simpl=%.2f) idxFmt=%v octN=%v anim[rot=%v,tra=%v,scl=%v]",
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
	)
	return true
}

module_deactivate :: proc(ctx: ^Core.Lib_Context) {
	_ = ctx
}

module_unload :: proc(ctx: ^Core.Lib_Context) {
	_ = ctx
	log.info("[Renderer] unloaded")
}
