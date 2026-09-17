package BF_GPU

import vma "../../dependencies/odin-vma"
import "base:runtime"
import "core:log"
import "core:mem"
import "core:strings"
import sdl "vendor:sdl3"
import vk "vendor:vulkan"

//* EXT/CAPABILITIES listing.
BF_GPU_VULKAN_API_VERSION :: vk.API_VERSION_1_3

// BF_GPU_REQUIRED_DEVICE_EXTENSIONS are device extensions the renderer
// cannot operate without. KHR_swapchain is required for the swapchain
// WSI; the surface extensions are instance-level and requested by
// window_vulkan_instance_extensions(). Missing any of these is fatal.
BF_GPU_REQUIRED_DEVICE_EXTENSIONS := []cstring {
	"VK_KHR_swapchain",
}

// BF_GPU_OPTIONAL_DEVICE_EXTENSIONS are extensions the renderer asks the
// driver for in vulkan_create_device when available. Each entry exists
// because a feature consumes it (graphics pipeline libraries for shared
// shader/pipeline caches, memory budget/priority for VMA hints, fragment
// shading rate for traditional pass, descriptor buffer for bindless).
// Missing on the current hardware logs a warning and is skipped.
//
// VK_EXT_mesh_shader is the optional meshlet extension the BF_GPU_Mesh
// extension module consumes. When present, the renderer builds the
// meshlet (task + mesh shader) pipeline alongside the traditional
// vertex pipeline. When absent, the renderer stays on the traditional
// path and the BF_GPU_Mesh extension silently no-ops.
@(private)
BF_GPU_OPTIONAL_DEVICE_EXTENSIONS := []cstring {
	"VK_EXT_memory_budget",
	"VK_KHR_fragment_shading_rate",
	"VK_EXT_descriptor_buffer",
	"VK_EXT_graphics_pipeline_library",
	"VK_EXT_mesh_shader",
}
/////////////////////////////////////////////////

Vulkan_Queue_Family :: struct {
	graphics:     u32,
	compute:      u32,
	transfer:     u32,
	present:      u32,
	//
	has_graphics: bool,
	has_compute:  bool,
	has_transfer: bool,
	has_present:  bool,
}

//* SWAPCHAIN
Vulkan_Swapchain :: struct {
	handle:        vk.SwapchainKHR,
	format:        vk.Format,
	extent:        vk.Extent2D,
	images:        []vk.Image,
	image_views:   []vk.ImageView,
	image_layouts: []vk.ImageLayout,
	image_count:   u32,
}

Vulkan_Context :: struct {
	instance:                vk.Instance,
	physical_device:         vk.PhysicalDevice,
	device:                  vk.Device,
	//
	queues:                  Vulkan_Queue_Family,
	//
	graphics_queue:          vk.Queue,
	compute_queue:           vk.Queue,
	transfer_queue:          vk.Queue,
	present_queue:           vk.Queue,
	//
	surface:                 vk.SurfaceKHR,
	//
	swapchain:               Vulkan_Swapchain,
	// Memory
	allocator:               vma.Allocator,
	// timeline
	graphics_timeline:       vk.Semaphore,
	compute_timeline:        vk.Semaphore,
	transfer_timeline:       vk.Semaphore,
	graphics_timeline_value: u64,
	compute_timeline_value:  u64,
	transfer_timeline_value: u64,
	//
	frames:                  [MAX_FRAMES_IN_FLIGHT]Vulkan_Frame,
	frame_index:             u32,
	//
	// VK_EXT_mesh_shader support detected at device-creation time.
	// When true the renderer builds the optional meshlet pipeline
	// during vulkan_meshlet_init; when false the meshlet pass is
	// permanently skipped and BF_GPU_Mesh is a no-op regardless of
	// its own attach state. Detection is purely device-side; the
	// renderer-side "should we even try" question is a separate
	// check against the BF_GPU_Mesh extension attach list.
	mesh_shader_supported:   bool,
	// timeline_semaphore_available mirrors the physical device's
	// reported timelineSemaphore feature. Requesting the feature on a
	// device that does not support it is a hard vkCreateDevice error,
	// so the device-create chain only asks for it when the physical
	// device reports true. The flag is read by the diagnostics / frame
	// submission paths so a non-supporting device still runs (without
	// timeline-semaphore-driven completion tracking).
	timeline_semaphore_available: bool,
	// host_query_reset_available mirrors the physical device's
	// reported hostQueryReset feature. When false the diagnostics
	// module falls back to vkCmdResetQueryPool on a transient
	// command buffer; vkResetQueryPool is invalid without the
	// feature and the validation layer reports it as a hard error.
	host_query_reset_available:   bool,
	// descriptor_binding_sampled_image_update_after_bind_available
	// mirrors the physical device's reported
	// descriptorBindingSampledImageUpdateAfterBind feature. The
	// bindless texture array binding uses UPDATE_AFTER_BIND on a
	// SAMPLED_IMAGE descriptor type, which is only valid when this
	// feature is enabled. When unavailable the bindless texture
	// array falls back to a PARTIALLY_BOUND-only binding (no runtime
	// updates) and the renderer logs a warning.
	descriptor_binding_sampled_image_update_after_bind_available: bool,
	// Optional 1.2/1.3 + pre-1.1 features. Detection runs once in
	// vulkan_check_required_features; vulkan_create_device only
	// requests features the physical device actually exposes
	// (requesting a feature that is not supported is a hard
	// vkCreateDevice error on most drivers). Grouped here so the
	// renderer can branch on availability without re-querying.
	descriptor_buffer_available:                bool, // VK_EXT_descriptor_buffer
	mesh_shader_extension_available:            bool, // VK_EXT_mesh_shader extension presence
	scalar_block_layout_available:              bool,
	storage_buffer_8bit_access_available:       bool,
	uniform_and_storage_buffer_8bit_access_available: bool,
	vulkan_memory_model_available:              bool,
	separate_depth_stencil_layouts_available:   bool,
	subgroup_size_control_available:            bool,
	compute_full_subgroups_available:           bool,
	shader_float16_available:                   bool,
	shader_int8_available:                      bool,
	descriptor_binding_uniform_buffer_update_after_bind_available: bool,
	descriptor_binding_storage_buffer_update_after_bind_available: bool,
	descriptor_binding_storage_image_update_after_bind_available: bool,
	descriptor_binding_update_unused_while_pending_available: bool,
	shader_storage_buffer_array_non_uniform_indexing_available: bool,
	shader_sampled_image_array_non_uniform_indexing_available: bool,
	shader_uniform_buffer_array_non_uniform_indexing_available: bool,
	storage_push_constant_8_available:          bool,
	inline_uniform_block_available:             bool,
	private_data_available:                     bool,
	pipeline_creation_cache_control_available: bool,
	shader_demote_to_helper_invocation_available: bool,
	shader_terminate_invocation_available:      bool,
	maintenance_4_available:                    bool,
	multi_draw_indirect_available:              bool,
	draw_indirect_first_instance_available:     bool,
	depth_clamp_available:                      bool,
	depth_bias_clamp_available:                 bool,
	sampler_anisotropy_available:               bool,
	fragment_stores_and_atomics_available:      bool,
	vertex_pipeline_stores_and_atomics_available: bool,
	shader_int16_available:                     bool,
	shader_storage_image_read_without_format_available: bool,
	fill_mode_non_solid_available:              bool, // wireframe debug
	wide_lines_available:                       bool,
	large_points_available:                     bool,
	image_cube_array_available:                 bool,
	independent_blend_available:                bool,
	logic_op_available:                         bool,
	dual_src_blend_available:                   bool,
	multi_viewport_available:                   bool,
	alpha_to_one_available:                     bool,
	sample_rate_shading_available:              bool,
	occlusion_query_precise_available:          bool,
	pipeline_statistics_query_available:        bool,
	texture_compression_bc_available:           bool,
	texture_compression_etc2_available:         bool,
	texture_compression_astc_ldr_available:     bool,
	variable_multisample_rate_available:        bool,
	inherited_queries_available:                bool,
	shader_image_gather_extended_available:     bool,
	shader_storage_image_extended_formats_available: bool,
	shader_storage_image_multisample_available: bool,
	robust_buffer_access_available:             bool,
	// RTX 3080 always exposes these; the UHD 605 in the dev box
	// only exposes the VK_QUEUE_COMPUTE bit on the shared family.
	// Drives the dedicated transfer queue path (transfer_queue vs
	// graphics_queue sharing).
	dedicated_transfer_queue_available:         bool,
	dedicated_compute_queue_available:          bool,
	//
	initialized:             bool,
}

MAX_FRAMES_IN_FLIGHT :: 2

@(private)
VULKAN_STATE: Vulkan_Context

//////////////////////////////////////////////////////////////////////////////////////
//* LIFECYCLE CODE ------------------------------------------------------
vulkan_init :: proc(frame: ^Frame_Context_State) -> bool {
	if VULKAN_STATE.initialized {
		log.warn("[BF_GPU/Vulkan] vulkan_init called twice; ignoring")
		return true
	}
	if frame == nil {
		log.error("[BF_GPU/Vulkan] vulkan_init requires Frame_Context_State")
		return false
	}

	// Lay out the per-kind buffer descriptions before the persistent
	// buffers come up so gpu_buffer_descriptions can be called from
	// anywhere in the package.
	vulkan_buffer_layout_init()

	if !vulkan_create_instance() {
		log.error("[BF_GPU/Vulkan] instance creation failed")
		vulkan_shutdown(frame)
		return false
	}
	if !vulkan_create_surface() {
		log.error("[BF_GPU/Vulkan] surface creation failed")
		vulkan_shutdown(frame)
		return false
	}
	if !vulkan_pick_physical_device() {
		log.error("[BF_GPU/Vulkan] physical device selection failed")
		vulkan_shutdown(frame)
		return false
	}
	if !vulkan_create_device() {
		log.error("[BF_GPU/Vulkan] logical device creation failed")
		vulkan_shutdown(frame)
		return false
	}
	if !vulkan_create_allocator() {
		log.error("[BF_GPU/Vulkan] VMA allocator creation failed")
		vulkan_shutdown(frame)
		return false
	}
	if !vulkan_create_command_resources() {
		log.error("[BF_GPU/Vulkan] command pool/buffer creation failed")
		vulkan_shutdown(frame)
		return false
	}
	if !vulkan_create_sync_objects() {
		log.error("[BF_GPU/Vulkan] sync object creation failed")
		vulkan_shutdown(frame)
		return false
	}
	if !vulkan_create_swapchain() {
		log.error("[BF_GPU/Vulkan] swapchain creation failed")
		vulkan_shutdown(frame)
		return false
	}
	if !vulkan_create_swapchain_image_views() {
		log.error("[BF_GPU/Vulkan] swapchain image view creation failed")
		vulkan_shutdown(frame)
		return false
	}

	// Diagnostics: GPU timestamp query pool + debug-utils probe. The
	// init captures the runtime logger so the validation
	// system-callback can route to it; the probe is what decides
	// whether debug labels are wired at all.
	vulkan_diag_init_logger()
	vulkan_diag_init()
	vulkan_diag_probe_debug_utils()

	// Persistent renderer buffers. Created once per session at startup;
	// their device addresses feed the FrameGlobalContext mirror that
	// the culling / indirect draw shaders read every frame. Doing this
	// before the bindless descriptor init means the descriptor pool
	// can be sized against the actual buffer count, and the shader
	// cache can be initialised against real addresses.
	if !vulkan_init_renderer_buffers(frame) {
		log.error("[BF_GPU/Vulkan] persistent renderer buffer creation failed")
		vulkan_shutdown(frame)
		return false
	}

	if !vulkan_descriptor_init() {
		log.error("[BF_GPU/Vulkan] bindless descriptor model init failed")
		vulkan_shutdown(frame)
		return false
	}

	if !vulkan_shader_cache_init() {
		log.error("[BF_GPU/Vulkan] shader cache init failed")
		vulkan_shutdown(frame)
		return false
	}

	if !vulkan_pipeline_cache_init() {
		log.warn("[BF_GPU/Vulkan] pipeline cache init failed; pipelines will recompile every session")
	}

	if !vulkan_culling_init() {
		log.error("[BF_GPU/Vulkan] culling pipeline init failed")
		vulkan_shutdown(frame)
		return false
	}

	if !vulkan_traditional_init() {
		log.error("[BF_GPU/Vulkan] traditional indirect renderer init failed")
		vulkan_shutdown(frame)
		return false
	}

	// Optional meshlet (task + mesh shader) pipeline. The init is a
	// no-op when VK_EXT_mesh_shader is unavailable OR the BF_GPU_Mesh
	// extension has not registered a descriptor; the failure path
	// keeps the traditional pipeline functional and silently skips
	// the meshlet pass.
	vulkan_meshlet_init()

	vulkan_defer_init()

	VULKAN_STATE.initialized = true
	vulkan_register_backend()
	log.info("[BF_GPU/Vulkan] Vulkan backend initialized")
	return true
}

vulkan_shutdown :: proc(frame: ^Frame_Context_State) {
	// Always attempt cleanup; partial-init failure can leave a half-
	// built context (instance but no device, device but no swapchain,
	// ...). Each branch guards on the actual handle before touching
	// it, so calling this with VULKAN_STATE zero is a no-op.
	vulkan_unregister_backend()

	if VULKAN_STATE.device != nil {
		vk.DeviceWaitIdle(VULKAN_STATE.device)
	}

	// Drain any pending BF_DAG GPU-completion notifications while
	// the external node is still alive. After vkDeviceWaitIdle every
	// in-flight submission has completed, so any frame slot whose
	// completion_value > 0 must still need a release. Doing this
	// before the sync-object teardown keeps the BF_DAG external node
	// in renderer_shutdown() consistent with the last GPU submission.
	flushed := vulkan_flush_pending_completion_signals()
	if flushed > 0 {
		log.infof("[BF_GPU/Vulkan] Drained %d pending GPU-completion signals at shutdown", flushed)
	}

	// Diagnostics: release the GPU timestamp query pool + debug
	// messenger before the device goes away. The query pool handle
	// is opaque to this file; vulkan_diag_shutdown encapsulates
	// its own unsafe teardown.
	vulkan_diag_shutdown()

	// Reap everything still queued; the device wait above guarantees
	// no in-flight work references these resources.
	vulkan_pipeline_map_clear()
	vulkan_shader_cache_shutdown()
	vulkan_pipeline_cache_shutdown()
	vulkan_culling_shutdown()
	vulkan_traditional_shutdown()
	vulkan_meshlet_shutdown()
	// Tear down the descriptor model before any persistent images /
	// samplers are released; vulkan_descriptor_shutdown nulls the
	// descriptors that reference them.
	vulkan_descriptor_shutdown()
	vulkan_defer_shutdown()
	vulkan_clear_resource_maps()

	vulkan_destroy_swapchain()

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		frame := &VULKAN_STATE.frames[i]
		if VULKAN_STATE.device != nil {
			if frame.image_available !=
			   cast(vk.Semaphore)0 {vk.DestroySemaphore(VULKAN_STATE.device, frame.image_available, nil)}
			if frame.render_finished !=
			   cast(vk.Semaphore)0 {vk.DestroySemaphore(VULKAN_STATE.device, frame.render_finished, nil)}
			if frame.command_pool !=
			   cast(vk.CommandPool)0 {vk.DestroyCommandPool(VULKAN_STATE.device, frame.command_pool, nil)}
		}
		frame.completion_value = 0
		frame.completion_signaled = false
	}
	if VULKAN_STATE.device != nil {
		if VULKAN_STATE.graphics_timeline != cast(vk.Semaphore)0 {
			vk.DestroySemaphore(VULKAN_STATE.device, VULKAN_STATE.graphics_timeline, nil)
		}
		if VULKAN_STATE.compute_timeline != cast(vk.Semaphore)0 {
			vk.DestroySemaphore(VULKAN_STATE.device, VULKAN_STATE.compute_timeline, nil)
		}
		if VULKAN_STATE.transfer_timeline != cast(vk.Semaphore)0 {
			vk.DestroySemaphore(VULKAN_STATE.device, VULKAN_STATE.transfer_timeline, nil)
		}
	}
	VULKAN_STATE.graphics_timeline = {}
	VULKAN_STATE.compute_timeline = {}
	VULKAN_STATE.transfer_timeline = {}

	// Tear down any persistent GPU buffers the backend created. The
	// frame param is optional during partial-init failure: when nil
	// the proc releases the buffer objects but leaves any stale
	// Frame_Context_State handle entries as-is (the Frame_Context_State
	// is owned by the renderer module and goes away through its own
	// path).
	vulkan_destroy_renderer_buffers(frame)
	// Belt-and-braces: even when the frame param is nil, drain anything
	// else that ended up in the buffer map (a future GPU_Backend path
	// could create user buffers between init and destroy).
	clear(&VULKAN_BUFFER_MAP)
	VULKAN_BUFFER_NEXT_ID = 1

if VULKAN_STATE.allocator != nil {
		// Release the upload ring before the VMA allocator goes away;
		// the ring's VkBuffer + VmaAllocation must outlive no one
		// here but they must be torn down before VMA itself.
		vulkan_upload_ring_shutdown()
		vma.DestroyAllocator(VULKAN_STATE.allocator)
		VULKAN_STATE.allocator = nil
	}

	if VULKAN_STATE.device != nil {
		vk.DestroyDevice(VULKAN_STATE.device, nil)
		VULKAN_STATE.device = nil
	}
	if VULKAN_STATE.surface != cast(vk.SurfaceKHR)0 {
		vk.DestroySurfaceKHR(VULKAN_STATE.instance, VULKAN_STATE.surface, nil)
		VULKAN_STATE.surface = {}
	}
	if VULKAN_STATE.instance != nil {
		vk.DestroyInstance(VULKAN_STATE.instance, nil)
		VULKAN_STATE.instance = nil
	}
	VULKAN_STATE.queues = {}
	VULKAN_STATE.graphics_queue = {}
	VULKAN_STATE.compute_queue = {}
	VULKAN_STATE.transfer_queue = {}
	VULKAN_STATE.present_queue = {}
	VULKAN_STATE.physical_device = {}
	VULKAN_STATE.graphics_timeline_value = 0
	VULKAN_STATE.compute_timeline_value = 0
	VULKAN_STATE.transfer_timeline_value = 0
	VULKAN_STATE.frame_index = 0
	VULKAN_STATE.initialized = false
	VULKAN_STATE.mesh_shader_supported = false
	VULKAN_STATE.timeline_semaphore_available = false
	VULKAN_STATE.host_query_reset_available = false
	VULKAN_STATE.descriptor_binding_sampled_image_update_after_bind_available = false
	VULKAN_STATE.descriptor_buffer_available = false
	VULKAN_STATE.mesh_shader_extension_available = false
	VULKAN_STATE.scalar_block_layout_available = false
	VULKAN_STATE.storage_buffer_8bit_access_available = false
	VULKAN_STATE.uniform_and_storage_buffer_8bit_access_available = false
	VULKAN_STATE.vulkan_memory_model_available = false
	VULKAN_STATE.separate_depth_stencil_layouts_available = false
	VULKAN_STATE.subgroup_size_control_available = false
	VULKAN_STATE.compute_full_subgroups_available = false
	VULKAN_STATE.shader_float16_available = false
	VULKAN_STATE.shader_int8_available = false
	VULKAN_STATE.descriptor_binding_uniform_buffer_update_after_bind_available = false
	VULKAN_STATE.descriptor_binding_storage_buffer_update_after_bind_available = false
	VULKAN_STATE.descriptor_binding_storage_image_update_after_bind_available = false
	VULKAN_STATE.descriptor_binding_update_unused_while_pending_available = false
	VULKAN_STATE.shader_storage_buffer_array_non_uniform_indexing_available = false
	VULKAN_STATE.shader_sampled_image_array_non_uniform_indexing_available = false
	VULKAN_STATE.shader_uniform_buffer_array_non_uniform_indexing_available = false
	VULKAN_STATE.storage_push_constant_8_available = false
	VULKAN_STATE.inline_uniform_block_available = false
	VULKAN_STATE.private_data_available = false
	VULKAN_STATE.pipeline_creation_cache_control_available = false
	VULKAN_STATE.shader_demote_to_helper_invocation_available = false
	VULKAN_STATE.shader_terminate_invocation_available = false
	VULKAN_STATE.maintenance_4_available = false
	VULKAN_STATE.multi_draw_indirect_available = false
	VULKAN_STATE.draw_indirect_first_instance_available = false
	VULKAN_STATE.depth_clamp_available = false
	VULKAN_STATE.depth_bias_clamp_available = false
	VULKAN_STATE.sampler_anisotropy_available = false
	VULKAN_STATE.fragment_stores_and_atomics_available = false
	VULKAN_STATE.vertex_pipeline_stores_and_atomics_available = false
	VULKAN_STATE.shader_int16_available = false
	VULKAN_STATE.shader_storage_image_read_without_format_available = false
	VULKAN_STATE.fill_mode_non_solid_available = false
	VULKAN_STATE.wide_lines_available = false
	VULKAN_STATE.large_points_available = false
	VULKAN_STATE.image_cube_array_available = false
	VULKAN_STATE.independent_blend_available = false
	VULKAN_STATE.logic_op_available = false
	VULKAN_STATE.dual_src_blend_available = false
	VULKAN_STATE.multi_viewport_available = false
	VULKAN_STATE.alpha_to_one_available = false
	VULKAN_STATE.sample_rate_shading_available = false
	VULKAN_STATE.occlusion_query_precise_available = false
	VULKAN_STATE.pipeline_statistics_query_available = false
	VULKAN_STATE.texture_compression_bc_available = false
	VULKAN_STATE.texture_compression_etc2_available = false
	VULKAN_STATE.texture_compression_astc_ldr_available = false
	VULKAN_STATE.variable_multisample_rate_available = false
	VULKAN_STATE.inherited_queries_available = false
	VULKAN_STATE.shader_image_gather_extended_available = false
	VULKAN_STATE.shader_storage_image_extended_formats_available = false
	VULKAN_STATE.shader_storage_image_multisample_available = false
	VULKAN_STATE.robust_buffer_access_available = false
	VULKAN_STATE.dedicated_transfer_queue_available = false
	VULKAN_STATE.dedicated_compute_queue_available = false
}

// vulkan_run_vma_defragmentation kicks off a single VMA defrag pass
// against the default GPU_ONLY pool. Cheap on idle heaps, no-op when
// there is nothing to move. Called by the diagnostics layer on a long
// period; safe to call any time the GPU is not mid-submit.
vulkan_run_vma_defragmentation :: proc() {
	if VULKAN_STATE.allocator == nil do return
	if !VULKAN_STATE.initialized do return

	// Fast path: only defragment when there's measurable free space in
	// the default pool. vma.CalculateStatistics is the slow-but-thorough
	// query; vma.GetHeapBudgets is the cheap alternative but doesn't
	// report fragmentation. Defrag cost is proportional to the number
	// of movable allocations, not the heap size, so the 25%-wasted
	// threshold is a reasonable "don't defrag for trivial gains".
	stats: vma.TotalStatistics
	vma.CalculateStatistics(VULKAN_STATE.allocator, &stats)
	unused_bytes := stats.total.statistics.blockBytes - stats.total.statistics.allocationBytes
	if unused_bytes < (stats.total.statistics.blockBytes / 4) {
		return
	}

	info: vma.DefragmentationInfo
	info.flags = {.ALGORITHM_FAST}
	info.pool = nil // all pools
	vma.BeginDefragmentation(VULKAN_STATE.allocator, info, &defrag_ctx)
	stats_out: vma.DefragmentationStats
	vma.EndDefragmentation(VULKAN_STATE.allocator, defrag_ctx, &stats_out)
	if stats_out.allocationsMoved > 0 || stats_out.bytesMoved > 0 {
		log.infof(
			"[BF_GPU/Vulkan] VMA defrag moved %d allocations (%d bytes)",
			stats_out.allocationsMoved,
			stats_out.bytesMoved,
		)
	}
}

// Shims that expose VULKAN_STATE.allocator + physical_device to the
// diagnostics module. The diagnostics file imports vma + vk already;
// these procs satisfy the @(private) declarations in Diagnostics.odin.
@(private)
allocator_accessor :: proc() -> vma.Allocator {
	return VULKAN_STATE.allocator
}
@(private)
physical_device_accessor :: proc() -> vk.PhysicalDevice {
	return VULKAN_STATE.physical_device
}

@(private)
defrag_ctx: vma.DefragmentationContext

//* ---------------------------------------------------------------------

vulkan_create_instance :: proc() -> bool {
	app_info := vk.ApplicationInfo {
		sType              = .APPLICATION_INFO,
		pApplicationName   = "Bifrost",
		applicationVersion = vk.MAKE_VERSION(0, 0, 1),
		pEngineName        = "Bifrost Engine",
		engineVersion      = vk.MAKE_VERSION(0, 0, 1),
		apiVersion         = vk.API_VERSION_1_3,
	}

	// Combine the SDL-required surface extensions with the
	// diagnostics-side debug-utils extension. The latter is only
	// added when validation is enabled, so a release build never
	// pays for the debug layer / extension chain.
	sdl_extensions := window_vulkan_instance_extensions()
	debug_extensions := vulkan_debug_utils_extension_names()

	total_extensions := len(sdl_extensions) + len(debug_extensions)
	extensions := make([]cstring, total_extensions)
	defer delete(extensions)
	copy(extensions, sdl_extensions)
	for ext, i in debug_extensions {
		extensions[len(sdl_extensions) + i] = ext
	}

	validation_layers := vulkan_validation_layer_names()

	// The vendor:vulkan loader procs are zero until
	// load_proc_addresses_global is called with a working
	// vkGetInstanceProcAddr. SDL exposes the loader entry point via
	// Vulkan_GetVkGetInstanceProcAddr; we must invoke it before
	// vk.CreateInstance or the call dereferences a nil proc pointer
	// and the process aborts with STATUS_ACCESS_VIOLATION.
	vk_get_instance_proc_addr := sdl.Vulkan_GetVkGetInstanceProcAddr()
	if vk_get_instance_proc_addr == nil {
		log.error("[BF_GPU/Vulkan] SDL could not resolve vkGetInstanceProcAddr")
		return false
	}
	vk.load_proc_addresses_global(rawptr(vk_get_instance_proc_addr))

	create_info := vk.InstanceCreateInfo {
		sType                      = .INSTANCE_CREATE_INFO,
		pApplicationInfo           = &app_info,
		enabledExtensionCount      = u32(len(extensions)),
		ppEnabledExtensionNames    = raw_data(extensions) if len(extensions) > 0 else nil,
		enabledLayerCount          = u32(len(validation_layers)),
		ppEnabledLayerNames        = raw_data(validation_layers) if len(validation_layers) > 0 else nil,
	}

	result := vk.CreateInstance(&create_info, nil, &VULKAN_STATE.instance)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkCreateInstance failed: %v", result)
		return false
	}

	vk.load_proc_addresses_instance(VULKAN_STATE.instance)
	log.infof(
		"[BF_GPU/Vulkan] Instance created: extensions=%d layers=%d",
		len(extensions),
		len(validation_layers),
	)
	return true
}

//* PICK PHYSICAL DEVICE
// prior to logical device creation
vulkan_pick_physical_device :: proc() -> bool {
	count: u32
	result := vk.EnumeratePhysicalDevices(VULKAN_STATE.instance, &count, nil)
	if result != .SUCCESS || count == 0 {
		log.errorf("[BF_GPU/Vulkan] no Vulkan physical devices found")
		return false
	}

	devices := make([]vk.PhysicalDevice, count)
	defer delete(devices)

	result = vk.EnumeratePhysicalDevices(VULKAN_STATE.instance, &count, raw_data(devices))
	if result != .SUCCESS do return false

	for device in devices {
		if vulkan_device_is_suitable(device) {
			VULKAN_STATE.physical_device = device
			return true
		}
	}
	log.error("[BF_GPU/Vulkan] no suitable Vulkan device found")
	return false
}
//* DEVICE QUERIES ========================================================
vulkan_device_is_suitable :: proc(device: vk.PhysicalDevice) -> bool {
	queues := vulkan_find_queue_families(device)
	if !queues.has_graphics do return false
	if !queues.has_present do return false
	// Compute / transfer are NOT a hard requirement: a shared graphics
	// family that supports both is sufficient and is the common case on
	// integrated GPUs. vulkan_find_queue_families falls back to the
	// graphics family when no dedicated compute/transfer family exists,
	// so has_compute / has_transfer may be set on the graphics family.
	// If neither a dedicated nor a graphics-shared family supplied them,
	// the renderer cannot do compute work - reject the device.
	if !queues.has_compute do return false
	if !queues.has_transfer do return false

	if !vulkan_check_required_extensions(device) do return false
	if !vulkan_check_required_features(device) do return false

	// if !vulkan_check_device_extensions(device) do return false
	VULKAN_STATE.queues = queues
	return true
}
vulkan_enumerate_device_extensions :: proc(device: vk.PhysicalDevice) -> ([dynamic]string, bool) {
	count: u32
	result := vk.EnumerateDeviceExtensionProperties(device, nil, &count, nil)
	if result != .SUCCESS || count == 0 do return nil, false
	properties := make([]vk.ExtensionProperties, count)
	defer delete(properties)

	result = vk.EnumerateDeviceExtensionProperties(device, nil, &count, raw_data(properties))
	if result != .SUCCESS do return nil, false

	names: [dynamic]string

	for i in 0 ..< len(properties) {
		cstr := cstring(raw_data(properties[i].extensionName[:]))
		append(&names, strings.clone(string(cstr)))
	}
	return names, true
}
vulkan_has_device_extension :: proc(device: vk.PhysicalDevice, required: cstring) -> bool {
	count: u32
	result := vk.EnumerateDeviceExtensionProperties(device, nil, &count, nil)
	if result != .SUCCESS || count == 0 do return false

	properties := make([]vk.ExtensionProperties, count)
	defer delete(properties)

	result = vk.EnumerateDeviceExtensionProperties(device, nil, &count, raw_data(properties))
	if result != .SUCCESS do return false

	for i in 0 ..< len(properties) {
		if extension_name_equal(properties[i].extensionName[:], required) do return true
	}
	return false
}
extension_name_equal :: proc(name: []byte, other: cstring) -> bool {
	a := name
	b := string(other)
	if len(b) > len(a) do return false
	for i in 0 ..< len(b) {
		if a[i] != b[i] do return false
	}
	return a[len(b)] == 0
}
vulkan_check_required_extensions :: proc(device: vk.PhysicalDevice) -> bool {
	// Required device extensions: a missing KHR_swapchain is fatal;
	// the renderer cannot present without it.
	for extension in BF_GPU_REQUIRED_DEVICE_EXTENSIONS {
		if !vulkan_has_device_extension(device, extension) {
			log.errorf("[BF_GPU/Vulkan] Missing required device extension: %s", extension)
			return false
		}
	}
	// Optional extensions: warn but do not reject. Tasks that
	// actually consume them (#4 pipelines, #8 fragment shading)
	// gate their own behaviour on the resulting device flag.
	for extension in BF_GPU_OPTIONAL_DEVICE_EXTENSIONS {
		if !vulkan_has_device_extension(device, extension) {
			log.warnf("[BF_GPU/Vulkan] Optional device extension unavailable: %s", extension)
		}
	}
	return true
}
vulkan_check_required_features :: proc(device: vk.PhysicalDevice) -> bool {
	features13 := vk.PhysicalDeviceVulkan13Features {
		sType = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
	}
	features12 := vk.PhysicalDeviceVulkan12Features {
		sType = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		pNext = &features13,
	}
	features2 := vk.PhysicalDeviceFeatures2 {
		sType = .PHYSICAL_DEVICE_FEATURES_2,
		pNext = &features12,
	}
	vk.GetPhysicalDeviceFeatures2(device, &features2)
	if !features13.dynamicRendering {
		log.error("[BF_GPU/Vulkan] Dynamic rendering is required but not supported")
		return false
	}
	if !features13.synchronization2 {
		log.error("[BF_GPU/Vulkan] Synchronization2 is required but not supported")
		return false
	}
	if !features12.bufferDeviceAddress {
		log.error("[BF_GPU/Vulkan] Buffer device address is required but not supported")
		return false
	}
	if !features12.descriptorIndexing {
		log.error("[BF_GPU/Vulkan] Descriptor indexing is required but not supported")
		return false
	}
	if !features12.runtimeDescriptorArray {
		log.error("[BF_GPU/Vulkan] Runtime descriptor array is required but not supported")
		return false
	}
	if !features12.descriptorBindingVariableDescriptorCount {
		log.error(
			"[BF_GPU/Vulkan] Descriptor binding variable descriptor count is required but not supported",
		)
		return false
	}
	if !features12.drawIndirectCount {
		log.error("[BF_GPU/Vulkan] Draw indirect count is required but not supported")
		return false
	}
	// Optional but driver-cheap features. Record availability so the
	// device-create chain only requests features the physical device
	// actually exposes. Requesting a feature that is not supported is
	// a hard vkCreateDevice error on most drivers and silently disables
	// the feature on others (Intel UHD 605 has been observed reporting
	// hostQueryReset = true in the feature query but failing the
	// device create when the same feature is requested via pNext).
	//
	// The capture below replaces the prior "set fields to zero, let
	// validation silently force them on" path. RTX 3080 (Ampere)
	// exposes every feature in this list; the UHD 605 fallback path
	// simply skips the unsupported ones. The device-create pNext
	// chain reads these booleans to decide which bits to request.
	VULKAN_STATE.timeline_semaphore_available = features12.timelineSemaphore == true
	if !VULKAN_STATE.timeline_semaphore_available {
		log.warn("[BF_GPU/Vulkan] timelineSemaphore not supported; timeline-based completion tracking disabled")
	}
	VULKAN_STATE.host_query_reset_available = features12.hostQueryReset == true
	if !VULKAN_STATE.host_query_reset_available {
		log.warn("[BF_GPU/Vulkan] hostQueryReset not supported; falling back to vkCmdResetQueryPool")
	}
	VULKAN_STATE.descriptor_binding_sampled_image_update_after_bind_available = features12.descriptorBindingSampledImageUpdateAfterBind == true
	if !VULKAN_STATE.descriptor_binding_sampled_image_update_after_bind_available {
		log.warn("[BF_GPU/Vulkan] descriptorBindingSampledImageUpdateAfterBind not supported; bindless texture array may be partially-bound only")
	}

	// 1.2 features the renderer wants when the driver offers them.
	VULKAN_STATE.scalar_block_layout_available = features12.scalarBlockLayout == true
	VULKAN_STATE.storage_buffer_8bit_access_available = features12.storageBuffer8BitAccess == true
	VULKAN_STATE.uniform_and_storage_buffer_8bit_access_available = features12.uniformAndStorageBuffer8BitAccess == true
	VULKAN_STATE.vulkan_memory_model_available = features12.vulkanMemoryModel == true
	VULKAN_STATE.separate_depth_stencil_layouts_available = features12.separateDepthStencilLayouts == true
	VULKAN_STATE.shader_float16_available = features12.shaderFloat16 == true
	VULKAN_STATE.shader_int8_available = features12.shaderInt8 == true
	VULKAN_STATE.descriptor_binding_uniform_buffer_update_after_bind_available = features12.descriptorBindingUniformBufferUpdateAfterBind == true
	VULKAN_STATE.descriptor_binding_storage_buffer_update_after_bind_available = features12.descriptorBindingStorageBufferUpdateAfterBind == true
	VULKAN_STATE.descriptor_binding_storage_image_update_after_bind_available = features12.descriptorBindingStorageImageUpdateAfterBind == true
	VULKAN_STATE.descriptor_binding_update_unused_while_pending_available = features12.descriptorBindingUpdateUnusedWhilePending == true
	VULKAN_STATE.shader_storage_buffer_array_non_uniform_indexing_available = features12.shaderStorageBufferArrayNonUniformIndexing == true
	VULKAN_STATE.shader_sampled_image_array_non_uniform_indexing_available = features12.shaderSampledImageArrayNonUniformIndexing == true
	VULKAN_STATE.shader_uniform_buffer_array_non_uniform_indexing_available = features12.shaderUniformBufferArrayNonUniformIndexing == true
	VULKAN_STATE.storage_push_constant_8_available = features12.storagePushConstant8 == true

	// 1.3 features the renderer wants when the driver offers them.
	VULKAN_STATE.subgroup_size_control_available = features13.subgroupSizeControl == true
	VULKAN_STATE.compute_full_subgroups_available = features13.computeFullSubgroups == true
	VULKAN_STATE.inline_uniform_block_available = features13.inlineUniformBlock == true
	VULKAN_STATE.private_data_available = features13.privateData == true
	VULKAN_STATE.pipeline_creation_cache_control_available = features13.pipelineCreationCacheControl == true
	VULKAN_STATE.shader_demote_to_helper_invocation_available = features13.shaderDemoteToHelperInvocation == true
	VULKAN_STATE.shader_terminate_invocation_available = features13.shaderTerminateInvocation == true
	VULKAN_STATE.maintenance_4_available = features13.maintenance4 == true

	// Pre-1.1 features that were previously left to the validation
	// layer's "force ON" pass. The RTX 3080 build requests all of
	// these; the UHD 605 build only requests the ones the device
	// reports as available.
	VULKAN_STATE.multi_draw_indirect_available = features2.features.multiDrawIndirect == true
	VULKAN_STATE.draw_indirect_first_instance_available = features2.features.drawIndirectFirstInstance == true
	VULKAN_STATE.depth_clamp_available = features2.features.depthClamp == true
	VULKAN_STATE.depth_bias_clamp_available = features2.features.depthBiasClamp == true
	VULKAN_STATE.sampler_anisotropy_available = features2.features.samplerAnisotropy == true
	VULKAN_STATE.fragment_stores_and_atomics_available = features2.features.fragmentStoresAndAtomics == true
	VULKAN_STATE.vertex_pipeline_stores_and_atomics_available = features2.features.vertexPipelineStoresAndAtomics == true
	VULKAN_STATE.shader_int16_available = features2.features.shaderInt16 == true
	VULKAN_STATE.shader_storage_image_read_without_format_available = features2.features.shaderStorageImageReadWithoutFormat == true
	VULKAN_STATE.fill_mode_non_solid_available = features2.features.fillModeNonSolid == true
	VULKAN_STATE.wide_lines_available = features2.features.wideLines == true
	VULKAN_STATE.large_points_available = features2.features.largePoints == true
	VULKAN_STATE.image_cube_array_available = features2.features.imageCubeArray == true
	VULKAN_STATE.independent_blend_available = features2.features.independentBlend == true
	VULKAN_STATE.logic_op_available = features2.features.logicOp == true
	VULKAN_STATE.dual_src_blend_available = features2.features.dualSrcBlend == true
	VULKAN_STATE.multi_viewport_available = features2.features.multiViewport == true
	VULKAN_STATE.alpha_to_one_available = features2.features.alphaToOne == true
	VULKAN_STATE.sample_rate_shading_available = features2.features.sampleRateShading == true
	VULKAN_STATE.occlusion_query_precise_available = features2.features.occlusionQueryPrecise == true
	VULKAN_STATE.pipeline_statistics_query_available = features2.features.pipelineStatisticsQuery == true
	VULKAN_STATE.texture_compression_bc_available = features2.features.textureCompressionBC == true
	VULKAN_STATE.texture_compression_etc2_available = features2.features.textureCompressionETC2 == true
	VULKAN_STATE.texture_compression_astc_ldr_available = features2.features.textureCompressionASTC_LDR == true
	VULKAN_STATE.variable_multisample_rate_available = features2.features.variableMultisampleRate == true
	VULKAN_STATE.inherited_queries_available = features2.features.inheritedQueries == true
	VULKAN_STATE.shader_image_gather_extended_available = features2.features.shaderImageGatherExtended == true
	VULKAN_STATE.shader_storage_image_extended_formats_available = features2.features.shaderStorageImageExtendedFormats == true
	VULKAN_STATE.shader_storage_image_multisample_available = features2.features.shaderStorageImageMultisample == true
	VULKAN_STATE.robust_buffer_access_available = features2.features.robustBufferAccess == true

	// Extension presence. VK_EXT_descriptor_buffer is the RTX 3080
	// bindless path; the renderer falls back to descriptor-indexing +
	// UPDATE_AFTER_BIND pool when the extension is absent (the
	// UHD 605 case).
	VULKAN_STATE.descriptor_buffer_available = vulkan_has_device_extension(
		device,
		"VK_EXT_descriptor_buffer",
	)
	if VULKAN_STATE.descriptor_buffer_available {
		log.info("[BF_GPU/Vulkan] VK_EXT_descriptor_buffer present; descriptor-buffer bindless path will be selected")
	}
	VULKAN_STATE.mesh_shader_extension_available = vulkan_has_device_extension(
		device,
		"VK_EXT_mesh_shader",
	)

	// Detect dedicated transfer / compute queue families. A device
	// with a non-graphics transfer family is what enables the
	// async asset-streaming path; without it everything collapses
	// onto the graphics queue. The dev box UHD 605 reports no
	// dedicated transfer family; RTX 3080 does.
	queue_count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nil)
	if queue_count > 0 {
		qprops := make([]vk.QueueFamilyProperties, queue_count)
		defer delete(qprops)
		vk.GetPhysicalDeviceQueueFamilyProperties(device, &queue_count, raw_data(qprops))
		VULKAN_STATE.dedicated_transfer_queue_available = false
		VULKAN_STATE.dedicated_compute_queue_available = false
		for qp in qprops {
			if .TRANSFER in qp.queueFlags && .GRAPHICS not_in qp.queueFlags && .COMPUTE not_in qp.queueFlags {
				VULKAN_STATE.dedicated_transfer_queue_available = true
			}
			if .COMPUTE in qp.queueFlags && .GRAPHICS not_in qp.queueFlags {
				VULKAN_STATE.dedicated_compute_queue_available = true
			}
		}
		if VULKAN_STATE.dedicated_transfer_queue_available {
			log.info("[BF_GPU/Vulkan] dedicated transfer queue family present; async asset streaming enabled")
		}
		if VULKAN_STATE.dedicated_compute_queue_available {
			log.info("[BF_GPU/Vulkan] dedicated compute queue family present; compute work can run off the graphics queue")
		}
	}
	return true
}
//* =======================================================================

//* SWAPCHAIN =============================================================
vulkan_query_surface_capabilities :: proc(
	device: vk.PhysicalDevice,
) -> (
	vk.SurfaceCapabilitiesKHR,
	bool,
) {
	capabilities := vk.SurfaceCapabilitiesKHR{}
	result := vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(
		device,
		VULKAN_STATE.surface,
		&capabilities,
	)
	if result != .SUCCESS {
		log.error("[BF_GPU/Vulkan] vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: %v", result)
		return capabilities, false
	}
	return capabilities, true
}
vulkan_query_surface_formats :: proc(device: vk.PhysicalDevice) -> ([]vk.SurfaceFormatKHR, bool) {
	count: u32
	result := vk.GetPhysicalDeviceSurfaceFormatsKHR(device, VULKAN_STATE.surface, &count, nil)
	if result != .SUCCESS || count == 0 {
		log.error("[BF_GPU/Vulkan] no surface formats available")
		return nil, false
	}

	formats := make([]vk.SurfaceFormatKHR, count)
	result = vk.GetPhysicalDeviceSurfaceFormatsKHR(
		device,
		VULKAN_STATE.surface,
		&count,
		raw_data(formats),
	)

	if result != .SUCCESS {
		delete(formats)
		log.errorf("[BF_GPU/Vulkan] failed querying surface formats: %v", result)
		return nil, false
	}

	return formats, true
}
vulkan_query_present_modes :: proc(device: vk.PhysicalDevice) -> ([]vk.PresentModeKHR, bool) {
	count: u32
	result := vk.GetPhysicalDeviceSurfacePresentModesKHR(device, VULKAN_STATE.surface, &count, nil)
	if result != .SUCCESS || count == 0 {
		log.error("[BF_GPU/Vulkan] no present modes available")
		return nil, false
	}
	modes := make([]vk.PresentModeKHR, count)

	result = vk.GetPhysicalDeviceSurfacePresentModesKHR(
		device,
		VULKAN_STATE.surface,
		&count,
		raw_data(modes),
	)
	if result != .SUCCESS {
		delete(modes)
		log.errorf("[BF_GPU/Vulkan] failed querying present modes: %v", result)
		return nil, false
	}
	return modes, true
}
vulkan_choose_surface_format :: proc(
	formats: []vk.SurfaceFormatKHR,
) -> (
	vk.SurfaceFormatKHR,
	bool,
) {
	for format in formats {
		if format.format == .B8G8R8A8_SRGB &&
		   format.colorSpace == .SRGB_NONLINEAR {return format, true}
	}
	// TODO: make swapchain format configurable via renderer settings
	// Fall back to the first format supplied by the implementation.
	if len(formats) > 0 {return formats[0], true}

	return {}, false
}
vulkan_choose_present_mode :: proc(modes: []vk.PresentModeKHR) -> vk.PresentModeKHR {
	for mode in modes {
		if mode == .MAILBOX {return .MAILBOX}
	}
	// Mailbox if available, otherwise FIFO.
	return .FIFO
}
vulkan_choose_swapchain_extent :: proc(
	capabilities: vk.SurfaceCapabilitiesKHR,
	window: Window_Handle,
) -> vk.Extent2D {
	if capabilities.currentExtent.width != 0xFFFFFFFF {
		return capabilities.currentExtent}

	width := window.width
	height := window.height

	if width < capabilities.minImageExtent.width {
		width = capabilities.minImageExtent.width}
	if width > capabilities.maxImageExtent.width {
		width = capabilities.maxImageExtent.width}

	if height < capabilities.minImageExtent.height {
		height = capabilities.minImageExtent.height}
	if height > capabilities.maxImageExtent.height {
		height = capabilities.maxImageExtent.height}

	return vk.Extent2D{width = width, height = height}
}
vulkan_create_swapchain :: proc() -> bool {
	device := VULKAN_STATE.physical_device
	capabilities, ok := vulkan_query_surface_capabilities(device)
	if !ok do return false
	formats, ok2 := vulkan_query_surface_formats(device)
	if !ok2 do return false
	defer delete(formats)
	present_modes, ok3 := vulkan_query_present_modes(device)
	if !ok3 do return false
	defer delete(present_modes)
	surface_format, ok4 := vulkan_choose_surface_format(formats)
	if !ok4 {
		log.error("[BF_GPU/Vulkan] Failed to choose surface format")
		return false
	}
	present_mode := vulkan_choose_present_mode(present_modes)
	window := window_get_handle()
	extent := vulkan_choose_swapchain_extent(capabilities, window)
	image_count := capabilities.minImageCount + 1

	if capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount {
		image_count = capabilities.maxImageCount}

	if VULKAN_STATE.swapchain.image_layouts != nil {
		delete(VULKAN_STATE.swapchain.image_layouts)
	}
	VULKAN_STATE.swapchain.image_layouts = make([]vk.ImageLayout, image_count)

	for i in 0 ..< image_count {
		VULKAN_STATE.swapchain.image_layouts[i] = .UNDEFINED
	}

	queue_indices := [2]u32{VULKAN_STATE.queues.graphics, VULKAN_STATE.queues.present}

	create_info := vk.SwapchainCreateInfoKHR {
		sType            = .SWAPCHAIN_CREATE_INFO_KHR,
		surface          = VULKAN_STATE.surface,
		minImageCount    = image_count,
		imageFormat      = surface_format.format,
		imageColorSpace  = surface_format.colorSpace,
		imageExtent      = extent,
		imageArrayLayers = 1,
		imageUsage       = {.COLOR_ATTACHMENT},
		preTransform     = capabilities.currentTransform,
		compositeAlpha   = {.OPAQUE},
		presentMode      = present_mode,
		clipped          = true,
		oldSwapchain     = {},
	}

	if VULKAN_STATE.queues.graphics != VULKAN_STATE.queues.present {
		create_info.imageSharingMode = .CONCURRENT
		create_info.queueFamilyIndexCount = 2
		create_info.pQueueFamilyIndices = &queue_indices[0]
	} else {
		create_info.imageSharingMode = .EXCLUSIVE
	}

	// vkCreateSwapchainKHR is resolved via the instance, but the
	// Odin vendor's load_proc_addresses_instance runs right after
	// CreateInstance when the loader has not yet finished
	// registering the swapchain proc and returns NULL. Re-run the
	// load with the now-stable instance so the swapchain-related
	// procs are populated. Safe to invoke repeatedly; it is a
	// pure var-set on the package-level procs.
	if cast(rawptr)vk.CreateSwapchainKHR == nil {
		vk.load_proc_addresses_instance(VULKAN_STATE.instance)
	}

	result := vk.CreateSwapchainKHR(
		VULKAN_STATE.device,
		&create_info,
		nil,
		&VULKAN_STATE.swapchain.handle,
	)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkCreateSwapchainKHR failed: %v", result)
		return false
	}

	VULKAN_STATE.swapchain.format = surface_format.format
	VULKAN_STATE.swapchain.extent = extent

	result = vk.GetSwapchainImagesKHR(
		VULKAN_STATE.device,
		VULKAN_STATE.swapchain.handle,
		&image_count,
		nil,
	)
	if result != .SUCCESS || image_count == 0 {
		log.errorf("[BF_GPU/Vulkan] vkGetSwapchainImagesKHR failed: %v", result)
		vulkan_destroy_swapchain()
		return false
	}
	VULKAN_STATE.swapchain.images = make([]vk.Image, image_count)
	result = vk.GetSwapchainImagesKHR(
		VULKAN_STATE.device,
		VULKAN_STATE.swapchain.handle,
		&image_count,
		raw_data(VULKAN_STATE.swapchain.images),
	)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkGetSwapchainImagesKHR failed: %v", result)
		delete(VULKAN_STATE.swapchain.images)
		VULKAN_STATE.swapchain.images = nil
		vk.DestroySwapchainKHR(VULKAN_STATE.device, VULKAN_STATE.swapchain.handle, nil)
		VULKAN_STATE.swapchain.handle = {}
		return false
	}

	VULKAN_STATE.swapchain.image_count = image_count
	for i in 0 ..< image_count {
		VULKAN_STATE.swapchain.image_layouts[i] = .UNDEFINED
	}

	log.infof(
		"[BF_GPU/Vulkan] Swapchain created: %dx%d, images=%d",
		extent.width,
		extent.height,
		image_count,
	)

	return true
}
vulkan_create_swapchain_image_views :: proc() -> bool {
	swapchain := &VULKAN_STATE.swapchain
	swapchain.image_views = make([]vk.ImageView, swapchain.image_count)

	for i in 0 ..< swapchain.image_count {
		create_info := vk.ImageViewCreateInfo {
			sType = .IMAGE_VIEW_CREATE_INFO,
			image = swapchain.images[i],
			viewType = .D2,
			format = swapchain.format,
			components = vk.ComponentMapping {
				r = .IDENTITY,
				g = .IDENTITY,
				b = .IDENTITY,
				a = .IDENTITY,
			},
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.COLOR},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		}
		result := vk.CreateImageView(
			VULKAN_STATE.device,
			&create_info,
			nil,
			&swapchain.image_views[i],
		)
		if result != .SUCCESS {
			log.errorf("[BF_GPU/Vulkan] vkCreateImageView for swapchain image %d: %v", i, result)
			for j in 0 ..< i {
				vk.DestroyImageView(VULKAN_STATE.device, swapchain.image_views[j], nil)
			}
			delete(swapchain.image_views)
			swapchain.image_views = nil
			return false
		}
	}
	return true
}
vulkan_destroy_swapchain :: proc() {
	swapchain := &VULKAN_STATE.swapchain
	for view in swapchain.image_views {
		if view != {} {
			vk.DestroyImageView(VULKAN_STATE.device, view, nil)
		}
	}
	delete(swapchain.image_views)
	swapchain.image_views = nil

	delete(swapchain.images)
	swapchain.images = nil

	delete(swapchain.image_layouts)
	swapchain.image_layouts = nil

	if swapchain.handle != {} {
		vk.DestroySwapchainKHR(VULKAN_STATE.device, swapchain.handle, nil)
		swapchain.handle = {}
	}

	swapchain^ = {}
}
//* SWAPCHAIN RUNTIME
vulkan_acquire_next_image :: proc(frame: ^Vulkan_Frame) -> (u32, vk.Result) {
	image_index: u32
	result := vk.AcquireNextImageKHR(
		VULKAN_STATE.device,
		VULKAN_STATE.swapchain.handle,
		0xFFFFFFFFFFFFFFFF,
		frame.image_available,
		{},
		&image_index,
	)
	return image_index, result
}
//* =====================================================================

vulkan_find_queue_families :: proc(device: vk.PhysicalDevice) -> Vulkan_Queue_Family {
	result := Vulkan_Queue_Family{}
	count: u32

	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, nil)
	properties := make([]vk.QueueFamilyProperties, count)
	defer delete(properties)

	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, raw_data(properties))

	// First pass: prefer dedicated compute/transfer families. The
	// QUEUE_TRANSFER / QUEUE_COMPUTE bits are set on virtually every
	// graphics family, so we filter on "no graphics" to find the
	// transfer-only family some discrete GPUs expose.
	for props, i in properties {
		family := u32(i)

		has_graphics := .GRAPHICS in props.queueFlags
		has_compute := .COMPUTE in props.queueFlags
		has_transfer := .TRANSFER in props.queueFlags

		if has_transfer && !has_graphics && !has_compute {
			result.transfer = family
			result.has_transfer = true
		}
		if has_compute && !has_graphics {
			result.compute = family
			result.has_compute = true
		}
	}

	// Second pass: fill general-purpose queues (graphics + present + any
	// missing compute/transfer). When the device has no dedicated
	// compute/transfer family (the integrated-GPU case), the graphics
	// family always supports both - fall back to it so the device is
	// still considered suitable.
	for props, i in properties {
		family := u32(i)
		if .GRAPHICS in props.queueFlags {
			if !result.has_graphics {
				result.graphics = family
				result.has_graphics = true
			}
			if !result.has_compute {
				result.compute = family
				result.has_compute = true
			}
			if !result.has_transfer {
				result.transfer = family
				result.has_transfer = true
			}
		}
		if sdl.Vulkan_GetPresentationSupport(VULKAN_STATE.instance, device, family) {
			if !result.has_present {
				result.present = family
				result.has_present = true
			}
		}
	}

	// Third pass: in case compute / transfer are supported by a non-
	// graphics, non-dedicated family (rare), still prefer them over the
	// graphics fallback so work can land on a less-loaded queue.
	for props, i in properties {
		family := u32(i)
		if .COMPUTE in props.queueFlags && family != result.graphics {
			if result.compute == result.graphics {
				result.compute = family
				result.has_compute = true
			}
		}
		if .TRANSFER in props.queueFlags && family != result.graphics {
			if result.transfer == result.graphics {
				result.transfer = family
				result.has_transfer = true
			}
		}
	}
	return result
}

//* Physical Device
vulkan_create_device :: proc() -> bool {
	queues := VULKAN_STATE.queues
	priority: f32 = 1.0
	queue_infos: [dynamic]vk.DeviceQueueCreateInfo

	vulkan_append_unique_queue_family(&queue_infos, queues.graphics, &priority)
	vulkan_append_unique_queue_family(&queue_infos, queues.compute, &priority)
	vulkan_append_unique_queue_family(&queue_infos, queues.transfer, &priority)
	vulkan_append_unique_queue_family(&queue_infos, queues.present, &priority)

	// Required + optional device extensions. Required extensions
	// (KHR_swapchain) are unconditionally enabled; optional ones
	// are filtered by what the physical device exposes. Forcing an
	// unavailable extension into VkDeviceCreateInfo is a hard
	// VK_ERROR_EXTENSION_NOT_PRESENT from vkCreateDevice.
	enabled_extensions: [dynamic]cstring
	defer delete(enabled_extensions)
	for extension in BF_GPU_REQUIRED_DEVICE_EXTENSIONS {
		append(&enabled_extensions, extension)
	}
	for extension in BF_GPU_OPTIONAL_DEVICE_EXTENSIONS {
		if vulkan_has_device_extension(VULKAN_STATE.physical_device, extension) {
			append(&enabled_extensions, extension)
		}
	}

	features13 := vk.PhysicalDeviceVulkan13Features {
		sType                        = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		dynamicRendering             = true,
		synchronization2             = true,
		subgroupSizeControl           = b32(VULKAN_STATE.subgroup_size_control_available),
		computeFullSubgroups          = b32(VULKAN_STATE.compute_full_subgroups_available),
		inlineUniformBlock            = b32(VULKAN_STATE.inline_uniform_block_available),
		privateData                   = b32(VULKAN_STATE.private_data_available),
		pipelineCreationCacheControl  = b32(VULKAN_STATE.pipeline_creation_cache_control_available),
		shaderDemoteToHelperInvocation = b32(VULKAN_STATE.shader_demote_to_helper_invocation_available),
		shaderTerminateInvocation     = b32(VULKAN_STATE.shader_terminate_invocation_available),
		maintenance4                  = b32(VULKAN_STATE.maintenance_4_available),
	}

	features12 := vk.PhysicalDeviceVulkan12Features {
		sType                                    = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		bufferDeviceAddress                      = true,
		descriptorIndexing                       = true,
		runtimeDescriptorArray                   = true,
		descriptorBindingPartiallyBound          = true,
		descriptorBindingVariableDescriptorCount = true,
		descriptorBindingSampledImageUpdateAfterBind = b32(
			VULKAN_STATE.descriptor_binding_sampled_image_update_after_bind_available,
		),
		drawIndirectCount                        = true,
		timelineSemaphore                        = b32(VULKAN_STATE.timeline_semaphore_available),
		hostQueryReset                           = b32(VULKAN_STATE.host_query_reset_available),
		scalarBlockLayout                        = b32(VULKAN_STATE.scalar_block_layout_available),
		storageBuffer8BitAccess                  = b32(VULKAN_STATE.storage_buffer_8bit_access_available),
		uniformAndStorageBuffer8BitAccess        = b32(VULKAN_STATE.uniform_and_storage_buffer_8bit_access_available),
		vulkanMemoryModel                        = b32(VULKAN_STATE.vulkan_memory_model_available),
		separateDepthStencilLayouts              = b32(VULKAN_STATE.separate_depth_stencil_layouts_available),
		shaderFloat16                            = b32(VULKAN_STATE.shader_float16_available),
		shaderInt8                               = b32(VULKAN_STATE.shader_int8_available),
		descriptorBindingUniformBufferUpdateAfterBind = b32(
			VULKAN_STATE.descriptor_binding_uniform_buffer_update_after_bind_available,
		),
		descriptorBindingStorageBufferUpdateAfterBind = b32(
			VULKAN_STATE.descriptor_binding_storage_buffer_update_after_bind_available,
		),
		descriptorBindingStorageImageUpdateAfterBind = b32(
			VULKAN_STATE.descriptor_binding_storage_image_update_after_bind_available,
		),
		descriptorBindingUpdateUnusedWhilePending   = b32(
			VULKAN_STATE.descriptor_binding_update_unused_while_pending_available,
		),
		shaderStorageBufferArrayNonUniformIndexing = b32(
			VULKAN_STATE.shader_storage_buffer_array_non_uniform_indexing_available,
		),
		shaderSampledImageArrayNonUniformIndexing = b32(
			VULKAN_STATE.shader_sampled_image_array_non_uniform_indexing_available,
		),
		shaderUniformBufferArrayNonUniformIndexing = b32(
			VULKAN_STATE.shader_uniform_buffer_array_non_uniform_indexing_available,
		),
		storagePushConstant8                       = b32(VULKAN_STATE.storage_push_constant_8_available),
	}
	features12.pNext = &features13

	// Pre-1.1 features. Previously these were left to the validation
	// layer's "WARNING-Setting-Limit-Adjusted" pass, which silently
	// forced them on at vkCreateDevice. Setting them explicitly here
	// makes the device-create intent legible and removes the need to
	// keep validation on to get correct feature state.
	//
	// PhysicalDeviceFeatures (pre-1.1) has no sType / pNext fields; the
	// 1.2/1.3 feature structs above are chained via DeviceCreateInfo's
	// pNext, while this struct is passed through pEnabledFeatures.
	features_v10 := vk.PhysicalDeviceFeatures {
		robustBufferAccess                   = b32(VULKAN_STATE.robust_buffer_access_available),
		fullDrawIndexUint32                   = true, // uint32 indices used by the renderer
		imageCubeArray                       = b32(VULKAN_STATE.image_cube_array_available),
		independentBlend                     = b32(VULKAN_STATE.independent_blend_available),
		multiDrawIndirect                    = b32(VULKAN_STATE.multi_draw_indirect_available),
		drawIndirectFirstInstance            = b32(VULKAN_STATE.draw_indirect_first_instance_available),
		depthClamp                           = b32(VULKAN_STATE.depth_clamp_available),
		depthBiasClamp                       = b32(VULKAN_STATE.depth_bias_clamp_available),
		fillModeNonSolid                     = b32(VULKAN_STATE.fill_mode_non_solid_available),
		wideLines                            = b32(VULKAN_STATE.wide_lines_available),
		largePoints                          = b32(VULKAN_STATE.large_points_available),
		alphaToOne                           = b32(VULKAN_STATE.alpha_to_one_available),
		multiViewport                        = b32(VULKAN_STATE.multi_viewport_available),
		samplerAnisotropy                    = b32(VULKAN_STATE.sampler_anisotropy_available),
		textureCompressionETC2               = b32(VULKAN_STATE.texture_compression_etc2_available),
		textureCompressionASTC_LDR           = b32(VULKAN_STATE.texture_compression_astc_ldr_available),
		textureCompressionBC                 = b32(VULKAN_STATE.texture_compression_bc_available),
		occlusionQueryPrecise                = b32(VULKAN_STATE.occlusion_query_precise_available),
		pipelineStatisticsQuery              = b32(VULKAN_STATE.pipeline_statistics_query_available),
		vertexPipelineStoresAndAtomics       = b32(VULKAN_STATE.vertex_pipeline_stores_and_atomics_available),
		fragmentStoresAndAtomics             = b32(VULKAN_STATE.fragment_stores_and_atomics_available),
		shaderImageGatherExtended            = b32(VULKAN_STATE.shader_image_gather_extended_available),
		shaderStorageImageExtendedFormats    = b32(VULKAN_STATE.shader_storage_image_extended_formats_available),
		shaderStorageImageMultisample        = b32(VULKAN_STATE.shader_storage_image_multisample_available),
		shaderStorageImageReadWithoutFormat  = b32(VULKAN_STATE.shader_storage_image_read_without_format_available),
		shaderStorageImageWriteWithoutFormat = true, // standard for compute writes
		shaderInt16                          = b32(VULKAN_STATE.shader_int16_available),
		sampleRateShading                    = b32(VULKAN_STATE.sample_rate_shading_available),
		dualSrcBlend                         = b32(VULKAN_STATE.dual_src_blend_available),
		logicOp                              = b32(VULKAN_STATE.logic_op_available),
		variableMultisampleRate              = b32(VULKAN_STATE.variable_multisample_rate_available),
		inheritedQueries                     = b32(VULKAN_STATE.inherited_queries_available),
	}

	// VK_EXT_mesh_shader feature chain. The driver only honours the
	// struct when VK_EXT_mesh_shader was added to the enabled
	// extensions above; absent the extension the struct is silently
	// ignored. The renderer keeps the supported flag separately so
	// vulkan_meshlet_init can short-circuit when the feature is not
	// present even though BF_GPU_Mesh attached.
	mesh_shader_features: vk.PhysicalDeviceMeshShaderFeaturesEXT
	mesh_shader_enabled := vulkan_has_device_extension(VULKAN_STATE.physical_device, "VK_EXT_mesh_shader")
	if mesh_shader_enabled {
		mesh_shader_features = vk.PhysicalDeviceMeshShaderFeaturesEXT {
			sType      = .PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT,
			taskShader = true,
			meshShader = true,
		}
		mesh_shader_features.pNext = nil
		features13.pNext = &mesh_shader_features
	}

	create_info := vk.DeviceCreateInfo {
		sType                   = .DEVICE_CREATE_INFO,
		queueCreateInfoCount    = u32(len(queue_infos)),
		pQueueCreateInfos       = raw_data(queue_infos),
		enabledExtensionCount   = u32(len(enabled_extensions)),
		ppEnabledExtensionNames = raw_data(enabled_extensions) if len(enabled_extensions) > 0 else nil,
		pEnabledFeatures        = &features_v10,
		pNext                   = &features12,
	}

	result := vk.CreateDevice(
		VULKAN_STATE.physical_device,
		&create_info,
		nil,
		&VULKAN_STATE.device,
	)
	if result != .SUCCESS {
		log.error("[BF_GPU/Vulkan] vkCreateDevice failed: %v", result)
		return false
	}

	vk.load_proc_addresses_device(VULKAN_STATE.device)
	vk.GetDeviceQueue(VULKAN_STATE.device, queues.graphics, 0, &VULKAN_STATE.graphics_queue)
	vk.GetDeviceQueue(VULKAN_STATE.device, queues.compute, 0, &VULKAN_STATE.compute_queue)
	vk.GetDeviceQueue(VULKAN_STATE.device, queues.transfer, 0, &VULKAN_STATE.transfer_queue)
	vk.GetDeviceQueue(VULKAN_STATE.device, queues.present, 0, &VULKAN_STATE.present_queue)

	// VK_EXT_mesh_shader detection. The extension being in the
	// enabled list is not enough on its own - the driver must also
	// report the taskShader / meshShader features as supported. We
	// trust the pNext chain built above to surface that; the boolean
	// flag here drives vulkan_meshlet_init's decision to compile
	// the optional meshlet pipeline.
	VULKAN_STATE.mesh_shader_supported = mesh_shader_enabled && mesh_shader_features.taskShader == true && mesh_shader_features.meshShader == true

	log.infof(
		"[BF_GPU/Vulkan] Device queues: graphics=%d compute=%d transfer=%d present=%d meshShaders=%v",
		queues.graphics,
		queues.compute,
		queues.transfer,
		queues.present,
		VULKAN_STATE.mesh_shader_supported,
	)
	return true
}

vulkan_append_unique_queue_family :: proc(
	infos: ^[dynamic]vk.DeviceQueueCreateInfo,
	family: u32,
	priority: ^f32,
) {
	for info in infos {if info.queueFamilyIndex == family do return}
	append(
		infos,
		vk.DeviceQueueCreateInfo {
			sType = .DEVICE_QUEUE_CREATE_INFO,
			queueFamilyIndex = family,
			queueCount = 1,
			pQueuePriorities = priority,
		},
	)
}

//* Create the command buffers per frame in flight (really just two atm, could do 3).
// A resettable per-frame pool is useful for re-recording every frame.
vulkan_create_command_resources :: proc() -> bool {
	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		frame := &VULKAN_STATE.frames[i]
		pool_info := vk.CommandPoolCreateInfo {
			sType            = .COMMAND_POOL_CREATE_INFO,
			flags            = {.RESET_COMMAND_BUFFER},
			queueFamilyIndex = VULKAN_STATE.queues.graphics,
		}
		result := vk.CreateCommandPool(VULKAN_STATE.device, &pool_info, nil, &frame.command_pool)
		if result != .SUCCESS {
			log.errorf("[BF_GPU/Vulkan] vkCreateCommandPool failed for frame %d: %v", i, result)
			return false
		}
		allocate_info := vk.CommandBufferAllocateInfo {
			sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
			commandPool        = frame.command_pool,
			level              = .PRIMARY,
			commandBufferCount = 1,
		}
		result = vk.AllocateCommandBuffers(
			VULKAN_STATE.device,
			&allocate_info,
			&frame.command_buffer,
		)
		if result != .SUCCESS {
			log.errorf(
				"[BF_GPU/Vulkan] vkAllocateCommandBuffers failed for frame %d: %v",
				i,
				result,
			)
			return false
		}
	}
	return true
}
// Fence starts signaled bc otherwise the first frame would wait forever for a fence that was never submitted.
vulkan_create_sync_objects :: proc() -> bool {
	semaphore_info := vk.SemaphoreCreateInfo {
		sType = .SEMAPHORE_CREATE_INFO,
	}

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		frame := &VULKAN_STATE.frames[i]
		result := vk.CreateSemaphore(
			VULKAN_STATE.device,
			&semaphore_info,
			nil,
			&frame.image_available,
		)
		if result != .SUCCESS {
			log.errorf("[BF_GPU/Vulkan] image_available semaphore creation failed: %v", result)
			return false
		}
		result = vk.CreateSemaphore(
			VULKAN_STATE.device,
			&semaphore_info,
			nil,
			&frame.render_finished,
		)
		if result != .SUCCESS {
			log.errorf("[BF_GPU/Vulkan] render_finished semaphore creation failed: %v", result)
			return false
		}
	}
	return vulkan_create_timeline_semaphores()
}

vulkan_create_allocator :: proc() -> bool {
	functions := vma.create_vulkan_functions()

	create_info := vma.AllocatorCreateInfo {
		flags            = {.BUFFER_DEVICE_ADDRESS},
		instance         = VULKAN_STATE.instance,
		physicalDevice   = VULKAN_STATE.physical_device,
		device           = VULKAN_STATE.device,
		pVulkanFunctions = &functions,
		vulkanApiVersion = BF_GPU_VULKAN_API_VERSION,
	}
	result := vma.CreateAllocator(create_info, &VULKAN_STATE.allocator)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] VMA allocator creation failed: %v", result)
		return false
	}
	// Bring up the persistent upload ring right after VMA. The ring is
	// the fast path for the scene + asset pool uploads; it lives until
	// vulkan_shutdown tears down the allocator.
	if !vulkan_upload_ring_init(VULKAN_UPLOAD_RING_DEFAULT_SLOT_BYTES, MAX_FRAMES_IN_FLIGHT) {
		log.error("[BF_GPU/Vulkan] upload ring creation failed")
		return false
	}
	return true
}

vulkan_create_timeline_semaphore :: proc(initial_value: u64, out: ^vk.Semaphore) -> bool {
	type_info := vk.SemaphoreTypeCreateInfo {
		sType         = .SEMAPHORE_TYPE_CREATE_INFO,
		semaphoreType = .TIMELINE,
		initialValue  = initial_value,
	}
	create_info := vk.SemaphoreCreateInfo {
		sType = .SEMAPHORE_CREATE_INFO,
		pNext = &type_info,
	}
	result := vk.CreateSemaphore(VULKAN_STATE.device, &create_info, nil, out)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] Timeline semaphore creation failed: %v", result)
		return false
	}
	return true
}
vulkan_create_timeline_semaphores :: proc() -> bool {
	if !vulkan_create_timeline_semaphore(0, &VULKAN_STATE.graphics_timeline) do return false
	if !vulkan_create_timeline_semaphore(0, &VULKAN_STATE.compute_timeline) do return false
	if !vulkan_create_timeline_semaphore(0, &VULKAN_STATE.transfer_timeline) do return false
	return true
}

//////////////////////////////////////////////////////////////////////////////////////
//* GPU_BACKEND REGISTRATION ------------------------------------------------------
//
// The Vulkan backend implements the GPU_Backend vtable declared in
// Renderer.odin. The renderer module calls register_gpu_backend() via
// vulkan_register_backend() after vulkan_init succeeds; until then the
// renderer runs in a no-backend mode that skips frame submission.
//
// Per-proc responsibilities:
//   - compile_shader / create_pipeline / destroy_shader / destroy_pipeline:
//       stubs that return success/nil. Real GLSL->SPIR-V + VkPipeline
//       wiring lives in Vk_Pipeline.odin / Vk_Shader.odin; those slots
//       exist here so the renderer's GPU_Backend calls stay stable.
//   - create_buffer / destroy_buffer / get_device_address / upload_buffer:
//       VMA-backed buffer lifecycle owned by Vulkan.odin. A small map
//       keeps the handle -> Vor_Buf* mapping alive across frames.
//   - record_frame:
//       delegates to vulkan_frame(), which performs acquire /
//       submit / present and drives the swapchain.

@(private)
Vulkan_Buffer_Map :: map[Gpu_Buffer_Handle]^Vulkan_Buffer

@(private)
VULKAN_BUFFER_MAP: Vulkan_Buffer_Map

@(private)
VULKAN_BUFFER_NEXT_ID: u64 = 1 // 0 is reserved as the invalid handle

@(private)
VULKAN_BACKEND: GPU_Backend = {
	name               = "Vulkan",
	compile_shader     = vulkan_backend_compile_shader,
	create_pipeline    = vulkan_backend_create_pipeline,
	create_buffer      = vulkan_backend_create_buffer,
	get_device_address = vulkan_backend_get_device_address,
	upload_buffer      = vulkan_backend_upload_buffer,
	record_frame       = vulkan_backend_record_frame,
	destroy_buffer     = vulkan_backend_destroy_buffer,
	destroy_pipeline   = vulkan_backend_destroy_pipeline,
	destroy_shader     = vulkan_backend_destroy_shader,
	create_image         = vulkan_backend_create_image,
	create_image_view    = vulkan_backend_create_image_view,
	create_sampler       = vulkan_backend_create_sampler,
	destroy_image        = vulkan_backend_destroy_image,
	destroy_image_view   = vulkan_backend_destroy_image_view,
	destroy_sampler      = vulkan_backend_destroy_sampler,
	create_asset_buffer  = vulkan_backend_create_asset_buffer,
	flush_deferred_destructions = vulkan_backend_flush_deferred_destructions,
}

vulkan_register_backend :: proc() {
	register_gpu_backend(&VULKAN_BACKEND)
}

vulkan_unregister_backend :: proc() {
	unregister_gpu_backend()
	// Detach the buffer map so the next backend starts clean.
	clear(&VULKAN_BUFFER_MAP)
	VULKAN_BUFFER_NEXT_ID = 1
}

// ---------------------------------------------------------------------------
// Backend proc implementations.
// ---------------------------------------------------------------------------

// vulkan_backend_compile_shader / vulkan_backend_create_pipeline /
// vulkan_backend_destroy_shader / vulkan_backend_destroy_pipeline
// are the GPU_Backend hook forwarders. The actual implementations live
// in Vk_Shader.odin and Vk_Pipeline.odin; this file keeps the vtable
// binding in one place so the public GPU_Backend surface has a single
// owner.

vulkan_backend_compile_shader :: proc(path, stage: cstring) -> rawptr {
	// Implementation: Vk_Shader.odin::vulkan_backend_compile_shader.
	return vulkan_backend_compile_shader_impl(path, stage)
}

vulkan_backend_create_pipeline :: proc(shader: rawptr, descriptor: rawptr) -> rawptr {
	// Implementation: Vk_Pipeline.odin::vulkan_backend_create_pipeline.
	return vulkan_backend_create_pipeline_impl(shader, descriptor)
}

vulkan_backend_destroy_shader :: proc(s: rawptr) {
	// Implementation: Vk_Shader.odin::vulkan_backend_destroy_shader.
	vulkan_backend_destroy_shader_impl(s)
}

vulkan_backend_destroy_pipeline :: proc(p: rawptr) {
	// Implementation: Vk_Pipeline.odin::vulkan_backend_destroy_pipeline.
	vulkan_backend_destroy_pipeline_impl(p)
}

vulkan_backend_create_buffer :: proc(kind: Gpu_Buffer_Kind, size: u64, stride: u32) -> Gpu_Buffer_Handle {
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] create_buffer called before Vulkan device creation")
		return Gpu_Buffer_Handle(0)
	}
	if msg := vulkan_validate_buffer_size(kind, size, stride); msg != "" {
		log.errorf("[BF_GPU/Vulkan] create_buffer rejected: %s", msg)
		return Gpu_Buffer_Handle(0)
	}
	if msg := vulkan_validate_buffer_alignment(size, stride); msg != "" {
		log.errorf("[BF_GPU/Vulkan] create_buffer rejected: %s", msg)
		return Gpu_Buffer_Handle(0)
	}
	// All renderer-managed persistent buffers are GPU-only storage
	// buffers reachable through buffer_device_address. Compute /
	// indirect / vertex usages get added by the frame layer when the
	// pass is bound; the renderer never asks for a non-storage flag.
	usage := vk.BufferUsageFlags{.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS}
	vbuf, ok := vulkan_create_buffer(vk.DeviceSize(size), usage, .GPU_ONLY)
	if !ok {
		return Gpu_Buffer_Handle(0)
	}

	handle := Gpu_Buffer_Handle(VULKAN_BUFFER_NEXT_ID)
	VULKAN_BUFFER_NEXT_ID += 1
	// The map owns the heap allocation for the buffer struct.
	entry := new(Vulkan_Buffer)
	entry^ = vbuf
	VULKAN_BUFFER_MAP[handle] = entry

	_ = kind
	_ = stride
	return handle
}

vulkan_backend_get_device_address :: proc(handle: Gpu_Buffer_Handle) -> u64 {
	entry, ok := VULKAN_BUFFER_MAP[handle]
	if !ok || entry == nil do return 0
	return u64(entry.device_address)
}

// gpu_buffer_usage_to_vk translates the host-side Gpu_Buffer_Usage
// bit-set into vk.BufferUsageFlags. Asset_Sync describes the
// intended use; this proc is the single source of truth for the
// translation so the asset pipeline never names vk.BufferUsageFlags
// directly.
@(private)
gpu_buffer_usage_to_vk :: proc(usage: Gpu_Buffer_Usage) -> vk.BufferUsageFlags {
	flags: vk.BufferUsageFlags
	if .Vertex_Buffer          in usage do flags += {.VERTEX_BUFFER}
	if .Index_Buffer           in usage do flags += {.INDEX_BUFFER}
	if .Storage_Buffer         in usage do flags += {.STORAGE_BUFFER}
	if .Uniform_Buffer         in usage do flags += {.UNIFORM_BUFFER}
	if .Shader_Device_Address  in usage do flags += {.SHADER_DEVICE_ADDRESS}
	if .Transfer_Src           in usage do flags += {.TRANSFER_SRC}
	if .Transfer_Dst           in usage do flags += {.TRANSFER_DST}
	return flags
}

// vulkan_backend_create_asset_buffer is the GPU_Backend implementation
// of create_asset_buffer. Asset_Sync passes the exact set of usage
// bits the cooked payload needs (vertex + transfer-dst, index +
// transfer-dst, storage + shader-device-address, ...). The buffer is
// allocated as GPU_ONLY memory: Asset_Sync uploads through the
// existing upload_buffer path, which performs a CPU-side staging
// copy when the backend supports it.
vulkan_backend_create_asset_buffer :: proc(
	usage: Gpu_Buffer_Usage,
	size: u64,
	stride: u32,
) -> Gpu_Buffer_Handle {
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] create_asset_buffer called before Vulkan device creation")
		return Gpu_Buffer_Handle(0)
	}
	vk_usage := gpu_buffer_usage_to_vk(usage)
	if vk_usage == {} {
		log.error("[BF_GPU/Vulkan] create_asset_buffer: empty usage set")
		return Gpu_Buffer_Handle(0)
	}
	// Asset uploads always need Transfer_Dst.
	if .TRANSFER_DST not_in vk_usage do vk_usage += {.TRANSFER_DST}
	if size == 0 {
		log.error("[BF_GPU/Vulkan] create_asset_buffer: zero size")
		return Gpu_Buffer_Handle(0)
	}
	vbuf, ok := vulkan_create_buffer(vk.DeviceSize(size), vk_usage, .GPU_ONLY)
	if !ok {
		return Gpu_Buffer_Handle(0)
	}

	handle := Gpu_Buffer_Handle(VULKAN_BUFFER_NEXT_ID)
	VULKAN_BUFFER_NEXT_ID += 1
	entry := new(Vulkan_Buffer)
	entry^ = vbuf
	VULKAN_BUFFER_MAP[handle] = entry
	diag_record_asset_buffer_created(MODULE_STATE_VALUE.frame_ctx.frame_idx)
	_ = stride
	return handle
}

vulkan_backend_upload_buffer :: proc(handle: Gpu_Buffer_Handle, data: rawptr, size: u64) -> bool {
	entry, ok := VULKAN_BUFFER_MAP[handle]
	if !ok || entry == nil || data == nil || size == 0 {
		return false
	}
	// Diagnostics: track host->device bytes + the upload call count.
	// The frame_index is the just-recorded renderer's frame counter
	// (the submit step will advance it; recording against the
	// pre-submit counter keeps "this frame uploaded X bytes" in
	// matching ring slots).
	diag_record_upload_bandwidth(MODULE_STATE_VALUE.frame_ctx.frame_idx, size)
	diag_record_upload_count(MODULE_STATE_VALUE.frame_ctx.frame_idx)
	// Staging upload via a transient host-visible buffer and a copy
	// command recorded on the current frame's graphics command buffer.
	// Fast path: slice into the persistent upload ring (allocated in
	// vulkan_create_allocator). Falls back to a per-upload transient
	// buffer only when the ring slot is exhausted.
	slot := u32(VULKAN_STATE.frame_index)
	src_buffer, src_offset, src_ptr, ring_ok := vulkan_upload_ring_alloc(slot, size, 4)
	if ring_ok {
		mem.copy(src_ptr, data, int(size))
	} else {
		trans_src: vk.Buffer
		trans_alloc: vma.Allocation
		trans_src, trans_alloc, ring_ok = vulkan_create_staging_buffer(data, size)
		if !ring_ok do return false
		defer vma.DestroyBuffer(VULKAN_STATE.allocator, trans_src, trans_alloc)
		src_buffer = trans_src
		src_offset = 0
	}

	frame := &VULKAN_STATE.frames[VULKAN_STATE.frame_index]
	if !vulkan_begin_command_buffer(frame.command_buffer) do return false
	defer {
		if !vulkan_end_command_buffer(frame.command_buffer) {
			log.error("[BF_GPU/Vulkan] upload_buffer: EndCommandBuffer failed")
		}
	}

	copy := vk.BufferCopy {srcOffset = src_offset, dstOffset = 0, size = vk.DeviceSize(size)}
	vk.CmdCopyBuffer(frame.command_buffer, src_buffer, entry.buffer, 1, &copy)

	// Host -> device synchronization barrier (writes by the copy must
	// be visible to the GPU shader-side reads).
	barrier := vk.BufferMemoryBarrier2 {
		sType               = .BUFFER_MEMORY_BARRIER_2,
		srcStageMask        = {.COPY},
		srcAccessMask       = {.TRANSFER_WRITE},
		dstStageMask        = {.COMPUTE_SHADER, .VERTEX_SHADER, .FRAGMENT_SHADER},
		dstAccessMask       = {.SHADER_READ},
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		buffer              = entry.buffer,
		offset              = 0,
		size                = vk.DeviceSize(size),
	}
	dep := vk.DependencyInfo {
		sType                    = .DEPENDENCY_INFO,
		bufferMemoryBarrierCount = 1,
		pBufferMemoryBarriers    = &barrier,
	}
	vk.CmdPipelineBarrier2(frame.command_buffer, &dep)
	return true
}

vulkan_backend_destroy_buffer :: proc(handle: Gpu_Buffer_Handle) {
	entry, ok := VULKAN_BUFFER_MAP[handle]
	if !ok || entry == nil do return
	vma.DestroyBuffer(VULKAN_STATE.allocator, entry.buffer, entry.allocation)
	free(entry)
	delete_key(&VULKAN_BUFFER_MAP, handle)
}

vulkan_backend_record_frame :: proc(gpu: ^GPU_Scene, ctx: ^Frame_Context_State) -> bool {
	if !VULKAN_STATE.initialized {
		log.warn("[BF_GPU/Vulkan] record_frame before init")
		return false
	}
	return vulkan_frame(gpu, ctx)
}

// ---------------------------------------------------------------------------
//* GPU_BACKEND: image / view / sampler creation + destruction.
//
// These wrap the lower-level vulkan_create_* helpers, allocate handle
// ids, and store the result in VULKAN_IMAGE_MAP / VULKAN_IMAGE_VIEW_MAP /
// VULKAN_SAMPLER_MAP. Destruction defers until the GPU work that
// references the resource has completed (vulkan_defer_*_*_destruction).
// ---------------------------------------------------------------------------

vulkan_backend_create_image :: proc(desc: Image_Description) -> Gpu_Image_Handle {
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] create_image called before Vulkan device creation")
		return GPU_IMAGE_INVALID
	}
	vimg, ok := vulkan_create_image(desc)
	if !ok do return GPU_IMAGE_INVALID

	handle := Gpu_Image_Handle(VULKAN_IMAGE_NEXT_ID)
	VULKAN_IMAGE_NEXT_ID += 1
	entry := new(Vulkan_Image)
	entry^ = vimg
	VULKAN_IMAGE_MAP[handle] = entry
	diag_record_image_created(MODULE_STATE_VALUE.frame_ctx.frame_idx)
	return handle
}

vulkan_backend_create_image_view :: proc(desc: Image_View_Description) -> Gpu_Image_View_Handle {
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] create_image_view called before Vulkan device creation")
		return GPU_IMAGE_VIEW_INVALID
	}
	view, ok := vulkan_create_image_view(desc.image, desc)
	if !ok do return GPU_IMAGE_VIEW_INVALID

	handle := Gpu_Image_View_Handle(VULKAN_IMAGE_VIEW_NEXT_ID)
	VULKAN_IMAGE_VIEW_NEXT_ID += 1
	VULKAN_IMAGE_VIEW_MAP[handle] = view
	return handle
}

vulkan_backend_create_sampler :: proc(desc: Sampler_Description) -> Gpu_Sampler_Handle {
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] create_sampler called before Vulkan device creation")
		return GPU_SAMPLER_INVALID
	}
	sampler, ok := vulkan_create_sampler(desc)
	if !ok do return GPU_SAMPLER_INVALID

	handle := Gpu_Sampler_Handle(VULKAN_SAMPLER_NEXT_ID)
	VULKAN_SAMPLER_NEXT_ID += 1
	VULKAN_SAMPLER_MAP[handle] = sampler
	return handle
}

vulkan_backend_destroy_image :: proc(handle: Gpu_Image_Handle) {
	entry, ok := VULKAN_IMAGE_MAP[handle]
	if !ok || entry == nil do return
	// Remove from the map immediately so a subsequent lookup returns
	// invalid; the actual VkImage / VmaAllocation release is deferred
	// so any in-flight GPU work that references the image can finish
	// first.
	delete_key(&VULKAN_IMAGE_MAP, handle)
	tag := VULKAN_STATE.graphics_timeline_value
	vulkan_defer_image_destruction(entry, tag)
	diag_record_image_destroyed(MODULE_STATE_VALUE.frame_ctx.frame_idx)
}

vulkan_backend_destroy_image_view :: proc(handle: Gpu_Image_View_Handle) {
	view, ok := VULKAN_IMAGE_VIEW_MAP[handle]
	if !ok || view == {} do return
	delete_key(&VULKAN_IMAGE_VIEW_MAP, handle)
	tag := VULKAN_STATE.graphics_timeline_value
	vulkan_defer_image_view_destruction(view, tag)
}

vulkan_backend_destroy_sampler :: proc(handle: Gpu_Sampler_Handle) {
	sampler, ok := VULKAN_SAMPLER_MAP[handle]
	if !ok || sampler == {} do return
	delete_key(&VULKAN_SAMPLER_MAP, handle)
	tag := VULKAN_STATE.graphics_timeline_value
	vulkan_defer_sampler_destruction(sampler, tag)
}

// vulkan_backend_flush_deferred_destructions is the GPU_Backend
// implementation of flush_deferred_destructions. Called from the render
// loop with the latest graphics_timeline_value; reaps every queued
// entry whose tag has been signaled.
vulkan_backend_flush_deferred_destructions :: proc(up_to: u64) {
	vulkan_collect_garbage(up_to)
}