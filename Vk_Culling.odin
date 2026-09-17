// BF_GPU/Vk_Culling.odin
//
// GPU-driven culling pipeline 
//
// The culling path turns Render_Scene into the indirect-draw inputs the
// shading pass consumes. Three compute passes run on the GPU:
//
//   StaticChunkCulling  ->  StaticModelCulling  ->  MeshCulling
//
// A reset pass (GeometryCullingCommandReset) runs first so every
// VkDrawIndexedIndirect / VkDrawMeshTasksIndirect slot starts the frame
// with instanceCount / groupCountX == 0; the culling passes then
// atomicAdd into the slots whose mesh / material bucket is visible.
//
// All four shaders read the Gpu_Frame_Global_Context SSBO through a
// buffer_reference passed via push constant; the dispatch uses only
// the per-pass descriptor set (set 2, binding 0 = HiZ depth pyramid).
// The reset + culling dispatches are issued every frame, but the
// dispatch group counts are gated on the per-frame chunk / model /
// mesh counts so a zero-visible frame skips work entirely.
//
// Pipeline state:
//   - 4 compute pipelines, one per shader (reset + 3 culling stages)
//   - 4 shader modules (compile_shader path through the shader cache)
//   - 1 placeholder 1x1 R32_SFLOAT depth-pyramid image + view + sampler
//     bound at set 2 binding 0. The placeholder exists so the culling
//     dispatches always have a valid sampler binding even when occlusion
//     culling is disabled; a real HiZ pyramid replaces it via
//     vulkan_descriptor_update_per_pass_depth_pyramid.
//
// Frame ordering inside vulkan_frame:
//
//     1. wait on graphics timeline (release BF_DAG gate)
//     2. acquire next image
//     3. begin command buffer
//     4. vulkan_upload_scene (host -> device buffer copies + barriers)
//     5. vulkan_record_culling_passes (this file)
//        - culling reset (writable indirect / count buffers)
//        - vkCmdPipelineBarrier2: COMPUTE -> COMPUTE write -> read
//        - static chunk culling
//        - barrier
//        - static model culling
//        - barrier
//        - mesh culling
//        - barrier: COMPUTE -> DRAW_INDIRECT + vertex / fragment
//     6. clear pass
//     7. end / submit / present
//
// All barriers use Vulkan 1.3 synchronization2.
//
// Visibility stays GPU-side: nothing in this file reads back the
// visible-instance buffers. The static chunk list, model visible list,
// and final indirect-command counts are consumed by the future draw
// pass without CPU round-trip.

package BF_GPU

import "core:log"
import "core:strings"
import vma "../../dependencies/odin-vma"
import vk "vendor:vulkan"

// ---------------------------------------------------------------------------
//* State.
//
// Each pipeline is built once at startup; shader handles are obtained
// from the shared shader cache (Vk_Shader.odin). Destroying the culling
// module drops both pipelines and the shader handles, then releases the
// HiZ placeholder image + view + sampler.
// ---------------------------------------------------------------------------

Culling_Pass :: enum u8 {
	Reset       = 0,
	StaticChunk = 1,
	StaticModel = 2,
	Mesh        = 3,
	COUNT       = 4,
}

Culling_Pipeline_Entry :: struct {
	pass:     Culling_Pass,
	pipeline: rawptr,
	shader:   rawptr,
	name:     string,
}

@(private)
VULKAN_CULLING_STATE: Vulkan_Culling_State

Vulkan_Culling_State :: struct {
	pipelines:   [Culling_Pass.COUNT]Culling_Pipeline_Entry,
	// Placeholder HiZ image: a 1x1 R32_SFLOAT image with a matching
	// image view and a CLAMP_TO_EDGE / LINEAR sampler. When the
	// renderer eventually produces a real depth pyramid the descriptor
	// at set 2 binding 0 is overwritten via
	// vulkan_descriptor_update_per_pass_depth_pyramid; this object
	// keeps the culling dispatches valid in the meantime.
	hiz_image:        vk.Image,
	hiz_image_view:   vk.ImageView,
	hiz_sampler:      vk.Sampler,
	hiz_allocation:   vma.Allocation,
	hiz_handle:       Gpu_Image_Handle,
	hiz_initialized:  bool,
	initialized:      bool,
}

// Shader source paths. Resolved through vulkan_resolve_shader_path
// against the BF_GPU shader root. Indexed by Culling_Pass.COUNT to
// give every pass a fixed slot; the .COUNT sentinel slot is left
// empty so the table size matches the pipelines array.
@(private)
CULLING_SHADER_PATHS := [Culling_Pass.COUNT]string {
	0 = "Passes/Culling/Geometry/GeometryCullingCommandReset.comp",
	1 = "Passes/Culling/Geometry/GeometryStaticChunkCulling.comp",
	2 = "Passes/Culling/Geometry/GeometryStaticModelCulling.comp",
	3 = "Passes/Culling/Geometry/GeometryMeshCulling.comp",
}

// CULLING_WORKGROUP_SIZE is the local_size_x every culling shader
// declares. Cached here so the dispatch math matches what the shader
// produced (the shaders are compiled with local_size_x = 32; using a
// different value silently leaves lanes idle).
CULLING_WORKGROUP_SIZE :: 32

// ---------------------------------------------------------------------------
//* Lifecycle.
// ---------------------------------------------------------------------------

// vulkan_culling_init compiles the four culling shaders, builds the
// matching compute pipelines against the per-pass descriptor set
// (set 2 only), and creates the placeholder HiZ image + view +
// sampler. Safe to call after vulkan_descriptor_init has brought the
// layouts up; safe to call again only after vulkan_culling_shutdown.
//
// Called from within vulkan_init, so the guard checks the device
// handle rather than VULKAN_STATE.initialized (which is only flipped
// at the very end of vulkan_init).
vulkan_culling_init :: proc() -> bool {
	if VULKAN_CULLING_STATE.initialized {
		log.warn("[BF_GPU/Vulkan] culling_init called twice; ignoring")
		return true
	}
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] culling_init before Vulkan device creation")
		return false
	}
	if !vulkan_descriptor_initialized() {
		log.error("[BF_GPU/Vulkan] culling_init before descriptor model init")
		return false
	}

	if !vulkan_culling_create_hiz_placeholder() {
		log.error("[BF_GPU/Vulkan] HiZ placeholder creation failed")
		vulkan_culling_shutdown()
		return false
	}

	if !vulkan_culling_build_pipelines() {
		log.error("[BF_GPU/Vulkan] culling pipeline creation failed")
		vulkan_culling_shutdown()
		return false
	}

	// Bind the placeholder HiZ into every frame-in-flight's per-pass
	// set so a culling dispatch never reads an unbound descriptor.
	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		vulkan_descriptor_update_per_pass_depth_pyramid(
			i,
			VULKAN_CULLING_STATE.hiz_image_view,
			.SHADER_READ_ONLY_OPTIMAL,
			VULKAN_CULLING_STATE.hiz_sampler,
		)
	}

	VULKAN_CULLING_STATE.initialized = true
	log.info("[BF_GPU/Vulkan] GPU culling pipeline initialized (4 compute shaders)")
	return true
}

// vulkan_culling_shutdown releases every culling pipeline and the HiZ
// placeholder. The shader modules drop through the shader cache's
// refcount (destroy_pipeline below only drops one refcount per
// pipeline; the cache is drained by vulkan_shader_cache_shutdown).
vulkan_culling_shutdown :: proc() {
	if !VULKAN_CULLING_STATE.initialized do return

	for i in 0 ..< int(Culling_Pass.COUNT) {
		entry := &VULKAN_CULLING_STATE.pipelines[i]
		if entry.pipeline != nil {
			vulkan_backend_destroy_pipeline_impl(entry.pipeline)
			entry.pipeline = nil
		}
		if entry.shader != nil {
			vulkan_backend_destroy_shader_impl(entry.shader)
			entry.shader = nil
		}
		entry.name = ""
	}

	vulkan_culling_destroy_hiz_placeholder()
	VULKAN_CULLING_STATE.initialized = false
}

// vulkan_culling_initialized exposes the state flag for tests and the
// frame recording path.
vulkan_culling_initialized :: proc() -> bool {
	return VULKAN_CULLING_STATE.initialized
}

// ---------------------------------------------------------------------------
//* Pipeline construction.
// ---------------------------------------------------------------------------

@(private)
vulkan_culling_build_pipelines :: proc() -> bool {
	layouts := vulkan_descriptor_pipeline_layouts()
	// The culling pipelines only consume set 2 (per-pass depth
	// pyramid). Using a slice of length 1 means the resulting
	// VkPipelineLayout only references the per-pass set; the
	// pipeline does not need a valid frame/persistent set bound.
	set_layouts := layouts[2:3]

	push_constant_size := u32(size_of(PC_ModelMeshCulling))

	for pass in Culling_Pass {
		if pass == .COUNT {break}
		path := CULLING_SHADER_PATHS[pass]
		// compile_shader_impl takes cstring; clone the path onto the
		// temp allocator since the implementation only reads it
		// during the call. Keeping a stable Odin string in the
		// pipelines[] name field avoids needing a heap cstring.
		cpath := strings.clone_to_cstring(path, context.temp_allocator)

		shader := vulkan_backend_compile_shader_impl(cpath, "compute")
		if shader == nil {
			log.errorf("[BF_GPU/Vulkan] culling: failed to compile shader %s", path)
			return false
		}

		descriptor := Compute_Pipeline_Description {
			name                  = path,
			compute_shader        = shader,
			push_constant_size    = push_constant_size,
			descriptor_set_layouts = set_layouts,
		}
		pipeline := vulkan_create_compute_pipeline(&descriptor)
		if pipeline == nil {
			log.errorf("[BF_GPU/Vulkan] culling: failed to build pipeline for %s", path)
			vulkan_backend_destroy_shader_impl(shader)
			return false
		}

		VULKAN_CULLING_STATE.pipelines[pass] = Culling_Pipeline_Entry {
			pass     = pass,
			pipeline = pipeline,
			shader   = shader,
			name     = path,
		}
	}

	return true
}

// ---------------------------------------------------------------------------
//* HiZ placeholder.
// ---------------------------------------------------------------------------

@(private)
vulkan_culling_create_hiz_placeholder :: proc() -> bool {
	// Use the backend (handle-returning) image creation so the
	// resulting image is registered in VULKAN_IMAGE_MAP. The view
	// creation then validates via the handle and the destruction
	// path uses the VMA-tracked allocation. The previous direct
	// vulkan_create_image() path did not register a handle and the
	// image-view validation rejected the call with "invalid image
	// handle", which surfaced as a hard init failure on the
	// placeholder.
	img_handle := vulkan_backend_create_image(image_description_default(
		.R32_Sfloat,
		{width = 1, height = 1},
		{.Sampled, .Transfer_Dst},
	))
	if img_handle == Gpu_Image_Handle(0) {
		log.error("[BF_GPU/Vulkan] culling: failed to create HiZ placeholder image")
		return false
	}
	img_entry, img_ok := VULKAN_IMAGE_MAP[img_handle]
	if !img_ok || img_entry == nil {
		log.error("[BF_GPU/Vulkan] culling: HiZ placeholder image not registered")
		return false
	}
	VULKAN_CULLING_STATE.hiz_image      = img_entry.image
	VULKAN_CULLING_STATE.hiz_allocation = img_entry.allocation
	VULKAN_CULLING_STATE.hiz_handle     = img_handle

	img_desc := Image_View_Description {
		image       = img_handle,
		kind        = .View_2D,
		format      = .R32_Sfloat,
		base_mip    = 0,
		mip_count   = 1,
		base_layer  = 0,
		layer_count = 1,
	}
	view, view_ok := vulkan_create_image_view_for_image(img_entry, img_desc)
	if !view_ok || view == {} {
		log.error("[BF_GPU/Vulkan] culling: failed to create HiZ placeholder image view")
		return false
	}
	VULKAN_CULLING_STATE.hiz_image_view = view

	sampler_desc := sampler_description_default()
	sampler, sampler_ok := vulkan_create_sampler(sampler_desc)
	if !sampler_ok || sampler == {} {
		log.error("[BF_GPU/Vulkan] culling: failed to create HiZ placeholder sampler")
		return false
	}
	VULKAN_CULLING_STATE.hiz_sampler = sampler

	// Transition the placeholder to SHADER_READ_ONLY_OPTIMAL so the
	// first-frame descriptor write does not observe an UNDEFINED
	// layout. Reuse the placeholder image's matching layout transition
	// here via a one-shot command buffer.
	if !vulkan_culling_init_hiz_layout() {
		log.warn("[BF_GPU/Vulkan] culling: HiZ placeholder layout transition failed")
	}

	VULKAN_CULLING_STATE.hiz_initialized = true
	return true
}

@(private)
vulkan_culling_init_hiz_layout :: proc() -> bool {
	if VULKAN_CULLING_STATE.hiz_image == {} do return false

	cmd := vulkan_culling_acquire_one_shot_command_buffer()
	if cmd == {} do return false
	defer vulkan_culling_release_one_shot_command_buffer(cmd)

	barrier := vk.ImageMemoryBarrier2 {
		sType = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask = {.TOP_OF_PIPE},
		srcAccessMask = {},
		dstStageMask = {.COMPUTE_SHADER, .FRAGMENT_SHADER},
		dstAccessMask = {.SHADER_READ},
		oldLayout = .UNDEFINED,
		newLayout = .SHADER_READ_ONLY_OPTIMAL,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = VULKAN_CULLING_STATE.hiz_image,
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}
	dep := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}
	vk.CmdPipelineBarrier2(cmd, &dep)
	return true
}

@(private)
vulkan_culling_destroy_hiz_placeholder :: proc() {
	state := &VULKAN_CULLING_STATE
	if state.hiz_sampler != {} && VULKAN_STATE.device != nil {
		vk.DestroySampler(VULKAN_STATE.device, state.hiz_sampler, nil)
	}
	if state.hiz_image_view != {} && VULKAN_STATE.device != nil {
		vk.DestroyImageView(VULKAN_STATE.device, state.hiz_image_view, nil)
	}
	// The image is owned by the backend (registered through the
	// handle map); destroy via the backend path so the VMA
	// allocation and the map entry are both released.
	if state.hiz_handle != Gpu_Image_Handle(0) {
		vulkan_backend_destroy_image(state.hiz_handle)
	}
	state.hiz_image = {}
	state.hiz_image_view = {}
	state.hiz_sampler = {}
	state.hiz_allocation = nil
	state.hiz_handle = Gpu_Image_Handle(0)
	state.hiz_initialized = false
}

// ---------------------------------------------------------------------------
//* Dispatch.
// ---------------------------------------------------------------------------

// vulkan_record_culling_passes issues the reset + 3 culling compute
// dispatches into the current frame's command buffer. Called from
// vulkan_frame after vulkan_upload_scene and before the clear pass.
//
// Visibility stays GPU-side: this proc never reads back the visible
// chunk / model / instance buffers. The shader side atomicAdds into
// the indirect-command slots; the draw pass consumes those slots.
//
// Frame ordering (Vulkan 1.3 synchronization2):
//
//   1. reset dispatch (writable: indirect commands, count buffers)
//   2. barrier: COMPUTE write -> COMPUTE shader read
//   3. static chunk dispatch (writable: visible-chunk + dispatch
//      count buffer)
//   4. barrier: COMPUTE write -> COMPUTE shader read
//   5. static model dispatch (writable: visible-model + dispatch
//      count buffer; read: visible-chunk)
//   6. barrier: COMPUTE write -> COMPUTE shader read
//   7. mesh dispatch (writable: instance index, indirect commands;
//      read: visible-model)
//
// Group counts: each pass dispatches ceil(count / WORKGROUP_SIZE)
// groups. Zero counts early-out without recording the dispatch so
// the per-frame command buffer stays cheap on idle frames.
vulkan_record_culling_passes :: proc(
	cmd_buffer: vk.CommandBuffer,
	frame: ^Frame_Context_State,
	gpu: ^GPU_Scene,
) -> bool {
	if cmd_buffer == nil do return false
	if !VULKAN_CULLING_STATE.initialized do return true // no-op, but not an error
	if frame == nil || gpu == nil do return false

	pc := PC_ModelMeshCulling {
		frame_global_context_buffer_addr = frame.buffers[Gpu_Buffer_Kind.Frame_Global_Context].device_addr,
		lod_count                        = lod_select_threshold_count(gpu.settings.lod_count),
		lod_bias                         = gpu.settings.lod_bias,
	}
	pc_bytes := transmute([^]u8)&pc
	pc_size := u32(size_of(PC_ModelMeshCulling))

	sets := vulkan_descriptor_sets_get()
	per_pass_set := sets.per_pass[VULKAN_STATE.frame_index]
	// Odin requires a local to take the address of; per_pass_set is
	// already a value (vk.DescriptorSet), so we make a local copy to
	// satisfy vk.CmdBindDescriptorSets' pointer requirement.
	per_pass_set_local := per_pass_set

	// 1. Reset indirect commands. Even when the frame has zero
	//    visible objects the reset must run so a stale count from the
	//    previous frame cannot leak through; the per-frame cost is
	//    bounded by global_indirect_command_count <= 1024 in v1.
	if !vulkan_culling_dispatch_reset(cmd_buffer, pc_bytes, pc_size, &per_pass_set_local) {
		return false
	}
	vulkan_culling_barrier_compute_to_compute(cmd_buffer)

	// 2. Static chunk culling. Skip when no chunks are streamed.
	chunk_count := gpu.frame_static_chunk_count
	if chunk_count > 0 {
		if !vulkan_culling_dispatch_static_chunk(cmd_buffer, chunk_count, pc_bytes, pc_size, &per_pass_set_local) {
			return false
		}
		vulkan_culling_barrier_compute_to_compute(cmd_buffer)
	}

	// 3. Static model culling. Runs after the chunk dispatch because
	//    it consumes the visible-chunk list; group count is read from
	//    the GPU-resident chunk-count slot written by pass #2.
	model_count := gpu.frame_model_count
	if chunk_count > 0 && model_count > 0 {
		if !vulkan_culling_dispatch_static_model(cmd_buffer, pc_bytes, pc_size, &per_pass_set_local) {
			return false
		}
		vulkan_culling_barrier_compute_to_compute(cmd_buffer)
	}

	// 4. Mesh culling. Reads the visible-model list written by pass
	//    #3. Group count is read from the GPU-resident model-count
	//    slot.
	if model_count > 0 {
		if !vulkan_culling_dispatch_mesh(cmd_buffer, pc_bytes, pc_size, &per_pass_set_local) {
			return false
		}
		vulkan_culling_barrier_compute_to_compute(cmd_buffer)
	}

	return true
}

// ---------------------------------------------------------------------------
//* Per-pass dispatch helpers.
// ---------------------------------------------------------------------------

@(private)
vulkan_culling_dispatch_reset :: proc(
	cmd_buffer: vk.CommandBuffer,
	pc_bytes: [^]u8,
	pc_size: u32,
	per_pass_set: ^vk.DescriptorSet,
) -> bool {
	entry := &VULKAN_CULLING_STATE.pipelines[Culling_Pass.Reset]
	if entry.pipeline == nil do return true

	pipeline, layout := vulkan_pipeline_vk(entry.pipeline)

	vk.CmdBindPipeline(cmd_buffer, .COMPUTE, pipeline)
	if per_pass_set^ != {} {
		vk.CmdBindDescriptorSets(
			cmd_buffer,
			.COMPUTE,
			layout,
			2,
			1,
			per_pass_set,
			0,
			nil,
		)
	}
	vk.CmdPushConstants(cmd_buffer, layout, {.COMPUTE}, 0, pc_size, pc_bytes)

	// ceil(global_indirect_command_count / 256). The shader bounds
	// itself against the in-frame scalar; the reset is ALWAYS issued
	// when the pipeline exists so stale counts from a previous frame
	// cannot leak through when the current frame has no visible
	// objects (a zero indirect_count must still clear the buffer to
	// instanceCount == 0 before any draw pass tries to consume it).
	indirect_count := u32(MODULE_STATE_VALUE.gpu.frame_indirect_cmd_count)
	if indirect_count == 0 { indirect_count = 1 }
	groups := (indirect_count + 255) / 256
	vk.CmdDispatch(cmd_buffer, groups, 1, 1)
	return true
}

@(private)
vulkan_culling_dispatch_static_chunk :: proc(
	cmd_buffer: vk.CommandBuffer,
	chunk_count: u32,
	pc_bytes: [^]u8,
	pc_size: u32,
	per_pass_set: ^vk.DescriptorSet,
) -> bool {
	entry := &VULKAN_CULLING_STATE.pipelines[Culling_Pass.StaticChunk]
	if entry.pipeline == nil do return true

	pipeline, layout := vulkan_pipeline_vk(entry.pipeline)

	vk.CmdBindPipeline(cmd_buffer, .COMPUTE, pipeline)
	if per_pass_set^ != {} {
		vk.CmdBindDescriptorSets(
			cmd_buffer,
			.COMPUTE,
			layout,
			2,
			1,
			per_pass_set,
			0,
			nil,
		)
	}
	vk.CmdPushConstants(cmd_buffer, layout, {.COMPUTE}, 0, pc_size, pc_bytes)

	groups := (chunk_count + CULLING_WORKGROUP_SIZE - 1) / CULLING_WORKGROUP_SIZE
	vk.CmdDispatch(cmd_buffer, groups, 1, 1)
	return true
}

@(private)
vulkan_culling_dispatch_static_model :: proc(
	cmd_buffer: vk.CommandBuffer,
	pc_bytes: [^]u8,
	pc_size: u32,
	per_pass_set: ^vk.DescriptorSet,
) -> bool {
	entry := &VULKAN_CULLING_STATE.pipelines[Culling_Pass.StaticModel]
	if entry.pipeline == nil do return true

	pipeline, layout := vulkan_pipeline_vk(entry.pipeline)

	vk.CmdBindPipeline(cmd_buffer, .COMPUTE, pipeline)
	if per_pass_set^ != {} {
		vk.CmdBindDescriptorSets(
			cmd_buffer,
			.COMPUTE,
			layout,
			2,
			1,
			per_pass_set,
			0,
			nil,
		)
	}
	vk.CmdPushConstants(cmd_buffer, layout, {.COMPUTE}, 0, pc_size, pc_bytes)

	// The static-model shader reads its own dispatch count from
	// static_chunk_count (the slot the chunk-culling pass wrote).
	// We use the per-frame count as a conservative upper bound so
	// the dispatch always covers the available work; the shader
	// rejects workgroups past the actual count via the GPU-side
	// atomic counter.
	chunk_count := u32(MODULE_STATE_VALUE.gpu.frame_static_chunk_count)
	if chunk_count == 0 do return true
	groups := (chunk_count + CULLING_WORKGROUP_SIZE - 1) / CULLING_WORKGROUP_SIZE
	vk.CmdDispatch(cmd_buffer, groups, 1, 1)
	return true
}

@(private)
vulkan_culling_dispatch_mesh :: proc(
	cmd_buffer: vk.CommandBuffer,
	pc_bytes: [^]u8,
	pc_size: u32,
	per_pass_set: ^vk.DescriptorSet,
) -> bool {
	entry := &VULKAN_CULLING_STATE.pipelines[Culling_Pass.Mesh]
	if entry.pipeline == nil do return true

	pipeline, layout := vulkan_pipeline_vk(entry.pipeline)

	vk.CmdBindPipeline(cmd_buffer, .COMPUTE, pipeline)
	if per_pass_set^ != {} {
		vk.CmdBindDescriptorSets(
			cmd_buffer,
			.COMPUTE,
			layout,
			2,
			1,
			per_pass_set,
			0,
			nil,
		)
	}
	vk.CmdPushConstants(cmd_buffer, layout, {.COMPUTE}, 0, pc_size, pc_bytes)

	// The mesh culling shader reads its dispatch count from
	// model_count (the slot the static-model pass wrote). Same
	// upper-bound strategy as the static-model dispatch.
	model_count := u32(MODULE_STATE_VALUE.gpu.frame_model_count)
	if model_count == 0 do return true
	groups := (model_count + CULLING_WORKGROUP_SIZE - 1) / CULLING_WORKGROUP_SIZE
	vk.CmdDispatch(cmd_buffer, groups, 1, 1)
	return true
}

// ---------------------------------------------------------------------------
//* Synchronization.
// ---------------------------------------------------------------------------

// vulkan_culling_barrier_compute_to_compute inserts the
// synchronization2 barrier the culling chain relies on. Every dispatch
// in the chain writes indirect / count buffers the next dispatch
// reads; the barriers keep those reads coherent.
//
// We use a coarse COMPUTE -> COMPUTE write -> read barrier instead of
// naming every individual buffer because the chain is short (4
// dispatches) and the driver tracks the finer-grained hazards anyway.
// The draw pass (#8) will issue its own dedicated barrier against
// the indirect-command / instance-index buffers.
vulkan_culling_barrier_compute_to_compute :: proc(cmd_buffer: vk.CommandBuffer) {
	barrier := vk.MemoryBarrier2 {
		sType         = .MEMORY_BARRIER_2,
		srcStageMask  = {.COMPUTE_SHADER},
		srcAccessMask = {.SHADER_WRITE},
		dstStageMask  = {.COMPUTE_SHADER},
		dstAccessMask = {.SHADER_READ, .SHADER_WRITE},
	}
	dep := vk.DependencyInfo {
		sType               = .DEPENDENCY_INFO,
		memoryBarrierCount  = 1,
		pMemoryBarriers     = &barrier,
	}
	vk.CmdPipelineBarrier2(cmd_buffer, &dep)
}

// vulkan_culling_to_graphics_barrier inserts the
// compute-write -> indirect-draw-read barrier the shading pass
// (#8) relies on. Issued once after the culling chain finishes so the
// visible-chunk / visible-model / instance-index / indirect-command
// writes are visible to the vertex / mesh / fragment stages.
vulkan_culling_to_graphics_barrier :: proc(cmd_buffer: vk.CommandBuffer) {
	barrier := vk.MemoryBarrier2 {
		sType         = .MEMORY_BARRIER_2,
		srcStageMask  = {.COMPUTE_SHADER},
		srcAccessMask = {.SHADER_WRITE},
		dstStageMask  = {.DRAW_INDIRECT, .VERTEX_SHADER, .FRAGMENT_SHADER},
		dstAccessMask = {.SHADER_READ, .INDIRECT_COMMAND_READ},
	}
	dep := vk.DependencyInfo {
		sType              = .DEPENDENCY_INFO,
		memoryBarrierCount = 1,
		pMemoryBarriers    = &barrier,
	}
	vk.CmdPipelineBarrier2(cmd_buffer, &dep)
}

// ---------------------------------------------------------------------------
//* One-shot command buffer for HiZ layout transition.
//
// The HiZ placeholder image needs a single layout transition before
// the descriptor set is first bound. We reuse the existing per-frame
// graphics command buffer rather than allocating a dedicated pool:
// the transition happens once at init before the swapchain loop
// runs, so contention with the per-frame work is impossible.
// ---------------------------------------------------------------------------

@(private)
vulkan_culling_acquire_one_shot_command_buffer :: proc() -> vk.CommandBuffer {
	frame := &VULKAN_STATE.frames[VULKAN_STATE.frame_index]
	if frame.command_buffer == {} || VULKAN_STATE.device == nil do return {}

	if !vulkan_begin_command_buffer(frame.command_buffer) do return {}
	return frame.command_buffer
}

@(private)
vulkan_culling_release_one_shot_command_buffer :: proc(cmd: vk.CommandBuffer) {
	if cmd == {} do return
	// We deliberately do NOT end / submit here. The owning frame
	// loop will end + submit as part of its normal recording; the
	// transition recorded against this buffer just rides along.
}
