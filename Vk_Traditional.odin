// BF_GPU/Vk_Traditional.odin
//
// Traditional indirect rendering pass.
//
// Owns:
//   - One VkPipeline built from Shaders/Passes/Shading/Traditional/
//     {Traditional.vert, Traditional.frag}. Vertex stage reads the
//     GPU-side pool of (transform, model, mesh, instance-index,
//     indirect-draw) data through buffer_device_address; fragment
//     stage emits a visibility-friendly RGBA8 colour to the swapchain.
//   - A swapchain-sized depth image + image view created against
//     VMA, recreated in lock-step with the swapchain extent.
//
// Pipeline wiring:
//   - Descriptor sets 0 (frame UBO), 1 (persistent/bindless),
//     and 2 (per-pass HiZ placeholder). The descriptor layout
//     pipeline slice exposed by vulkan_descriptor_pipeline_layouts()
//     is consumed verbatim so the pipeline layout stays in sync
//     with the bindless model.
//   - Dynamic rendering info declares the swapchain colour format
//     + D32_SFLOAT depth format. The pipeline does NOT bake
//     blend/depth state; everything is set via CmdSet* / push
//     constants at record time.
//   - Push constants = PC_Traditional_Meshlet_Pass (frame global
//     context address + base descriptor offset + material render
//     type + cone-culling toggle). One push per render call.
//
// Frame integration:
//   - vulkan_frame() calls vulkan_record_traditional_passes after the
//     culling chain finishes and before the clear pass. The pass
//     transitions depth+swapchain, opens the dynamic rendering scope,
//     binds the pipeline + descriptors + push constants, sets
//     viewport + scissor, and issues vkCmdDrawIndexedIndirect per
//     traditional bucket. The compute-to-indirect barrier emitted by
//     vulkan_culling_to_graphics_barrier is the only synchronization
//     needed: culling writes the indirect commands' instanceCount
//     fields, the indirect draw reads them.
//
// Resource lifecycle:
//   - Init runs after the swapchain exists (so the depth extent is
//     known) and after the descriptor model is up (so the pipeline
//     layout can borrow set 0/1/2 layouts). vulkan_init wires the
//     call between descriptor_init and culling_init.
//   - Swapchain recreation funnels through
//     vulkan_recreate_dependent_resources, which destroys the
//     depth image + view and rebuilds them at the new extent.
//   - Shutdown releases both; the renderer_shutdown path tears the
//     whole module down through vulkan_shutdown which already calls
//     DeviceWaitIdle.

package BF_GPU

import vma "../../dependencies/odin-vma"
import "core:log"
import "core:strings"
import vk "vendor:vulkan"

// ---------------------------------------------------------------------------
// State.
// ---------------------------------------------------------------------------

Vulkan_Traditional_State :: struct {
	// Compiled shaders (rawptr handles into VULKAN_SHADER_MAP).
	vertex_shader: rawptr,
	fragment_shader: rawptr,

	// Pipeline object + layout (rawptr handle into VULKAN_PIPELINE_MAP).
	pipeline: rawptr,

	// Depth attachment. Lifetime is bound to the swapchain extent;
	// recreated through vulkan_recreate_dependent_resources.
	depth_image:      vk.Image,
	depth_image_view: vk.ImageView,
	depth_allocation: vma.Allocation,
	depth_format:     vk.Format,
	depth_handle:     Gpu_Image_Handle,

	initialized: bool,
}

@(private)
VULKAN_TRADITIONAL_STATE: Vulkan_Traditional_State

// TR_BUCKET_TRADITIONAL_FIRST / _LAST mark the inclusive range of
// bucket indices the traditional pass dispatches. With the default
// GEOMETRY_BUCKET_COUNT of 8 and PIPELINE_TRADITIONAL == 0, the four
// traditional buckets occupy indices 0..3; the four meshlet buckets
// (4..7) belong to BF_GPU_Mesh.
TR_BUCKET_TRADITIONAL_FIRST :: 0
TR_BUCKET_TRADITIONAL_LAST  :: 3

// TR_PC_BYTE_SIZE is the byte width of the TraditionalMeshletPassPC
// push-constant block as the shader sees it. The GLSL struct
// (uint64_t + 3 uint = 20 bytes) does not match size_of on the host
// side: Odin pads the trailing three u32s to the u64 alignment, so
// the struct reports size_of = 24. We pass 20 to vkCmdPushConstants
// so we only write the bytes the shader actually consumes.
TR_PC_BYTE_SIZE :: u32(20)

// ---------------------------------------------------------------------------
// Lifecycle.
// ---------------------------------------------------------------------------

// vulkan_traditional_init compiles the traditional shaders, builds the
// graphics pipeline against the bindless descriptor layouts, and
// creates the depth attachment sized to the current swapchain extent.
// Returns false on any failure; the renderer falls back to a clear-only
// pass in that case.
//
// Called from within vulkan_init, so the guard checks the device
// handle rather than VULKAN_STATE.initialized (which is only flipped
// at the very end of vulkan_init).
vulkan_traditional_init :: proc() -> bool {
	if VULKAN_TRADITIONAL_STATE.initialized {
		log.warn("[BF_GPU/Vulkan] traditional_init called twice; ignoring")
		return true
	}
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] traditional_init before Vulkan device creation")
		return false
	}
	if !vulkan_descriptor_initialized() {
		log.error("[BF_GPU/Vulkan] traditional_init before descriptor model init")
		return false
	}
	if VULKAN_STATE.swapchain.handle == {} {
		log.error("[BF_GPU/Vulkan] traditional_init before swapchain exists")
		return false
	}

	if !vulkan_traditional_create_depth_attachment() {
		log.error("[BF_GPU/Vulkan] traditional: depth attachment creation failed")
		vulkan_traditional_shutdown()
		return false
	}

	if !vulkan_traditional_build_pipeline() {
		log.error("[BF_GPU/Vulkan] traditional: pipeline build failed")
		vulkan_traditional_shutdown()
		return false
	}

	VULKAN_TRADITIONAL_STATE.initialized = true
	log.info("[BF_GPU/Vulkan] traditional indirect renderer initialized")
	return true
}

vulkan_traditional_shutdown :: proc() {
	if !VULKAN_TRADITIONAL_STATE.initialized do return

	if VULKAN_TRADITIONAL_STATE.pipeline != nil {
		vulkan_backend_destroy_pipeline_impl(VULKAN_TRADITIONAL_STATE.pipeline)
	}
	if VULKAN_TRADITIONAL_STATE.vertex_shader != nil {
		vulkan_backend_destroy_shader_impl(VULKAN_TRADITIONAL_STATE.vertex_shader)
	}
	if VULKAN_TRADITIONAL_STATE.fragment_shader != nil {
		vulkan_backend_destroy_shader_impl(VULKAN_TRADITIONAL_STATE.fragment_shader)
	}
	vulkan_traditional_destroy_depth_attachment()

	VULKAN_TRADITIONAL_STATE = {}
}

// vulkan_traditional_initialized exposes the state flag for tests and
// the frame recording path.
vulkan_traditional_initialized :: proc() -> bool {
	return VULKAN_TRADITIONAL_STATE.initialized
}

// vulkan_traditional_depth_view_for_frame returns the depth image view
// for the current frame slot. The image view is shared across frames
// (one depth target total), so callers do not need a per-frame index.
// Returns a zero view when the depth attachment is not initialized.
vulkan_traditional_depth_view_for_frame :: proc() -> vk.ImageView {
	if !VULKAN_TRADITIONAL_STATE.initialized do return {}
	return VULKAN_TRADITIONAL_STATE.depth_image_view
}

// ---------------------------------------------------------------------------
// Depth attachment.
// ---------------------------------------------------------------------------

// vulkan_traditional_create_depth_attachment creates the D32_SFLOAT
// depth target sized to the current swapchain extent. The image is
// created in OPTIMAL tiling with DEPTH_STENCIL_ATTACHMENT usage so the
// dynamic-rendering scope can bind it directly.
vulkan_traditional_create_depth_attachment :: proc() -> bool {
	extent := VULKAN_STATE.swapchain.extent
	if extent.width == 0 || extent.height == 0 {
		log.error("[BF_GPU/Vulkan] traditional: cannot create depth at zero extent")
		return false
	}

	desc := image_description_default(
		.D32_Sfloat,
		{width = extent.width, height = extent.height},
		{.Depth_Stencil_Attachment},
	)
	// Register the depth image in the backend handle map so the
	// image_view validation can resolve the handle. The previous
	// code path used vulkan_create_image (low-level, no handle) and
	// then passed `{}` for the image handle to vulkan_create_image_view_for_image,
	// which the validator rejected as GPU_IMAGE_INVALID. Using the
	// backend (handle-returning) path keeps the lifecycle symmetric
	// with the destruction path in vulkan_traditional_shutdown.
	img_handle := vulkan_backend_create_image(desc)
	if img_handle == Gpu_Image_Handle(0) {
		log.error("[BF_GPU/Vulkan] traditional: depth image creation failed")
		return false
	}
	img_entry, img_ok := VULKAN_IMAGE_MAP[img_handle]
	if !img_ok || img_entry == nil {
		log.error("[BF_GPU/Vulkan] traditional: depth image not registered")
		return false
	}
	vimg := img_entry

	view_desc := image_view_description_default(
		img_handle,
		.View_2D,
		.D32_Sfloat,
	)
	view_desc.base_mip = 0
	view_desc.mip_count = 1
	view_desc.base_layer = 0
	view_desc.layer_count = 1

	view, view_ok := vulkan_create_image_view_for_image(img_entry, view_desc)
	if !view_ok || view == {} {
		log.error("[BF_GPU/Vulkan] traditional: depth image view creation failed")
		vulkan_backend_destroy_image(img_handle)
		return false
	}

	VULKAN_TRADITIONAL_STATE.depth_image       = vimg.image
	VULKAN_TRADITIONAL_STATE.depth_image_view  = view
	VULKAN_TRADITIONAL_STATE.depth_allocation  = vimg.allocation
	VULKAN_TRADITIONAL_STATE.depth_format      = vimg.format
	VULKAN_TRADITIONAL_STATE.depth_handle      = img_handle

	log.infof(
		"[BF_GPU/Vulkan] traditional depth attachment: %dx%d, format=%v",
		extent.width, extent.height, vimg.format,
	)
	return true
}

// vulkan_traditional_destroy_depth_attachment releases the depth image
// + view. Safe to call from a partial-init failure path; safe to call
// multiple times.
vulkan_traditional_destroy_depth_attachment :: proc() {
	state := &VULKAN_TRADITIONAL_STATE
	if state.depth_image_view != {} && VULKAN_STATE.device != nil {
		vk.DestroyImageView(VULKAN_STATE.device, state.depth_image_view, nil)
	}
	// The image is owned by the backend (registered through the
	// handle map); destroy via the backend path so the VMA
	// allocation and the map entry are both released.
	if state.depth_handle != Gpu_Image_Handle(0) {
		vulkan_backend_destroy_image(state.depth_handle)
	}
	state.depth_image = {}
	state.depth_image_view = {}
	state.depth_allocation = nil
	state.depth_format = {}
	state.depth_handle = Gpu_Image_Handle(0)
}

// vulkan_traditional_recreate_depth_attachment is the swapchain
// recreation hook. Tears down the depth image + view and rebuilds at
// the new extent. Called from vulkan_recreate_dependent_resources
// after the new swapchain image views exist.
vulkan_traditional_recreate_depth_attachment :: proc() {
	if !VULKAN_TRADITIONAL_STATE.initialized do return
	vulkan_traditional_destroy_depth_attachment()
	if !vulkan_traditional_create_depth_attachment() {
		log.error("[BF_GPU/Vulkan] traditional: failed to recreate depth attachment")
		// Leave the state half-broken: pipeline still references a
		// depth view but the view handle is now zero. The frame
		// recording path checks the view handle and skips the
		// dynamic rendering scope when the depth attachment is gone.
	}
}

// ---------------------------------------------------------------------------
// Pipeline construction.
// ---------------------------------------------------------------------------

// vulkan_traditional_build_pipeline compiles the vert + frag shaders
// and builds the graphics pipeline. The pipeline layout borrows the
// descriptor set 0/1/2 layouts from vulkan_descriptor_pipeline_layouts
// and exposes one push constant range for PC_Traditional_Meshlet_Pass.
vulkan_traditional_build_pipeline :: proc() -> bool {
	vertex_path := strings.clone_to_cstring(
		"Passes/Shading/Common/Traditional.vert",
	)
	fragment_path := strings.clone_to_cstring(
		"Passes/Shading/Traditional/Traditional.frag",
		context.temp_allocator,
	)

	vert := vulkan_backend_compile_shader_impl(vertex_path, "vertex")
	if vert == nil {
		log.error("[BF_GPU/Vulkan] traditional: failed to compile Traditional.vert")
		return false
	}
	frag := vulkan_backend_compile_shader_impl(fragment_path, "fragment")
	if frag == nil {
		log.error("[BF_GPU/Vulkan] traditional: failed to compile Traditional.frag")
		vulkan_backend_destroy_shader_impl(vert)
		return false
	}

	layouts := vulkan_descriptor_pipeline_layouts()
	// The traditional graphics pipeline consumes all three descriptor
	// sets (frame + persistent + per-pass). The descriptor module owns
	// every layout; we only borrow references in the pipeline layout.
	set_layouts := layouts

	push_constant_size := TR_PC_BYTE_SIZE

	swapchain_format := VULKAN_STATE.swapchain.format
	depth_format     := VULKAN_TRADITIONAL_STATE.depth_format

	descriptor := Graphics_Pipeline_Description {
		name                = "BF_GPU.Traditional",
		vertex_shader       = vert,
		fragment_shader     = frag,
		push_constant_size  = push_constant_size,
		descriptor_set_layouts = set_layouts,
		color_formats       = []vk.Format{swapchain_format},
		depth_format        = depth_format,
		depth_test          = true,
		depth_write         = true,
		cull_mode           = {.BACK},
		front_face          = .CLOCKWISE,
		samples             = 1,
	}

	pipeline := vulkan_create_graphics_pipeline(&descriptor)
	if pipeline == nil {
		log.error("[BF_GPU/Vulkan] traditional: vkCreateGraphicsPipelines failed")
		vulkan_backend_destroy_shader_impl(vert)
		vulkan_backend_destroy_shader_impl(frag)
		return false
	}

	VULKAN_TRADITIONAL_STATE.vertex_shader   = vert
	VULKAN_TRADITIONAL_STATE.fragment_shader = frag
	VULKAN_TRADITIONAL_STATE.pipeline        = pipeline
	log.info("[BF_GPU/Vulkan] traditional graphics pipeline built")
	return true
}

// ---------------------------------------------------------------------------
// Frame recording.
// ---------------------------------------------------------------------------

// vulkan_record_traditional_passes emits the depth / swapchain
// transitions, dynamic rendering scope, descriptor bindings, push
// constants, dynamic state, and one vkCmdDrawIndexedIndirect per
// traditional geometry bucket. Called from vulkan_frame() after the
// culling chain and before the swapchain present.
//
// Frame ordering:
//
//   1. Transition depth attachment: UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL.
//   2. Transition swapchain image: present/UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL.
//   3. Begin dynamic rendering (swapchain color + depth).
//   4. Bind pipeline + push constants.
//   5. Bind descriptor sets 0, 1, 2 (frame + persistent + per-pass).
//   6. Set viewport + scissor (dynamic state).
//   7. For each traditional bucket:
//        vkCmdDrawIndexedIndirect(buffer, bucket_offset, 1, stride).
//   8. End dynamic rendering.
//   9. Transition depth back to DEPTH_READ_ONLY_OPTIMAL so the culling
//      pipeline (next frame) can sample it as a depth pyramid source.
//  10. Transition swapchain to PRESENT_SRC_KHR.
//
// Steps 1, 2, 9, 10 are barrier insertions; steps 4-7 are the draw
// work itself. The compute-to-indirect barrier the culling pass
// inserted earlier is the only synchronization required between the
// culling chain and this pass -- vkCmdDrawIndexedIndirect reads the
// instanceCount the culling shaders wrote.
//
// Returns true on a successful recording; false aborts the frame so
// the caller can skip submit/present. The proc never panics on a
// missing depth view -- the initialization check guarantees it is set
// when vulkan_traditional_initialized() returns true.
vulkan_record_traditional_passes :: proc(
	cmd_buffer: vk.CommandBuffer,
	frame: ^Vulkan_Frame,
	image_index: u32,
	frame_ctx: ^Frame_Context_State,
) -> bool {
	if !VULKAN_TRADITIONAL_STATE.initialized do return true
	if cmd_buffer == {} || frame == nil || frame_ctx == nil do return false
	if VULKAN_TRADITIONAL_STATE.pipeline == nil do return true
	if VULKAN_TRADITIONAL_STATE.depth_image == {} ||
	   VULKAN_TRADITIONAL_STATE.depth_image_view == {} {
		return true
	}

	// 1. Depth: UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL.
	vulkan_traditional_transition_depth(cmd_buffer, .UNDEFINED, .DEPTH_ATTACHMENT_OPTIMAL)

	// 2. Swapchain: whatever the previous user left it in ->
	//    COLOR_ATTACHMENT_OPTIMAL. Existing helper handles the
	//    presentation / undefined -> color transition.
	vulkan_transition_swapchain_to_color_attachment(cmd_buffer, image_index)

	// 3. Dynamic rendering scope (swapchain color + depth).
	swapchain_format := VULKAN_STATE.swapchain.format
	depth_format := VULKAN_TRADITIONAL_STATE.depth_format

	color_attachment := vk.RenderingAttachmentInfo {
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = VULKAN_STATE.swapchain.image_views[image_index],
		imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
		loadOp      = .CLEAR,
		storeOp     = .STORE,
		clearValue  = vk.ClearValue {
			color = vk.ClearColorValue{float32 = [4]f32{0.025, 0.025, 0.035, 1.0}},
		},
	}

	depth_attachment := vk.RenderingAttachmentInfo {
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = VULKAN_TRADITIONAL_STATE.depth_image_view,
		imageLayout = .DEPTH_ATTACHMENT_OPTIMAL,
		loadOp      = .CLEAR,
		storeOp     = .STORE,
		clearValue  = vk.ClearValue{depthStencil = vk.ClearDepthStencilValue{depth = 1.0, stencil = 0}},
	}

	rendering_info := vk.RenderingInfo {
		sType                 = .RENDERING_INFO,
		renderArea = vk.Rect2D {
			offset = vk.Offset2D{x = 0, y = 0},
			extent = VULKAN_STATE.swapchain.extent,
		},
		layerCount            = 1,
		colorAttachmentCount  = 1,
		pColorAttachments     = &color_attachment,
		pDepthAttachment       = &depth_attachment,
	}

	vk.CmdBeginRendering(cmd_buffer, &rendering_info)

	// 4. Bind the pipeline + push constants.
	pipeline, layout := vulkan_pipeline_vk(VULKAN_TRADITIONAL_STATE.pipeline)
	if pipeline == {} || layout == {} {
		// Defensive: a zero pipeline handle means the GPU_Backend
		// build step never produced a pipeline. Close the rendering
		// scope and return false so the caller skips submit.
		vk.CmdEndRendering(cmd_buffer)
		log.error("[BF_GPU/Vulkan] traditional: pipeline or layout is nil")
		return false
	}

	vk.CmdBindPipeline(cmd_buffer, .GRAPHICS, pipeline)

	pc := PC_Traditional_Meshlet_Pass {
		frame_global_context_buffer_addr = frame_ctx.buffers[Gpu_Buffer_Kind.Frame_Global_Context].device_addr,
		base_descriptor_offset            = 0,
		material_render_type              = 0,
		disable_cone_culling              = 1,
	}
	pc_bytes := transmute([^]u8)&pc
	pc_size  := TR_PC_BYTE_SIZE
	vk.CmdPushConstants(
		cmd_buffer,
		layout,
		{.VERTEX, .FRAGMENT},
		0,
		pc_size,
		pc_bytes,
	)

	// 5. Bind descriptor sets 0, 1, 2 in one call.
	sets := vulkan_descriptor_sets_get()
	fid := int(VULKAN_STATE.frame_index)
	if fid < 0 || fid >= MAX_FRAMES_IN_FLIGHT {
		vk.CmdEndRendering(cmd_buffer)
		log.errorf("[BF_GPU/Vulkan] traditional: invalid frame_index %d", fid)
		return false
	}

	bound := [3]vk.DescriptorSet {
		sets.frame[fid],
		sets.persistent[fid],
		sets.per_pass[fid],
	}
	vk.CmdBindDescriptorSets(
		cmd_buffer,
		.GRAPHICS,
		layout,
		0,
		3,
		&bound[0],
		0,
		nil,
	)

	// 6. Dynamic viewport + scissor match the swapchain extent.
	viewport := vk.Viewport {
		x        = 0.0,
		y        = 0.0,
		width    = cast(f32)VULKAN_STATE.swapchain.extent.width,
		height   = cast(f32)VULKAN_STATE.swapchain.extent.height,
		minDepth = 0.0,
		maxDepth = 1.0,
	}
	scissor := vk.Rect2D {
		offset = vk.Offset2D{x = 0, y = 0},
		extent = VULKAN_STATE.swapchain.extent,
	}
	vk.CmdSetViewport(cmd_buffer, 0, 1, &viewport)
	vk.CmdSetScissor(cmd_buffer, 0, 1, &scissor)

	// 7. Issue one vkCmdDrawIndexedIndirect per traditional bucket.
	//    The culling shaders populated the instanceCount fields for
	//    visible instances; buckets with zero visible instances are
	//    no-ops at draw time (allowed by the spec).
	indirect_handle := frame_ctx.buffers[Gpu_Buffer_Kind.Global_Indirect_Command].handle
	indirect_entry, indirect_ok := VULKAN_BUFFER_MAP[indirect_handle]
	if !indirect_ok || indirect_entry == nil || indirect_entry.buffer == {} {
		// No indirect command buffer is acceptable (no geometry in the
		// scene). Skip the draws; the rest of the frame proceeds with
		// the clear-only presentation.
		vk.CmdEndRendering(cmd_buffer)
		vulkan_traditional_transition_depth(cmd_buffer, .DEPTH_ATTACHMENT_OPTIMAL, .DEPTH_READ_ONLY_OPTIMAL)
		vulkan_transition_swapchain_to_present(cmd_buffer, image_index)
		return true
	}

	stride := u32(size_of(GPU_Indirect_Command))
	for bucket in TR_BUCKET_TRADITIONAL_FIRST ..= TR_BUCKET_TRADITIONAL_LAST {
		offset := vk.DeviceSize(u64(bucket) * u64(stride))
		vk.CmdDrawIndexedIndirect(
			cmd_buffer,
			indirect_entry.buffer,
			offset,
			1,
			stride,
		)
	}

	// 8. End rendering.
	vk.CmdEndRendering(cmd_buffer)

	// 9. Depth -> DEPTH_READ_ONLY_OPTIMAL so the next frame's culling
	//    pass can sample it as a depth pyramid source. The HiZ
	//    generator is not in v1; this transition is purely
	//    forward-compatible so a future HiZ pass can read the depth
	//    without an extra barrier.
	vulkan_traditional_transition_depth(cmd_buffer, .DEPTH_ATTACHMENT_OPTIMAL, .DEPTH_READ_ONLY_OPTIMAL)

	// 10. Swapchain -> PRESENT_SRC_KHR.
	vulkan_transition_swapchain_to_present(cmd_buffer, image_index)

	return true
}

// vulkan_traditional_transition_depth inserts a synchronization2
// image-memory barrier on the depth attachment. Used at the entry
// (UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL) and the exit
// (DEPTH_ATTACHMENT_OPTIMAL -> DEPTH_READ_ONLY_OPTIMAL) of the
// rendering scope.
vulkan_traditional_transition_depth :: proc(
	cmd_buffer: vk.CommandBuffer,
	old_layout, new_layout: vk.ImageLayout,
) {
	if VULKAN_TRADITIONAL_STATE.depth_image == {} do return

	src_stage: vk.PipelineStageFlags2 = {.TOP_OF_PIPE}
	src_access: vk.AccessFlags2 = {}
	dst_stage: vk.PipelineStageFlags2 = {.LATE_FRAGMENT_TESTS}
	dst_access: vk.AccessFlags2 = {.DEPTH_STENCIL_ATTACHMENT_WRITE}

	#partial switch new_layout {
	case .DEPTH_READ_ONLY_OPTIMAL:
		dst_stage = {.FRAGMENT_SHADER, .COMPUTE_SHADER}
		dst_access = {.DEPTH_STENCIL_ATTACHMENT_READ, .SHADER_READ}
	}

	barrier := vk.ImageMemoryBarrier2 {
		sType               = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask        = src_stage,
		srcAccessMask       = src_access,
		dstStageMask        = dst_stage,
		dstAccessMask       = dst_access,
		oldLayout           = old_layout,
		newLayout           = new_layout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image               = VULKAN_TRADITIONAL_STATE.depth_image,
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask     = {.DEPTH},
			baseMipLevel   = 0,
			levelCount     = 1,
			baseArrayLayer = 0,
			layerCount     = 1,
		},
	}
	dep := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}
	vk.CmdPipelineBarrier2(cmd_buffer, &dep)
}