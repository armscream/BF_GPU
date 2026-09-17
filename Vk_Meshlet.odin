// BF_GPU/Vk_Meshlet.odin
//
// Meshlet (task + mesh shader) graphics pipeline and per-frame recording.
//
// The meshlet path is the optional, GPU-driven, mesh-shader extension
// the BF_GPU_Mesh extension module contributes. The renderer only
// builds this pipeline when:
//
//   - VULKAN_STATE.mesh_shader_supported is true (VK_EXT_mesh_shader
//     exposed by the physical device and the task/mesh features were
//     reported), and
//   - the BF_GPU_Mesh extension is attached to the renderer extension
//     point AND has registered a Meshlet_Pipeline_Descriptor.
//
// The pipeline mirrors the traditional pipeline's bindless descriptor
// layout (set 0 = frame, set 1 = persistent/bindless, set 2 = per-pass
// HiZ depth pyramid), share constant buffer, and dynamic rendering
// scope, so a frame can present traditional and meshlet geometry into
// the same swapchain without rebinding most of the state.
//
// ---------------------------------------------------------------------------
// Per-frame integration
// ---------------------------------------------------------------------------
//
// vulkan_record_meshlet_passes runs after vulkan_record_traditional_passes
// inside vulkan_frame. The two passes share the same swapchain colour
// + depth attachment: the traditional pass transitions depth into
// DEPTH_READ_ONLY_OPTIMAL on exit, so the meshlet pass re-opens the
// same scope against DEPTH_ATTACHMENT_OPTIMAL again. This is intentional
// - the pass ordering (opaque first, then meshlet) matches the bucket
// numbering (traditional = 0..3, meshlet = 4..7) the culling pipeline
// already produces, and avoids a redundant layout transition.
//
// Each meshlet bucket issues exactly one vkCmdDrawMeshTasksIndirect,
// reading its groupCountX from the meshlet region of the global
// indirect command buffer (byte offset = traditionalCount * 20). When
// groupCountX == 0 the draw is a no-op at the GPU side, so a frame
// with zero meshlet geometry skips work entirely.
//
// ---------------------------------------------------------------------------
// Bucket numbering
// ---------------------------------------------------------------------------
//
//   0..3 : Traditional  *  (Opaque/Transparent)  *  (Back/None)
//   4..7 : Mesh         *  (Opaque/Transparent)  *  (Back/None)
//
// The four meshlet buckets map to render_bucket_index(pipeline=Mesh)
// in the same way the four traditional buckets map to render_bucket_index
// (pipeline=Traditional). Culling shaders index them through
// pipeline_offset = 1 (PIPELINE_MESHLET) in Gpu_Mesh_Allocation.

package BF_GPU

import "core:log"
import "core:strings"
import vk "vendor:vulkan"

// ---------------------------------------------------------------------------
// State.
// ---------------------------------------------------------------------------

@(private)
VULKAN_MESHLET_STATE: Vulkan_Meshlet_State

Vulkan_Meshlet_State :: struct {
	// Compiled shaders (rawptr handles into VULKAN_SHADER_MAP).
	task_shader:  rawptr,
	mesh_shader:  rawptr,
	fragment_shader: rawptr,

	// Pipeline object + layout (rawptr handle into VULKAN_PIPELINE_MAP).
	pipeline: rawptr,

	// Descriptor used to build the pipeline. The descriptor is owned
	// by the extension; we hold a copy of the path strings + max
	// meshlets-per-workgroup so vulkan_meshlet_shutdown can release
	// the shader handles even after the extension detaches.
	max_meshlets_per_wg: u32,

	initialized: bool,
}

// ML_BUCKET_MESHLET_FIRST / _LAST mark the inclusive range of bucket
// indices the meshlet pass dispatches. With the default
// GEOMETRY_BUCKET_COUNT of 8 and PIPELINE_MESHLET == 1, the four
// meshlet buckets occupy indices 4..7; the four traditional buckets
// (0..3) belong to the traditional pass.
ML_BUCKET_MESHLET_FIRST :: 4
ML_BUCKET_MESHLET_LAST  :: 7

// ML_INDIRECT_CMD_STRIDE_BYTES is the stride of the meshlet region's
// VkDrawMeshTasksIndirectCommandEXT entries. The GLSL
// VK_GLOBAL_MESHLET_CMD_STRIDE_BYTES constant pins the same value on
// the shader side; tests_vulkan_meshlet asserts the two stay in sync.
ML_INDIRECT_CMD_STRIDE_BYTES :: u32(12)

// ML_TRADITIONAL_CMD_STRIDE_BYTES is the stride of the traditional
// region's VkDrawIndexedIndirectCommand entries. Mirrors the GLSL
// VK_GLOBAL_TRADITIONAL_CMD_STRIDE_BYTES.
ML_TRADITIONAL_CMD_STRIDE_BYTES :: u32(20)

// ---------------------------------------------------------------------------
// Init / shutdown.
// ---------------------------------------------------------------------------

// vulkan_meshlet_init compiles the task + mesh shaders from the
// BF_GPU_Mesh extension's Meshlet_Pipeline_Descriptor and builds the
// graphics pipeline against the bindless descriptor layouts. Returns
// false on any failure; the renderer falls back to a no-op meshlet
// pass in that case (the traditional pass keeps working).
//
// This proc is gated on VULKAN_STATE.mesh_shader_supported. When
// VK_EXT_mesh_shader is unavailable the proc returns false without
// touching any state, so the meshlet pass is permanently skipped and
// BF_GPU_Mesh is a no-op regardless of its own attach state.
vulkan_meshlet_init :: proc() -> bool {
	if VULKAN_MESHLET_STATE.initialized {
		log.warn("[BF_GPU/Vulkan] meshlet_init called twice; ignoring")
		return true
	}
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] meshlet_init before Vulkan device creation")
		return false
	}
	if !VULKAN_STATE.mesh_shader_supported {
		log.info("[BF_GPU/Vulkan] mesh shaders not supported by physical device; meshlet pipeline disabled")
		return false
	}
	if !vulkan_descriptor_initialized() {
		log.error("[BF_GPU/Vulkan] meshlet_init before descriptor model init")
		return false
	}
	if VULKAN_STATE.swapchain.handle == {} {
		log.error("[BF_GPU/Vulkan] meshlet_init before swapchain exists")
		return false
	}

	// BF_GPU_Mesh is the only contributor; resolve its descriptor
	// and use the task/mesh shader paths it hands us.
	descriptor_raw := meshlet_pipeline_descriptor_get()
	if descriptor_raw == nil {
		log.info("[BF_GPU/Vulkan] meshlet descriptor not registered; meshlet pipeline disabled")
		return false
	}
	descriptor := cast(^Meshlet_Pipeline_Descriptor)descriptor_raw
	if descriptor.task_shader == nil || descriptor.mesh_shader == nil {
		log.warn("[BF_GPU/Vulkan] meshlet descriptor missing task/mesh shader paths")
		return false
	}

	if !vulkan_meshlet_build_pipeline(descriptor) {
		log.error("[BF_GPU/Vulkan] meshlet pipeline build failed")
		vulkan_meshlet_shutdown()
		return false
	}

	VULKAN_MESHLET_STATE.initialized = true
	log.info("[BF_GPU/Vulkan] meshlet (task + mesh) pipeline initialized")
	return true
}

// vulkan_meshlet_shutdown releases the meshlet pipeline + shader
// handles. Safe to call from a partial-init failure path; safe to call
// multiple times.
vulkan_meshlet_shutdown :: proc() {
	if !VULKAN_MESHLET_STATE.initialized {
		// Even when init never ran we may have shader handles from a
		// failed build; clean them up so a re-init does not leak
		// stale refcounts.
		if VULKAN_MESHLET_STATE.pipeline != nil {
			vulkan_backend_destroy_pipeline_impl(VULKAN_MESHLET_STATE.pipeline)
		}
		if VULKAN_MESHLET_STATE.task_shader != nil {
			vulkan_backend_destroy_shader_impl(VULKAN_MESHLET_STATE.task_shader)
		}
		if VULKAN_MESHLET_STATE.mesh_shader != nil {
			vulkan_backend_destroy_shader_impl(VULKAN_MESHLET_STATE.mesh_shader)
		}
		if VULKAN_MESHLET_STATE.fragment_shader != nil {
			vulkan_backend_destroy_shader_impl(VULKAN_MESHLET_STATE.fragment_shader)
		}
		VULKAN_MESHLET_STATE = {}
		return
	}

	if VULKAN_MESHLET_STATE.pipeline != nil {
		vulkan_backend_destroy_pipeline_impl(VULKAN_MESHLET_STATE.pipeline)
	}
	if VULKAN_MESHLET_STATE.task_shader != nil {
		vulkan_backend_destroy_shader_impl(VULKAN_MESHLET_STATE.task_shader)
	}
	if VULKAN_MESHLET_STATE.mesh_shader != nil {
		vulkan_backend_destroy_shader_impl(VULKAN_MESHLET_STATE.mesh_shader)
	}
	if VULKAN_MESHLET_STATE.fragment_shader != nil {
		vulkan_backend_destroy_shader_impl(VULKAN_MESHLET_STATE.fragment_shader)
	}
	VULKAN_MESHLET_STATE = {}
}

// vulkan_meshlet_initialized exposes the state flag for tests and the
// frame recording path.
vulkan_meshlet_initialized :: proc() -> bool {
	return VULKAN_MESHLET_STATE.initialized
}

// ---------------------------------------------------------------------------
// Pipeline construction.
// ---------------------------------------------------------------------------

// vulkan_meshlet_build_pipeline compiles the task + mesh + fragment
// shaders from the BF_GPU_Mesh Meshlet_Pipeline_Descriptor and
// builds the meshlet graphics pipeline against the same bindless
// descriptor set layouts the traditional pipeline uses.
vulkan_meshlet_build_pipeline :: proc(descriptor: ^Meshlet_Pipeline_Descriptor) -> bool {
	// The meshlet path produces a flat colour + per-fragment visibility
	// info into the swapchain. The fragment shader is the same
	// Traditional.frag that the traditional pipeline uses; the task
	// and mesh shaders come from BF_GPU_Mesh. The pipeline does NOT
	// declare a vertex_shader; vertex data flows through buffer
	// device address SSBOs the task/mesh pair consumes.
	fragment_path := strings.clone_to_cstring(
		"Passes/Shading/Traditional/Traditional.frag",
		context.temp_allocator,
	)
	task_path := strings.clone_to_cstring(string(descriptor.task_shader), context.temp_allocator)
	mesh_path := strings.clone_to_cstring(string(descriptor.mesh_shader), context.temp_allocator)

	frag := vulkan_backend_compile_shader_impl(fragment_path, "fragment")
	if frag == nil {
		log.error("[BF_GPU/Vulkan] meshlet: failed to compile Traditional.frag")
		return false
	}
	task := vulkan_backend_compile_shader_impl(task_path, "task")
	if task == nil {
		log.error("[BF_GPU/Vulkan] meshlet: failed to compile task shader")
		vulkan_backend_destroy_shader_impl(frag)
		return false
	}
	mesh := vulkan_backend_compile_shader_impl(mesh_path, "mesh")
	if mesh == nil {
		log.error("[BF_GPU/Vulkan] meshlet: failed to compile mesh shader")
		vulkan_backend_destroy_shader_impl(task)
		vulkan_backend_destroy_shader_impl(frag)
		return false
	}

	layouts := vulkan_descriptor_pipeline_layouts()
	descriptor_set_layouts := layouts

	// The Traditional.vert push-constant block (PC_Traditional_Meshlet_Pass)
	// is reused by Meshlet.task / Meshlet.mesh. The byte size must
	// match TR_PC_BYTE_SIZE the traditional pass uses.
	push_constant_size := TR_PC_BYTE_SIZE

	swapchain_format := VULKAN_STATE.swapchain.format
	depth_format := vulkan_traditional_depth_format()

	pipeline_desc := Graphics_Pipeline_Description {
		name                = "BF_GPU.Meshlet",
		vertex_shader       = nil,
		fragment_shader     = frag,
		task_shader         = task,
		mesh_shader         = mesh,
		push_constant_size  = push_constant_size,
		descriptor_set_layouts = descriptor_set_layouts,
		color_formats       = []vk.Format{swapchain_format},
		depth_format        = depth_format,
		depth_test          = true,
		depth_write         = true,
		cull_mode           = {.BACK},
		front_face          = .CLOCKWISE,
		samples             = 1,
	}

	pipeline := vulkan_create_graphics_pipeline(&pipeline_desc)
	if pipeline == nil {
		log.error("[BF_GPU/Vulkan] meshlet: vkCreateGraphicsPipelines failed")
		vulkan_backend_destroy_shader_impl(task)
		vulkan_backend_destroy_shader_impl(mesh)
		vulkan_backend_destroy_shader_impl(frag)
		return false
	}

	VULKAN_MESHLET_STATE.task_shader      = task
	VULKAN_MESHLET_STATE.mesh_shader      = mesh
	VULKAN_MESHLET_STATE.fragment_shader  = frag
	VULKAN_MESHLET_STATE.pipeline         = pipeline
	VULKAN_MESHLET_STATE.max_meshlets_per_wg = descriptor.max_meshlets_per_wg
	log.infof(
		"[BF_GPU/Vulkan] meshlet pipeline built (max_meshlets_per_wg=%d)",
		VULKAN_MESHLET_STATE.max_meshlets_per_wg,
	)
	return true
}

// vulkan_traditional_depth_format returns the D32_SFLOAT format the
// traditional pass uses for the depth attachment. The meshlet pass
// reuses the same attachment, so the pipeline's depth format must
// match exactly. Returns UNDEFINED when the traditional state is not
// initialized (caller should fall through to the clear-only fallback).
vulkan_traditional_depth_format :: proc() -> vk.Format {
	if vulkan_traditional_initialized() {
		return VULKAN_TRADITIONAL_STATE.depth_format
	}
	return vk.Format.UNDEFINED
}

// ---------------------------------------------------------------------------
// Per-frame recording.
// ---------------------------------------------------------------------------

// vulkan_record_meshlet_passes emits the depth/swapchain transitions,
// dynamic rendering scope, descriptor bindings, push constants, and
// one vkCmdDrawMeshTasksIndirect per meshlet geometry bucket. Called
// from vulkan_frame() after the traditional pass and before the
// swapchain present.
//
// Frame ordering:
//
//   1. Re-open depth attachment: DEPTH_READ_ONLY_OPTIMAL -> DEPTH_ATTACHMENT_OPTIMAL.
//   2. Swapchain: PRESENT_SRC_KHR -> COLOR_ATTACHMENT_OPTIMAL.
//   3. Begin dynamic rendering (swapchain color + depth).
//   4. Bind the meshlet pipeline + push constants.
//   5. Bind descriptor sets 0, 1, 2 (frame + persistent + per-pass).
//   6. Set viewport + scissor (dynamic state).
//   7. For each meshlet bucket (4..7):
//        vkCmdDrawMeshTasksIndirect(buffer, meshlet_region_offset, 1, 12).
//   8. End dynamic rendering.
//   9. Depth -> DEPTH_READ_ONLY_OPTIMAL.
//  10. Swapchain -> PRESENT_SRC_KHR.
//
// The meshlet region sits at byte offset (traditionalCount * 20) of the
// global indirect command buffer. Each entry is 12 bytes (one
// VkDrawMeshTasksIndirectCommandEXT). The culling shaders populated
// the groupCountX fields during the previous chain.
//
// Returns true on a successful recording; false aborts the frame so
// the caller can skip submit/present. The proc never panics on a
// missing pipeline - the initialization check guarantees it is set
// when vulkan_meshlet_initialized() returns true.
vulkan_record_meshlet_passes :: proc(
	cmd_buffer: vk.CommandBuffer,
	frame: ^Vulkan_Frame,
	image_index: u32,
	frame_ctx: ^Frame_Context_State,
) -> bool {
	if !VULKAN_MESHLET_STATE.initialized do return true
	if cmd_buffer == {} || frame == nil || frame_ctx == nil do return false
	if VULKAN_MESHLET_STATE.pipeline == nil do return true
	if !vulkan_traditional_initialized() {
		// The meshlet pass shares the traditional depth attachment.
		// Without the traditional pass we cannot write depth, so we
		// silently skip the meshlet draw path.
		return true
	}
	if VULKAN_TRADITIONAL_STATE.depth_image == {} ||
	   VULKAN_TRADITIONAL_STATE.depth_image_view == {} {
		return true
	}

	// 1. Depth: DEPTH_READ_ONLY_OPTIMAL -> DEPTH_ATTACHMENT_OPTIMAL.
	vulkan_meshlet_transition_depth(cmd_buffer, .DEPTH_READ_ONLY_OPTIMAL, .DEPTH_ATTACHMENT_OPTIMAL)

	// 2. Swapchain -> COLOR_ATTACHMENT_OPTIMAL.
	vulkan_transition_swapchain_to_color_attachment(cmd_buffer, image_index)

	// 3. Dynamic rendering scope.
	color_attachment := vk.RenderingAttachmentInfo {
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = VULKAN_STATE.swapchain.image_views[image_index],
		imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
		loadOp      = .LOAD,
		storeOp     = .STORE,
		clearValue  = vk.ClearValue{color = vk.ClearColorValue{float32 = [4]f32{0, 0, 0, 0}}},
	}
	depth_attachment := vk.RenderingAttachmentInfo {
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = VULKAN_TRADITIONAL_STATE.depth_image_view,
		imageLayout = .DEPTH_ATTACHMENT_OPTIMAL,
		loadOp      = .LOAD,
		storeOp     = .STORE,
		clearValue  = vk.ClearValue{depthStencil = vk.ClearDepthStencilValue{depth = 1.0, stencil = 0}},
	}
	rendering_info := vk.RenderingInfo {
		sType                = .RENDERING_INFO,
		renderArea = vk.Rect2D {
			offset = vk.Offset2D{x = 0, y = 0},
			extent = VULKAN_STATE.swapchain.extent,
		},
		layerCount           = 1,
		colorAttachmentCount = 1,
		pColorAttachments    = &color_attachment,
		pDepthAttachment     = &depth_attachment,
	}
	vk.CmdBeginRendering(cmd_buffer, &rendering_info)

	// 4. Bind pipeline + push constants.
	pipeline, layout := vulkan_pipeline_vk(VULKAN_MESHLET_STATE.pipeline)
	if pipeline == {} || layout == {} {
		vk.CmdEndRendering(cmd_buffer)
		log.error("[BF_GPU/Vulkan] meshlet: pipeline or layout is nil")
		return false
	}
	vk.CmdBindPipeline(cmd_buffer, .GRAPHICS, pipeline)

	pc := PC_Traditional_Meshlet_Pass {
		frame_global_context_buffer_addr = frame_ctx.buffers[Gpu_Buffer_Kind.Frame_Global_Context].device_addr,
		base_descriptor_offset            = 0,
		material_render_type              = 0,
		disable_cone_culling              = 0,
	}
	pc_bytes := cast([^]u8)&pc
	pc_size  := TR_PC_BYTE_SIZE
	vk.CmdPushConstants(
		cmd_buffer,
		layout,
		{.TASK_EXT, .TASK_NV, .MESH_EXT, .MESH_NV, .FRAGMENT},
		0,
		pc_size,
		pc_bytes,
	)

	// 5. Bind descriptor sets 0, 1, 2.
	sets := vulkan_descriptor_sets_get()
	fid := int(VULKAN_STATE.frame_index)
	if fid < 0 || fid >= MAX_FRAMES_IN_FLIGHT {
		vk.CmdEndRendering(cmd_buffer)
		log.errorf("[BF_GPU/Vulkan] meshlet: invalid frame_index %d", fid)
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

	// 6. Viewport + scissor.
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

	// 7. Issue one vkCmdDrawMeshTasksIndirect per meshlet bucket.
	indirect_handle := frame_ctx.buffers[Gpu_Buffer_Kind.Global_Indirect_Command].handle
	indirect_entry, indirect_ok := VULKAN_BUFFER_MAP[indirect_handle]
	if !indirect_ok || indirect_entry == nil || indirect_entry.buffer == {} {
		// No indirect command buffer is acceptable (no geometry).
		// Skip the draws; the rest of the frame proceeds normally.
		vk.CmdEndRendering(cmd_buffer)
		vulkan_meshlet_transition_depth(cmd_buffer, .DEPTH_ATTACHMENT_OPTIMAL, .DEPTH_READ_ONLY_OPTIMAL)
		vulkan_transition_swapchain_to_present(cmd_buffer, image_index)
		return true
	}

	// The meshlet region starts at byte offset (traditionalCount * 20).
	// Each meshlet bucket entry is a VkDrawMeshTasksIndirectCommandEXT
	// (12 bytes / 3 u32). The bucket index in ML_BUCKET_MESHLET_FIRST..LAST
	// is local to the meshlet region, so the device-side stride and
	// offset must be in meshlet-region coordinates.
	stride := ML_INDIRECT_CMD_STRIDE_BYTES
	meshlet_region_offset := vk.DeviceSize(u64(ML_TRADITIONAL_CMD_STRIDE_BYTES) * 4)
	for bucket in ML_BUCKET_MESHLET_FIRST ..= ML_BUCKET_MESHLET_LAST {
		local_bucket := bucket - ML_BUCKET_MESHLET_FIRST
		offset := meshlet_region_offset + vk.DeviceSize(u64(local_bucket) * u64(stride))
		vk.CmdDrawMeshTasksIndirectEXT(
			cmd_buffer,
			indirect_entry.buffer,
			offset,
			1,
			stride,
		)
	}

	// 8. End rendering.
	vk.CmdEndRendering(cmd_buffer)

	// 9. Depth -> DEPTH_READ_ONLY_OPTIMAL (so the next frame's culling
	//    can sample it; same forward-compat rationale as the
	//    traditional pass).
	vulkan_meshlet_transition_depth(cmd_buffer, .DEPTH_ATTACHMENT_OPTIMAL, .DEPTH_READ_ONLY_OPTIMAL)

	// 10. Swapchain -> PRESENT_SRC_KHR.
	vulkan_transition_swapchain_to_present(cmd_buffer, image_index)

	return true
}

// vulkan_meshlet_transition_depth inserts a synchronization2 image
// memory barrier on the shared depth attachment. Mirrors
// vulkan_traditional_transition_depth; the meshlet pass needs its own
// copy because the transitions in/out are different (the meshlet pass
// reopens the scope the traditional pass closed).
vulkan_meshlet_transition_depth :: proc(
	cmd_buffer: vk.CommandBuffer,
	old_layout, new_layout: vk.ImageLayout,
) {
	if VULKAN_TRADITIONAL_STATE.depth_image == {} do return

	src_stage: vk.PipelineStageFlags2 = {.LATE_FRAGMENT_TESTS}
	src_access: vk.AccessFlags2 = {.DEPTH_STENCIL_ATTACHMENT_WRITE}
	dst_stage: vk.PipelineStageFlags2 = {.LATE_FRAGMENT_TESTS}
	dst_access: vk.AccessFlags2 = {.DEPTH_STENCIL_ATTACHMENT_WRITE}

	#partial switch new_layout {
	case .DEPTH_READ_ONLY_OPTIMAL:
		dst_stage = {.FRAGMENT_SHADER, .COMPUTE_SHADER}
		dst_access = {.DEPTH_STENCIL_ATTACHMENT_READ, .SHADER_READ}
	}
	#partial switch old_layout {
	case .DEPTH_READ_ONLY_OPTIMAL:
		src_stage = {.FRAGMENT_SHADER, .COMPUTE_SHADER}
		src_access = {.DEPTH_STENCIL_ATTACHMENT_READ, .SHADER_READ}
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
