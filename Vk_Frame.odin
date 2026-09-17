package BF_GPU

import "core:log"
import "core:time"
import vk "vendor:vulkan"

Vulkan_Frame :: struct {
	command_pool:    vk.CommandPool,
	command_buffer:  vk.CommandBuffer,
	// WSI synchronization
	image_available: vk.Semaphore,
	render_finished: vk.Semaphore,
	// Graphics submission that last used this frame slot.
	// 0 means the slot has never been submitted.
	completion_value: u64,
	// True once vulkan_poll_graphics_completion() has released the
	// BF_DAG GPU-completion external node for this slot's submission.
	// Prevents double-signal when polling multiple times across frames.
	completion_signaled: bool,
}

// vulkan_graphics_timeline_counter returns the current value of the
// graphics queue's timeline semaphore via vk.GetSemaphoreCounterValue.
// This is a NON-BLOCKING query: it just reads the device-side counter
// without stalling the host. Returns 0 when the device is unavailable
// or the query fails (any caller in that state treats 0 as "nothing
// completed yet" which is the safe interpretation).
vulkan_graphics_timeline_counter :: proc() -> u64 {
	if !VULKAN_STATE.initialized do return 0
	if VULKAN_STATE.device == nil do return 0
	if VULKAN_STATE.graphics_timeline == cast(vk.Semaphore)0 do return 0

	counter: u64 = 0
	result := vk.GetSemaphoreCounterValue(VULKAN_STATE.device, VULKAN_STATE.graphics_timeline, &counter)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkGetSemaphoreCounterValue failed: %v", result)
		return 0
	}
	return counter
}

// process_completion_signals walks the frame slots and, for every slot
// whose `completion_value` is <= current_counter AND has not yet been
// signaled, fires the BF_DAG GPU-completion external node + reaps the
// per-completion garbage. Returns the number of slots newly signaled.
//
// Pure logic on top of the Vulkan_Frame state — does not touch the
// device itself. `vulkan_poll_graphics_completion` is the wrapper that
// queries the device and delegates here. Splitting the call lets the
// GPU completion tests exercise the state machine without a live
// Vulkan device.
process_completion_signals :: proc(current_counter: u64) -> int {
	signaled := 0
	for &f in VULKAN_STATE.frames {
		if f.completion_value == 0 do continue
		if f.completion_signaled do continue
		if current_counter < f.completion_value do continue

		// Mark BEFORE the signal call: if the signal ever re-enters
		// the poll path (the external-signal path is fully synchronous
		// inside BF_DAG today, but the flag keeps us correct against
		// future async notifications), the flag is the canonical
		// dedup guarantee.
		f.completion_signaled = true
		gpu_completion_signal()
		// Anything queued for destruction tagged <= completion_value
		// is now safe to release; reap it before the next submit so
		// VMA allocations return to the free pool promptly.
		vulkan_collect_garbage(f.completion_value)
		signaled += 1
	}
	return signaled
}

// vulkan_poll_graphics_completion is the non-blocking completion
// detection point. It scans every frame slot whose GPU submission has
// not yet been signalled to BF_DAG and releases the external node for
// any slot whose timeline value has been reached. Safe to call when
// no submissions are in flight (no-op), when the device is gone (no-op),
// or when the graphics timeline has never been created (no-op).
//
// Returns the number of slots newly signaled.
vulkan_poll_graphics_completion :: proc() -> int {
	if !VULKAN_STATE.initialized do return 0
	counter := vulkan_graphics_timeline_counter()
	// The renderer-diagnostics timeline-gag check needs the latest
	// observed counter, not the counter from a successful signal -
	// a counter that never advances is itself the bug the lag warning
	// is meant to surface.
	diag_record_completion_value(counter)
	return process_completion_signals(counter)
}

// vulkan_flush_pending_completion_signals does a final non-blocking
// sweep over the frame slots. Called from vulkan_shutdown after
// vkDeviceWaitIdle has guaranteed every in-flight submission is done,
// so any slot whose completion_value > 0 must have completed and
// must still need a signal. Returns the number of slots signaled
// during the drain; a non-zero count means shutdown caught a frame
// that never reached a poll point while alive (informational, not
// fatal).
vulkan_flush_pending_completion_signals :: proc() -> int {
	if !VULKAN_STATE.initialized do return 0
	counter := vulkan_graphics_timeline_counter()
	return process_completion_signals(counter)
}

//* =========== DRAW ==================
vulkan_frame :: proc(gpu: ^GPU_Scene, ctx: ^Frame_Context_State) -> bool {
	if !VULKAN_STATE.initialized do return false

	// Poll every frame slot for any submission that has reached its
	// timeline value and release the BF_DAG GPU-completion external
	// node for each. This runs first so a slot whose GPU work finished
	// mid-frame (or while the CPU was busy with extraction / upload)
	// still signals promptly, instead of waiting for the next frame
	// to begin recording on its slot. The poll is non-blocking — it
	// never stalls the CPU waiting for the GPU.
	vulkan_poll_graphics_completion()

	frame := &VULKAN_STATE.frames[VULKAN_STATE.frame_index]

	// The frame slot's command buffer and WSI semaphores cannot be
	// reused until its previous graphics submission completed.
	// Blocking here is still required for command-buffer / semaphore
	// reuse safety: the GPU may still be using this slot's
	// image_available / render_finished pair. The signal release for
	// this slot was already handled by the poll above, so we do not
	// signal again here.
	wait_ns: u64 = 0
	if frame.completion_value > 0 {
		wait_begin := time.now()
		if !vulkan_wait_graphics_timeline(frame.completion_value) do return false
		wait_ns = u64(time.duration_nanoseconds(time.diff(time.now(), wait_begin)))
	}
	diag_take_gpu_wait_ns(MODULE_STATE_VALUE.frame_ctx.frame_idx, wait_ns)

	// Reset the upload ring slot for this frame. Safe because the
	// timeline wait above proved the GPU has consumed every prior
	// submission on it (the ring has slot_count == MAX_FRAMES_IN_FLIGHT
	// slots, so each slot is at most one frame behind the CPU).
	vulkan_upload_ring_reset_slot(u32(VULKAN_STATE.frame_index))

	// Slow-path diagnostics: heap-budget query + VMA defrag on their
	// own periods. diag_maybe_tick is internally rate-limited.
	diag_maybe_tick(MODULE_STATE_VALUE.frame_ctx.frame_idx)

	if window_resize_pending() {return vulkan_recreate_swapchain()}

	image_index, acquire_result := vulkan_acquire_next_image(frame)
	if acquire_result == .ERROR_OUT_OF_DATE_KHR {
		return vulkan_recreate_swapchain()
	}

	if acquire_result != .SUCCESS && acquire_result != .SUBOPTIMAL_KHR {
		log.errorf("[BF_GPU/Vulkan] vkAcquireNextImageKHR failed: %v", acquire_result)
		return false
	}

	if !vulkan_begin_command_buffer(frame.command_buffer) do return false

	// Diagnostics: tag the frame label boundary and write the
	// Begin timestamp so the GPU timing path has a reference point.
	vulkan_diag_begin_frame_label(frame.command_buffer, "BF_GPU.Frame")
	vulkan_diag_write_timestamp(frame.command_buffer, .Begin)

	// Per-frame Render_Scene -> GPU_Scene -> Vulkan buffer upload.
	// Recorded into the same command buffer the clear pass + future
	// draws will consume, so the buffer memory barriers it inserts
	// also serialise the GPU-side culling / draw dispatch against the
	// host-side data.
	if gpu != nil && ctx != nil {
		vulkan_diag_begin_frame_label(frame.command_buffer, "BF_GPU.Upload")
		vulkan_upload_scene(frame.command_buffer, ctx, gpu)
		vulkan_diag_end_frame_label(frame.command_buffer)
		vulkan_diag_write_timestamp(frame.command_buffer, .Upload_End)
	}

	// GPU culling: reset indirect commands, run the static-chunk ->
	// static-model -> mesh culling chain, and end with a compute ->
	// graphics barrier so the future draw pass observes the
	// freshly-populated indirect / instance-index buffers. Visibility
	// stays GPU-side; no readback here.
	if gpu != nil && ctx != nil && vulkan_culling_initialized() {
		vulkan_diag_begin_frame_label(frame.command_buffer, "BF_GPU.Culling")
		if !vulkan_record_culling_passes(frame.command_buffer, ctx, gpu) {
			vulkan_diag_end_frame_label(frame.command_buffer)
			return false
		}
		vulkan_culling_to_graphics_barrier(frame.command_buffer)
		vulkan_diag_end_frame_label(frame.command_buffer)
		vulkan_diag_write_timestamp(frame.command_buffer, .Culling_End)
	}

	// Traditional indirect rendering: dynamic-rendering scope with
	// swapchain color + depth, descriptor bindings, push constants,
	// dynamic viewport/scissor, and one vkCmdDrawIndexedIndirect per
	// traditional bucket. Falls back to a clear-only pass when the
	// traditional pipeline is not yet initialised (a transient state
	// during shutdown or after a pipeline-build failure).
	if vulkan_traditional_initialized() {
		vulkan_diag_begin_frame_label(frame.command_buffer, "BF_GPU.Traditional")
		if !vulkan_record_traditional_passes(frame.command_buffer, frame, image_index, ctx) {
			vulkan_diag_end_frame_label(frame.command_buffer)
			return false
		}
		vulkan_diag_end_frame_label(frame.command_buffer)
		vulkan_diag_write_timestamp(frame.command_buffer, .Traditional_End)
	} else {
		vulkan_record_clear_pass(frame, image_index)
		vulkan_diag_write_timestamp(frame.command_buffer, .Traditional_End)
	}

	// Meshlet (task + mesh shader) pass. Runs after the traditional pass
	// when BF_GPU_Mesh is attached AND VK_EXT_mesh_shader is supported;
	// otherwise it is permanently a no-op. The pass reopens the
	// swapchain colour + depth attachment, dispatches one
	// vkCmdDrawMeshTasksIndirect per meshlet geometry bucket
	// (indices 4..7), and leaves the depth attachment in
	// DEPTH_READ_ONLY_OPTIMAL for the next frame's culling pass.
	if vulkan_meshlet_initialized() {
		vulkan_diag_begin_frame_label(frame.command_buffer, "BF_GPU.Meshlet")
		if !vulkan_record_meshlet_passes(frame.command_buffer, frame, image_index, ctx) {
			vulkan_diag_end_frame_label(frame.command_buffer)
			return false
		}
		vulkan_diag_end_frame_label(frame.command_buffer)
		vulkan_diag_write_timestamp(frame.command_buffer, .Meshlet_End)
	}

	vulkan_diag_end_frame_label(frame.command_buffer) // end frame label
	vulkan_diag_write_timestamp(frame.command_buffer, .Submit_End)

	if !vulkan_end_command_buffer(frame.command_buffer) do return false
	if !vulkan_submit_graphics(frame) do return false

	// Read back any GPU timestamps that have landed. The pool is
	// sized to MAX_FRAMES_IN_SLOTS * Timer_Slot.COUNT and the read
	// uses .WAIT so it never stalls the CPU; if the slot is still
	// in-flight the call returns NOT_READY and we skip the roll.
	vulkan_diag_collect_timestamps(MODULE_STATE_VALUE.frame_ctx.frame_idx)

	present_result := vulkan_present(frame, image_index)
	if present_result == .ERROR_OUT_OF_DATE_KHR || present_result == .SUBOPTIMAL_KHR {
		if !vulkan_recreate_swapchain() do return false
	} else if present_result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vQueuePresentKHR failed: %v", present_result)
		return false
	}

	VULKAN_STATE.frame_index = (VULKAN_STATE.frame_index + 1) % MAX_FRAMES_IN_FLIGHT

	return true
}
//* ///////////////////////////////////
vulkan_submit_graphics :: proc(frame: ^Vulkan_Frame) -> bool {
	// Q successful graphics submission gets a unique completion value.
	VULKAN_STATE.graphics_timeline_value += 1
	signal_value := VULKAN_STATE.graphics_timeline_value

	signal_semaphores := [2]vk.Semaphore{frame.render_finished, VULKAN_STATE.graphics_timeline}
	command_buffers := [1]vk.CommandBuffer{frame.command_buffer}
	signal_values := [2]u64{0, signal_value}

	// Dedicated transfer queue path: when the device exposes a
	// separate transfer family, the renderer submits asset-streaming
	// copies onto that queue (see vulkan_submit_transfer below). The
	// graphics queue must wait on the transfer timeline so shader
	// reads see the freshly uploaded asset data. The transfer
	// timeline value starts at 0 (no transfers yet) and the wait is
	// a no-op when nothing has signalled. The integrated-GPU path
	// (no dedicated transfer queue) keeps the wait list at length 1.
	//
	// Arrays are sized to 2 unconditionally so the dedicated path can
	// populate index 1 with the transfer timeline; the integrated
	// path leaves index 1 zeroed and reports waitSemaphoreCount = 1.
	wait_semaphores: [2]vk.Semaphore = {frame.image_available, {}}
	wait_stages: [2]vk.PipelineStageFlags = {{.COLOR_ATTACHMENT_OUTPUT}, {}}
	wait_values: [2]u64 = {0, 0}
	wait_count: u32 = 1
	if VULKAN_STATE.dedicated_transfer_queue_available &&
	   VULKAN_STATE.transfer_timeline != cast(vk.Semaphore)0 {
		wait_semaphores[1] = VULKAN_STATE.transfer_timeline
		wait_stages[1] = vk.PipelineStageFlags{.VERTEX_SHADER, .COMPUTE_SHADER, .FRAGMENT_SHADER}
		wait_values[1] = VULKAN_STATE.transfer_timeline_value
		wait_count = 2
	}

	timeline_info := vk.TimelineSemaphoreSubmitInfo {
		sType                     = .TIMELINE_SEMAPHORE_SUBMIT_INFO,
		waitSemaphoreValueCount   = wait_count,
		pWaitSemaphoreValues      = &wait_values[0],
		signalSemaphoreValueCount = 2,
		pSignalSemaphoreValues    = &signal_values[0],
	}

	submit_info := vk.SubmitInfo {
		sType                = .SUBMIT_INFO,
		pNext                = &timeline_info,
		waitSemaphoreCount   = wait_count,
		pWaitSemaphores      = &wait_semaphores[0],
		pWaitDstStageMask    = &wait_stages[0],
		commandBufferCount   = 1,
		pCommandBuffers      = &command_buffers[0],
		signalSemaphoreCount = 2,
		pSignalSemaphores    = &signal_semaphores[0],
	}
	result := vk.QueueSubmit(VULKAN_STATE.graphics_queue, 1, &submit_info, {})
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkQueueSubmit failed: %v", result)
		// Do not leave the counter in an inconsistent state
		VULKAN_STATE.graphics_timeline_value -= 1
		return false
	}
	frame.completion_value = signal_value
	// A fresh submission has not yet been observed by the poll; clear
	// the per-slot dedup flag so vulkan_poll_graphics_completion
	// signals it once the GPU reaches signal_value. Clearing here also
	// makes the per-slot state well-defined if the slot is being
	// reused after a previous frame's poll already signalled it.
	frame.completion_signaled = false
	return true
}

// vulkan_submit_transfer submits a single command buffer on the
// dedicated transfer queue (when available) and signals
// transfer_timeline at the new value so the graphics queue (which
// waits on transfer_timeline at the start of its submit) sees the
// uploaded data. No-op when the device has no dedicated transfer
// family — in that case transfers happen on the graphics queue.
//
// The asset pipeline (Asset_Sync.odin) calls this once per frame
// after the streaming copies for that frame have been recorded into
// `transfer_cmd`. Recording is the caller's responsibility; this
// proc only handles the submit + timeline signal.
vulkan_submit_transfer :: proc(transfer_cmd: vk.CommandBuffer) -> bool {
	if !VULKAN_STATE.dedicated_transfer_queue_available do return true
	if VULKAN_STATE.transfer_queue == {} do return false

	VULKAN_STATE.transfer_timeline_value += 1
	signal_value := VULKAN_STATE.transfer_timeline_value
	signal_semaphores := [1]vk.Semaphore{VULKAN_STATE.transfer_timeline}
	signal_values := [1]u64{signal_value}

	timeline_info := vk.TimelineSemaphoreSubmitInfo {
		sType                     = .TIMELINE_SEMAPHORE_SUBMIT_INFO,
		waitSemaphoreValueCount   = 0,
		signalSemaphoreValueCount = 1,
		pSignalSemaphoreValues    = &signal_values[0],
	}

	command_buffers := [1]vk.CommandBuffer{transfer_cmd}

	submit_info := vk.SubmitInfo {
		sType                 = .SUBMIT_INFO,
		pNext                 = &timeline_info,
		commandBufferCount    = 1,
		pCommandBuffers       = &command_buffers[0],
		signalSemaphoreCount  = 1,
		pSignalSemaphores     = &signal_semaphores[0],
	}
	result := vk.QueueSubmit(VULKAN_STATE.transfer_queue, 1, &submit_info, {})
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] transfer QueueSubmit failed: %v", result)
		VULKAN_STATE.transfer_timeline_value -= 1
		return false
	}
	return true
}
vulkan_present :: proc(frame: ^Vulkan_Frame, image_index: u32) -> vk.Result {
	swapchains := [1]vk.SwapchainKHR{VULKAN_STATE.swapchain.handle}
	image_indices := [1]u32{image_index}
	wait_semaphores := [1]vk.Semaphore{frame.render_finished}

	present_info := vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &wait_semaphores[0],
		swapchainCount     = 1,
		pSwapchains        = &swapchains[0],
		pImageIndices      = &image_indices[0],
	}

	return vk.QueuePresentKHR(VULKAN_STATE.present_queue, &present_info)
}
vulkan_recreate_swapchain :: proc() -> bool {
	if VULKAN_STATE.device == nil do return false

	vk.DeviceWaitIdle(VULKAN_STATE.device)
	vulkan_destroy_swapchain()

	if !vulkan_create_swapchain() do return false
	if !vulkan_create_swapchain_image_views() {
		vulkan_destroy_swapchain()
		return false
	}
	// Dependent resources (HDR target, depth, HiZ, framebuffers) are
	// recreated against the new swapchain extent via
	// vulkan_recreate_dependent_resources.
	vulkan_recreate_dependent_resources()
	window_clear_resize()

	log.infof(
		"[BF_GPU/Vulkan] Swapchain recreated: %dx%d",
		VULKAN_STATE.swapchain.extent.width,
		VULKAN_STATE.swapchain.extent.height,
	)
	return true
}

// vulkan_recreate_dependent_resources is the swapchain-recreation hook
// for render targets whose lifetime is tied to the swapchain extent.
// The traditional depth attachment is rebuilt against the new extent;
// the traditional pipeline itself does not need to be rebuilt
// because dynamic rendering picks up the new colour format / depth
// view at record time.
vulkan_recreate_dependent_resources :: proc() {
	vulkan_traditional_recreate_depth_attachment()
}
vulkan_wait_graphics_timeline :: proc(value: u64) -> bool {
	if value == 0 do return true

	semaphores := [1]vk.Semaphore{VULKAN_STATE.graphics_timeline}
	values := [1]u64{value}
	wait_info := vk.SemaphoreWaitInfo {
		sType          = .SEMAPHORE_WAIT_INFO,
		semaphoreCount = 1,
		pSemaphores    = &semaphores[0],
		pValues        = &values[0],
	}

	result := vk.WaitSemaphores(VULKAN_STATE.device, &wait_info, 0xFFFFFFFFFFFFFFFF)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkWaitSemaphores failed: %v", result)
		return false
	}
	return true
}
vulkan_transition_swapchain_image_to_attachment :: proc(
	command_buffer: vk.CommandBuffer,
	image_index: u32,
) {
	swapchain := &VULKAN_STATE.swapchain

	barrier := vk.ImageMemoryBarrier2 {
		sType = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask = {.TOP_OF_PIPE},
		srcAccessMask = {},
		dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
		oldLayout = swapchain.image_layouts[image_index],
		newLayout = .ATTACHMENT_OPTIMAL,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = swapchain.images[image_index],
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}

	dependency_info := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}

	swapchain.image_layouts[image_index] = .PRESENT_SRC_KHR

	vk.CmdPipelineBarrier2(command_buffer, &dependency_info)
}
vulkan_transition_swapchain_image_to_present :: proc(
	command_buffer: vk.CommandBuffer,
	image: vk.Image,
) {
	barrier := vk.ImageMemoryBarrier2 {
		sType = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		srcAccessMask = {.COLOR_ATTACHMENT_WRITE},
		dstStageMask = {.BOTTOM_OF_PIPE},
		dstAccessMask = {},
		oldLayout = .ATTACHMENT_OPTIMAL,
		newLayout = .PRESENT_SRC_KHR,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = image,
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}

	dependency_info := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}

	vk.CmdPipelineBarrier2(command_buffer, &dependency_info)
}
vulkan_transition_swapchain_image :: proc(
	command_buffer: vk.CommandBuffer,
	image_index: u32,
	old_layout: vk.ImageLayout,
	new_layout: vk.ImageLayout,
) {
	swapchain := &VULKAN_STATE.swapchain

	barrier := vk.ImageMemoryBarrier2 {
		sType = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask = {.TOP_OF_PIPE},
		srcAccessMask = {},
		dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
		oldLayout = old_layout,
		newLayout = new_layout,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = swapchain.images[image_index],
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}

	dependency_info := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}

	vk.CmdPipelineBarrier2(command_buffer, &dependency_info)
}
vulkan_record_swapchain_render :: proc(frame: ^Vulkan_Frame, image_index: u32) {
	swapchain := &VULKAN_STATE.swapchain

	vulkan_transition_swapchain_to_color_attachment(frame.command_buffer, image_index)

	clear_value := vk.ClearValue {
		color = vk.ClearColorValue{float32 = [4]f32{0.025, 0.025, 0.035, 1.0}},
	}

	color_attachment := vk.RenderingAttachmentInfo {
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = swapchain.image_views[image_index],
		imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
		loadOp      = .CLEAR,
		storeOp     = .STORE,
		clearValue  = clear_value,
	}

	rendering_info := vk.RenderingInfo {
		sType = .RENDERING_INFO,
		renderArea = vk.Rect2D{offset = vk.Offset2D{x = 0, y = 0}, extent = swapchain.extent},
		layerCount = 1,
		colorAttachmentCount = 1,
		pColorAttachments = &color_attachment,
	}

	vk.CmdBeginRendering(frame.command_buffer, &rendering_info)

	// No graphics pipeline yet.
	// This establishes the dynamic-rendering scope.

	vk.CmdEndRendering(frame.command_buffer)

	vulkan_transition_swapchain_to_present(frame.command_buffer, image_index)
}
vulkan_transition_swapchain_to_color_attachment :: proc(
	command_buffer: vk.CommandBuffer,
	image_index: u32,
) {
	swapchain := &VULKAN_STATE.swapchain

	barrier := vk.ImageMemoryBarrier2 {
		sType = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask = {},
		srcAccessMask = {},
		dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
		oldLayout = swapchain.image_layouts[image_index],
		newLayout = .COLOR_ATTACHMENT_OPTIMAL,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = swapchain.images[image_index],
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}

	dependency_info := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}

	vk.CmdPipelineBarrier2(command_buffer, &dependency_info)

	swapchain.image_layouts[image_index] = .COLOR_ATTACHMENT_OPTIMAL
}
vulkan_transition_swapchain_to_present :: proc(
	command_buffer: vk.CommandBuffer,
	image_index: u32,
) {
	swapchain := &VULKAN_STATE.swapchain

	barrier := vk.ImageMemoryBarrier2 {
		sType = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		srcAccessMask = {.COLOR_ATTACHMENT_WRITE},
		dstStageMask = {},
		dstAccessMask = {},
		oldLayout = .COLOR_ATTACHMENT_OPTIMAL,
		newLayout = .PRESENT_SRC_KHR,
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		image = swapchain.images[image_index],
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		},
	}

	dependency_info := vk.DependencyInfo {
		sType                   = .DEPENDENCY_INFO,
		imageMemoryBarrierCount = 1,
		pImageMemoryBarriers    = &barrier,
	}

	vk.CmdPipelineBarrier2(command_buffer, &dependency_info)

	swapchain.image_layouts[image_index] = .PRESENT_SRC_KHR
}
vulkan_record_clear_pass :: proc(frame: ^Vulkan_Frame, image_index: u32) {
	swapchain := &VULKAN_STATE.swapchain

	vulkan_transition_swapchain_to_color_attachment(frame.command_buffer, image_index)

	clear_value := vk.ClearValue {
		color = vk.ClearColorValue{float32 = [4]f32{0.02, 0.025, 0.04, 1.0}},
	}

	color_attachment := vk.RenderingAttachmentInfo {
		sType       = .RENDERING_ATTACHMENT_INFO,
		imageView   = swapchain.image_views[image_index],
		imageLayout = .COLOR_ATTACHMENT_OPTIMAL,
		loadOp      = .CLEAR,
		storeOp     = .STORE,
		clearValue  = clear_value,
	}

	rendering_info := vk.RenderingInfo {
		sType = .RENDERING_INFO,
		renderArea = vk.Rect2D{offset = vk.Offset2D{x = 0, y = 0}, extent = swapchain.extent},
		layerCount = 1,
		colorAttachmentCount = 1,
		pColorAttachments = &color_attachment,
	}

	vk.CmdBeginRendering(frame.command_buffer, &rendering_info)

	vk.CmdEndRendering(frame.command_buffer)

	vulkan_transition_swapchain_to_present(frame.command_buffer, image_index)
}
