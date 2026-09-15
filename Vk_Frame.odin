package BF_GPU

import "core:log"
import vk "vendor:vulkan"
/*
GPU_Frame_Graph :: struct {
    passed: [dynamic]GPU_Pass,
}
*/

Vulkan_Frame :: struct {
	command_pool:     vk.CommandPool,
	command_buffer:   vk.CommandBuffer,
	// WSI synchronization
	image_available:  vk.Semaphore,
	render_finished:  vk.Semaphore,
	// Graphics submission that last used this frame slot.
	// 0 means the slot has never been submitted.
	completion_value: u64,
}

//* =========== DRAW ==================
vulkan_frame :: proc() -> bool {
	if !VULKAN_STATE.initialized do return false
	frame := &VULKAN_STATE.frames[VULKAN_STATE.frame_index]

	// The frame slot's command buffer and WSI semaphores cannot be
	// reused until its previous graphics submission completed.
	if !vulkan_wait_graphics_timeline(frame.completion_value) do return false

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

	vulkan_record_swapchain_render(frame, image_index)

	if !vulkan_end_command_buffer(frame.command_buffer) do return false
	if !vulkan_submit_graphics(frame) do return false

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

	wait_semaphores := [1]vk.Semaphore{frame.image_available}
	wait_stage := vk.PipelineStageFlags{vk.PipelineStageFlag.COLOR_ATTACHMENT_OUTPUT}
	wait_stages := [1]vk.PipelineStageFlags{wait_stage}
	signal_semaphores := [2]vk.Semaphore{frame.render_finished, VULKAN_STATE.graphics_timeline}

	command_buffers := [1]vk.CommandBuffer{frame.command_buffer}

	wait_values := [1]u64{0}
	signal_values := [2]u64{0, signal_value}

	timeline_info := vk.TimelineSemaphoreSubmitInfo {
		sType                     = .TIMELINE_SEMAPHORE_SUBMIT_INFO,
		waitSemaphoreValueCount   = 1,
		pWaitSemaphoreValues      = &wait_values[0],
		signalSemaphoreValueCount = 2,
		pSignalSemaphoreValues    = &signal_values[0],
	}

	submit_info := vk.SubmitInfo {
		sType                = .SUBMIT_INFO,
		pNext                = &timeline_info,
		waitSemaphoreCount   = 1,
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
	window_clear_resize()

	log.infof(
		"[BF_GPU/Vulkan] Swapchain recreated: %dx%d",
		VULKAN_STATE.swapchain.extent.width,
		VULKAN_STATE.swapchain.extent.height,
	)
	return true
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
		log.errorf("[BF_GPU/Vulkan] vkWaitSemaphores failed: &v", result)
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

	vulkan_transition_to_color_attachment(frame.command_buffer, image_index)

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

	vulkan_transition_to_present(frame.command_buffer, image_index)
}
vulkan_transition_to_color_attachment :: proc(command_buffer: vk.CommandBuffer, image_index: u32) {
	swapchain := &VULKAN_STATE.swapchain
	old_layout := swapchain.image_layouts[image_index]

	barrier := vk.ImageMemoryBarrier2 {
		sType = .IMAGE_MEMORY_BARRIER_2,
		srcStageMask = {},
		srcAccessMask = {},
		dstStageMask = {.COLOR_ATTACHMENT_OUTPUT},
		dstAccessMask = {.COLOR_ATTACHMENT_WRITE},
		oldLayout = old_layout,
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
vulkan_transition_to_present :: proc(command_buffer: vk.CommandBuffer, image_index: u32) {
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
