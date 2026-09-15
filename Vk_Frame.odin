package BF_GPU

import "core:log"
import vk "vendor:vulkan"
/*
GPU_Frame_Graph :: struct {
    passed: [dynamic]GPU_Pass,
}
*/

vulkan_submit_graphics :: proc(frame: ^Vulkan_Frame) -> bool {
	wait_stage := [1]vk.PipelineStageFlags{{.COLOR_ATTACHMENT_OUTPUT}}
	wait_semaphores := [1]vk.Semaphore{frame.image_available}
	signal_semaphores := [1]vk.Semaphore{frame.render_finished}
	command_buffers := [1]vk.CommandBuffer{frame.command_buffer}
	submit_info := vk.SubmitInfo {
		sType                = .SUBMIT_INFO,
		waitSemaphoreCount   = 1,
		pWaitSemaphores      = &wait_semaphores[0],
		pWaitDstStageMask    = &wait_stage[0],
		commandBufferCount   = 1,
		pCommandBuffers      = &command_buffers[0],
		signalSemaphoreCount = 1,
		pSignalSemaphores    = &signal_semaphores[0],
	}
	result := vk.QueueSubmit(VULKAN_STATE.graphics_queue, 1, &submit_info, {})
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkQueueSubmit failed: %v", result)
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
	window_clear_resize()

	log.infof(
		"[BF_GPU/Vulkan] Swapchain recreated: %dx%d",
		VULKAN_STATE.swapchain.extent.width,
		VULKAN_STATE.swapchain.extent.height,
	)
	return true
}

vulkan_frame :: proc() -> bool {
	if !VULKAN_STATE.initialized do return false
	if window_resize_pending() do return vulkan_recreate_swapchain()

	frame := &VULKAN_STATE.frames[VULKAN_STATE.frame_index]
	image_index, acquire_result := vulkan_acquire_next_image(frame)
	if acquire_result == .ERROR_OUT_OF_DATE_KHR do return vulkan_recreate_swapchain()

	if acquire_result != .SUCCESS && acquire_result != .SUBOPTIMAL_KHR {
		log.errorf("[BF_GPU/Vulkan] vkAckuireNextImageKHR failed: %v", acquire_result)
		return false
	}
	if !vulkan_begin_command_buffer(frame.command_buffer) do return false

	//** Rendering will be inserted HERE **//
	///

	if !vulkan_end_command_buffer(frame.command_buffer) do return false
	if !vulkan_submit_graphics(frame) do return false

	present_result := vulkan_present(frame, image_index)
	if present_result == .ERROR_OUT_OF_DATE_KHR ||
	   present_result == .SUBOPTIMAL_KHR {return vulkan_recreate_swapchain()}

	if present_result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkQueuePresentKHR failed: %v", present_result)
		return false
	}
    VULKAN_STATE.frame_index = (VULKAN_STATE.frame_index + 1) % MAX_FRAMES_IN_FLIGHT
    //TODO: there is currently no CPU-side frame completion wait.
    return true
}
