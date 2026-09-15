package BF_GPU

import vma "../../dependencies/odin-vma"
import "core:log"
import vk "vendor:vulkan"

Vulkan_Buffer :: struct {
	buffer:         vk.Buffer,
	allocation:     vma.Allocation,
	size:           vk.DeviceSize,
	device_address: vk.DeviceAddress,
}
Vulkan_Buffer_Store :: struct {
	buffers: [Gpu_Buffer_Kind]Vulkan_Buffer,
}
Gpu_Buffer_Description :: struct {
	size:         vk.DeviceSize,
	capacity:     u64,
	stride:       u32,
	usage:        vk.BufferUsageFlags,
	memory_usage: vma.MemoryUsage,
}

gpu_buffer_descriptions :: proc(kind: Gpu_Buffer_Kind) -> Gpu_Buffer_Description {
	_ = kind
	desc := Gpu_Buffer_Description {
		capacity     = 1024,
		stride       = 16,
		usage        = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
		memory_usage = .GPU_ONLY,
	}
	desc.size = vk.DeviceSize(desc.capacity * u64(desc.stride))
	return desc
}

vulkan_create_buffer :: proc(
	size: vk.DeviceSize,
	usage: vk.BufferUsageFlags,
	memory_usage: vma.MemoryUsage,
) -> (
	Vulkan_Buffer,
	bool,
) {
	result := Vulkan_Buffer{}
	create_info := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = size,
		usage       = usage,
		sharingMode = .EXCLUSIVE,
	}
	allocation_info := vma.AllocationCreateInfo {usage = memory_usage}
	allocation_info.requiredFlags = {}

	vk_result := vma.CreateBuffer(VULKAN_STATE.allocator, create_info, allocation_info, &result.buffer, &result.allocation, nil)
	if vk_result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] Failed to create buffer! %v", vk_result)
		return result, false
	}

	result.size = size

	address_info := vk.BufferDeviceAddressInfo {
		sType = .BUFFER_DEVICE_ADDRESS_INFO,
		buffer = result.buffer,
	}
	result.device_address = vk.GetBufferDeviceAddress(VULKAN_STATE.device, &address_info)
	return result, true
}
vulkan_upload_buffer :: proc(dst: ^Vulkan_Buffer, data: rawptr, size: u64) -> bool

vulkan_create_renderer_buffers :: proc(
	frame: ^Frame_Context_State,
	store: ^Vulkan_Buffer_Store,
) -> bool {
	for kind in Gpu_Buffer_Kind {if kind == .COUNT {break}
		desc := gpu_buffer_descriptions(kind)

		buffer, ok := vulkan_create_buffer(desc.size, desc.usage, desc.memory_usage)
		if !ok {
			log.errorf("[BF_GPU/Vulkan] failed creating buffer &v", kind)
			return false
		}

		store.buffers[kind] = buffer
		frame.buffers[kind] = Gpu_Buffer_Entry {
			handle      = Gpu_Buffer_Handle(u64(kind) + 1),
			device_addr = u64(buffer.device_address),
			size        = u64(desc.size),
			capacity    = desc.capacity,
			stride      = desc.stride,
		}
	}
	refresh_frame_addresses(frame)
	return true
}

//* RUNTIME ==============================================
vulkan_begin_command_buffer :: proc(cmd_buffer: vk.CommandBuffer) -> bool {
	result := vk.ResetCommandBuffer(cmd_buffer, {})
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkResetCommandBuffer failed: %v", result)
		return false
	}
	begin_info := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
	}
	result = vk.BeginCommandBuffer(cmd_buffer, &begin_info)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkBeginCommandBuffer failed: %v", result)
		return false
	}
	return true
}
vulkan_end_command_buffer :: proc(cmd_buffer: vk.CommandBuffer) -> bool {
	result := vk.EndCommandBuffer(cmd_buffer)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkEndCommandBuffer failed: %v", result)
		return false
	}
	return true
}