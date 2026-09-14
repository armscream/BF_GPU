package BF_GPU

import vma "../../dependencies/odin-vma"
import "core:log"
import vk "vendor:vulkan"

Vulkan_Buffer :: struct {
	buffer:      vk.Buffer,
	allocation:  vma.Allocation,
	size:        vk.DeviceSize,
	defice_addr: vk.DeviceAddress,
}
Vulkan_Buffer_Store :: struct {
	buffers: [Gpu_Buffer_Kind]Vulkan_Buffer,
}
Gpu_Buffer_Description :: struct {
	initial_capacity:  u64,
	stride:            u32,
	usage:             vk.BufferUsageFlags,
	memory_properties: vk.MemoryPropertyFlags,
}

vulkan_create_buffer :: proc(
	size: vk.DeviceSize,
	usage: vk.BufferUsageFlags,
	memory_properties: vk.MemoryPropertyFlags,
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
	if vk.CreateBuffer(VULKAN_STATE.device, &create_info, nil, &result.buffer) !=
	   .SUCCESS {return result, false}

	requirements := vk.MemoryRequirements{}
	vk.GetBufferMemoryRequirements(VULKAN_STATE.device, result.buffer, &requirements)
	memory_type := vulkan_find_memory_type(requirements.memoryTypeBits, memory_properties)
	allocate_info := vk.MemoryAllocateInfo {
		sType           = .MEMORY_ALLOCATE_INFO,
		allocationSize  = requirements.size,
		memoryTypeIndex = memory_type,
	}

	if vk.AllocateMemory(VULKAN_STATE.device, &allocate_info, nil, &result.memory) != .SUCCESS {
		vk.DestroyBuffer(VULKAN_STATE.device, result.buffer, nil)
		return result, false
	}

	if vk.BindBufferMemory(VULKAN_STATE.device, result.buffer, result.memory, 0) !=
	   .SUCCESS {return result, false}

	result.size = size
	result.usage = usage

	return result, true
}
vulkan_upload_buffer :: proc(dst: ^Vulkan_Buffer, data: rawptr, size: u64) -> bool

vulkan_create_renderer_buffers :: proc(
	frame: ^Frame_Context_State,
	store: ^Vulkan_Buffer_Store,
) -> bool {
	for kind in Gpu_Buffer_Kind {if kind == .COUNT {break}
		desc := gpu_buffer_descriptions(kind)

		buffer, ok := vulkan_create_buffer(desc.size, desc.usage, desc.memory_properties)
		if !ok {
			log.errorf("[BF_GPU/Vulkan] failed creating buffer &v", kind)
			return false
		}

		store.buffers[kind] = buffer
		frame.buffers[kind] = Gpu_Buffer_Entry {
			handle      = Gpu_Buffer_Handle{kind + 1},
			device_addr = buffer.device_address,
			size        = desc.size,
			capacity    = desc.capacity,
			stride      = desc.stride,
		}
	}
	refresh_frame_addresses(frame)
	return true
}
