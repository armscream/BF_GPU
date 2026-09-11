package BF_GPU

import "core:log"
import "core:sync"
import vk "vendor:vulkan"

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

Vulkan_Frame :: struct {
	command_pool:    vk.CommandPool,
	command_buffer:  vk.CommandBuffer,
	//
	image_available: vk.Semaphore,
	render_finished: vk.Semaphore,
	fence:           vk.Fence,
}

//* SWAPCHAIN
Vulkan_Swapchain :: struct {
	handle:      vk.SwapchainKHR,
	format:      vk.Format,
	extent:      vk.Extent2D,
	images:      []vk.Image,
	image_views: []vk.ImageView,
	image_count: u32,
}

Vulkan_Context :: struct {
	instance:        vk.Instance,
	//
	physical_device: vk.PhysicalDevice,
	device:          vk.Device,
	//
	queues:          Vulkan_Queue_Family,
	//
	graphics_queue:  vk.Queue,
	compute_queue:   vk.Queue,
	transfer_queue:  vk.Queue,
	present_queue:   vk.Queue,
	//
	surface:         vk.SurfaceKHR,
	//
	frames:          [MAX_FRAMES_IN_FLIGHT]Vulkan_Frame,
	frame_index:     u32,
	//
	initialized:     bool,
}

MAX_FRAMES_IN_FLIGHT :: 2

@(private)
VULKAN_STATE: Vulkan_Context

vulkan_init :: proc() -> bool {
    if !vulkan_create_instance() do return false
    if !vulkan_create_surface() do return false
    if !vulkan_pick_physical_device() do return false
    if !vulkan_create_device() do return false
    if !vulkan_create_command_resources() do return false
    if !vulkan_create_swapchain() do return false
    if !vulkan_create_sync_objects() do return false

    return true
}

vulkan_create_instance :: proc() -> bool {
	app_info := vk.ApplicationInfo {
		sType              = .APPLICATION_INFO,
		pApplicationName   = "Bifrost",
		applicationVersion = vk.MAKE_VERSION(0, 0, 1),
		pEngineName        = "Bifrost Engine",
		engineVersion      = vk.MAKE_VERSION(0, 0, 1),
		apiVersion         = vk.API_VERSION_1_3,
	}

	extensions := window_vulkan_instance_extensions()

	create_info := vk.InstanceCreateInfo {
		sType                   = .INSTANCE_CREATE_INFO,
		pApplicationInfo        = &app_info,
		enabledExtensionCount   = u32(len(extensions)),
		ppEnabledExtensionNames = raw_data(extensions),
	}

	result := vk.CreateInstance(&create_info, nil, &VULKAN_STATE.instance)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkCreateInstance failed: %v", result)
		return false
	}

	vk.load_proc_addresses_instance(VULKAN_STATE.instance)
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

vulkan_device_is_suitable :: proc(device: vk.PhysicalDevice) -> bool {
	queues := vulkan_find_queue_families(device)
	if !queues.has_graphics do return false
	if !queues.has_present do return false

	if !vulkan_check_device_extensions(device) do return false
	return true
}

vulkan_find_queue_families :: proc(device: vk.PhysicalDevice) -> Vulkan_Queue_Family {
	result := Vulkan_Queue_Family{}
	count: u32

	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, nil)
	properties := make([]vk.QueueFamilyProperties, count)
	defer delete(properties)

	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, raw_data(properties))

	for i, props in properties {
		if .GRAPHICS in props.queueFlags {
			result.graphics = u32(i)
			result.has_graphics = true
		}
		if .COMPUTE in props.queueFlags {
			result.compute = u32(i)
			result.has_compute = true
		}
        if .TRANSFER in props.queueFlags {
			result.transfer = u32(i)
			result.has_transfer = true
		}

		present_supported: int
		vk.GetPhysicalDeviceSurfaceSupportKHR(
			device,
			u32(i),
			VULKAN_STATE.surface,
			&present_supported,
		)
		if present_supported != 0 {
			result.present = u32(i)
			result.has_present = true
		}
	}
	return result
}

//* Physical Device
vulkan_create_device :: proc() -> bool {
	queues := VULKAN_STATE.queues
	priority: f32 = 1.0
	queue_infos: [dynamic]vk.DeviceQueueCreateInfo

	append(
		&queue_infos,
		vk.DeviceQueueCreateInfo {
			sType = .DEVICE_QUEUE_CREATE_INFO,
			queueFamilyIndex = queues.graphics,
			queueCount = 1,
			pQueuePriorities = &priority,
		},
	)

	if queues.present != queues.graphics {
		append(
			&queue_infos,
			vk.DeviceQueueCreateInfo {
				sType = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = queues.present,
				queueCount = 1,
				pQueuePriorities = &priority,
			},
		)
	}

	features13 := vk.PhysicalDeviceVulkan13Features {
		sType            = .PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
		dynamicRendering = true,
		synchronization2 = true,
	}

	features12 := vk.PhysicalDeviceVulkan12Features {
		sType                                    = .PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
		bufferDeviceAddress                      = true,
		descriptorIndexing                       = true,
		runtimeDescriptorArray                   = true,
		descriptorBindingPartiallyBound          = true,
		descriptorBindingVariableDescriptorCount = true,
	}
	features12.pNext = &features13

	create_info := vk.DeviceCreateInfo {
		sType                = .DEVICE_CREATE_INFO,
		queueCreateInfoCount = u32(len(queue_infos)),
		pQueueCreateInfos    = raw_data(queue_infos),
		pNext                = &features12,
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
	vk.GetDeviceQueue(VULKAN_STATE.device, queues.present, 0, &VULKAN_STATE.present_queue)
	return true
}
