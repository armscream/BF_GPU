package BF_GPU

import vma "../../dependencies/odin-vma"
import "core:log"
import sdl "vendor:sdl3"
import vk "vendor:vulkan"

//* EXT/CAPABILITIES listing.
BF_GPU_DEVICE_EXTENSION_COUNT :: 5

BF_GPU_DEVICE_EXTENSIONS: [BF_GPU_DEVICE_EXTENSION_COUNT]cstring = {
	"VK_EXT_graphics_pipeline_library",
	"VK_EXT_memory_priority",
	"VK_EXT_memory_budget",
	"VK_KHR_fragment_shading_rate",
	"VK_EXT_descriptor_buffer",
}

Vulkan_Capabilities :: struct {
	api_version:                                  u32,

	// Core / Vulkan 1.3
	dynamic_rendering:                            bool,
	synchronization2:                             bool,

	// Vulkan 1.2
	buffer_device_address:                        bool,
	descriptor_indexing:                          bool,
	runtime_descriptor_array:                     bool,
	descriptor_binding_partially_bound:           bool,
	descriptor_binding_variable_descriptor_count: bool,
	draw_indirect_count:                          bool,

	// Required BF_GPU extensions
	graphics_pipeline_library:                    bool,
	memory_priority:                              bool,
	memory_budget:                                bool,
	fragment_shading_rate:                        bool,
	descriptor_buffer:                            bool,

	// Extension features
	pipeline_fragment_shading_rate:               bool,
	primitive_fragment_shading_rate:              bool,
	attachment_fragment_shading_rate:             bool,
	descriptor_buffer_feature:                    bool,
	descriptor_buffer_push_descriptors:           bool,
	descriptor_buffer_capture_replay:             bool,
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

Vulkan_Frame :: struct {
	command_pool:    vk.CommandPool,
	command_buffer:  vk.CommandBuffer,
	//
	// WSI synchronization
	image_available: vk.Semaphore,
	render_finished: vk.Semaphore,
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
	instance:                vk.Instance,
	//
	physical_device:         vk.PhysicalDevice,
	device:                  vk.Device,
	//
	queues:                  Vulkan_Queue_Family,
	capabilities:            Vulkan_Capabilities,
	//
	graphics_queue:          vk.Queue,
	compute_queue:           vk.Queue,
	transfer_queue:          vk.Queue,
	present_queue:           vk.Queue,
	//
	surface:                 vk.SurfaceKHR,
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
	initialized:             bool,
}

MAX_FRAMES_IN_FLIGHT :: 2

@(private)
VULKAN_STATE: Vulkan_Context

//////////////////////////////////////////////////////////////////////////////////////
//* LIFECYCLE CODE
vulkan_init :: proc() -> bool {
	if !vulkan_create_instance() do return false
	if !vulkan_create_surface() do return false
	if !vulkan_pick_physical_device() do return false
	if !vulkan_create_device() do return false
	if !vulkan_create_allocator() do return false
	if !vulkan_create_command_resources() do return false
	if !vulkan_create_swapchain() do return false // TODO: create swapchain, and make it recreate after resize
	if !vulkan_create_sync_objects() do return false
	VULKAN_STATE.initialized = true
	log.info("[BF_GPU/Vulkan] Vulkan backend initialized")
	return true
}

vulkan_shutdown :: proc() {
	if !VULKAN_STATE.initialized && VULKAN_STATE.device == nil do return
	vk.DeviceWaitIdle(VULKAN_STATE.device)
	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		frame := &VULKAN_STATE.frames[i]
		if frame.image_available !=
		   nil {vk.DestroySemaphore(VULKAN_STATE.device, frame.image_available, nil)}
		if frame.render_finished !=
		   nil {vk.DestroySemaphore(VULKAN_STATE.device, frame.render_finished, nil)}
		if frame.fence != nil {vk.DestroyFence(VULKAN_STATE.device, frame.fence, nil)}
		if frame.command_pool !=
		   nil {vk.DestroyCommandPool(VULKAN_STATE.device, frame.command_pool, nil)}
	}
	vk.DestroyDevice(VULKAN_STATE.device, nil)
	if VULKAN_STATE.surface !=
	   nil {vk.DestroySurfaceKHR(VULKAN_STATE.instance, VULKAN_STATE.surface, nil)}
	if VULKAN_STATE.instance != nil {vk.DestroyInstance(VULKAN_STATE.instance, nil)}
	// Destroy VMA
	vma.DestroyAllocator(allocator)
	VULKAN_STATE = {}
}
/////////////////////////////////////////////////////////////////////////

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
	if !queues.has_compute do return false
	if !queues.has_transfer do return false

	caps, ok := vulkan_query_capabilities(device)
	if !ok do return false

	// if !vulkan_check_device_extensions(device) do return false
	VULKAN_STATE.queues = queues
	VULKAN_STATE.capabilities = caps
	return true
}

//* EXTENSIONS
vulkan_enumerate_device_extensions :: proc(device: vk.PhysicalDevice) -> ([dynamic]cstring, bool) {
	count: u32
	result := vk.EnumerateDeviceExtensionProperties(device, nil, &count, nil)
	if result != .SUCCESS || count == 0 do return nil, false
	properties := make([]vk.ExtensionProperties, count)
	defer delete(properties)

	result = vk.EnumerateDeviceExtensionProperties(device, nil, &count, raw_data(properties))
	if result != .SUCCESS do return nil, false

	names: [dynamic]cstring

	for property in properties {append(&names, property.extensionName)}
	return names, true
}
vulkan_has_device_extension :: proc(extensions: []cstring, name: cstring) -> bool {
	for extension in extensions {if extension == name do return true}
	return false
}
vulkan_check_required_extensions :: proc(device: vk.PhysicalDevice) -> bool {
	extensions, ok := vulkan_enumerate_device_extensions(device)
	if !ok {
		log.error("[BF_GPU/Vulkan] Failed to enumerate device extensions.")
		return false
	}
	defer delete(extensions)

	for required in BF_GPU_DEVICE_EXTENSIONS {
		if !vulkan_has_device_extension(extensions, required) {
			log.error("[BF_GPU/Vulkan] Required device extension '{}' is not available.", required)
			return false
		}
	}
	return true
}
////////////////////////////////////////////////////////

vulkan_find_queue_families :: proc(device: vk.PhysicalDevice) -> Vulkan_Queue_Family {
	result := Vulkan_Queue_Family{}
	count: u32

	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, nil)
	properties := make([]vk.QueueFamilyProperties, count)
	defer delete(properties)

	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, raw_data(properties))

	// First pass: prefer dedicated compute/transfer families.

	for i, props in properties {
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

	// Second pass: fill general-purpose queues.
	for i, props in properties {
		family := u32(i)
		if .GRAPHICS in props.queueFlags {
			if !result.has_graphics {
				result.graphics = family
				result.has_graphics = true
			}
		}
		if !result.has_compute && .COMPUTE in props.queueFlags {
			result.compute = family
			result.has_compute = true
		}
		if !result.has_transfer && .TRANSFER in props.queueFlags {
			result.transfer = family
			result.has_transfer = true
		}
		if sdl.Vulkan_GetPresentationSupport(VULKAN_STATE.instance, device, family) {
			if !result.has_present {
				result.present = family
				result.has_present = true
			}
		}
	}
	return result
}

//* Physical Device
vulkan_query_capabilities :: proc(device: vk.PhysicalDevice) -> (Vulkan_Capabilites, bool) {
	caps := Vulkan_Capabilites{}
	properties := vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(device, &properties)
	caps.api_version = properties.apiVersion

	if caps.api_version < vk.API_VERSION_1_3 {
		log.error("[BF_GPU/Vulkan] Vulkan 1.3 is required")
		return caps, false
	}
	if !vulkan_check_required_extensions(device) do return caps, false

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

	caps.dynamic_rendering = features13.dynamicRendering
	caps.synchronization2 = features13.synchronization2

	caps.buffer_device_address = features12.bufferDeviceAddress
	caps.descriptor_indexing = features12.descriptorIndexing
	caps.runtime_descriptor_array = features12.runtimeDescriptorArray
	caps.descriptor_binding_partially_bound = features12.descriptorBindingPartiallyBound
	caps.descriptor_binding_variable_descriptor_count =
		features12.descriptorBindingVariableDescriptorCount
	caps.drawIndirectCount = features12.drawIndirectCount

	graphics_pipeline_library_features := VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT {
		sType                   = .PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
		pNext                   = nil,
		graphicsPipelineLibrary = true,
	}

	memory_priority_features := VkPhysicalDeviceMemoryPriorityFeaturesEXT {
		sType          = .PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
		pNext          = nil,
		memoryPriority = true,
	}

	fragment_shading_rate_features := VkPhysicalDeviceFragmentShadingRateFeaturesKHR {
		sType                         = .PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
		pNext                         = nil,
		attachmentFragmentShadingRate = true,
		pipelineFragmentShadingRate   = true,
	}

	descriptor_buffer_features := VkPhysicalDeviceDescriptorBufferFeaturesEXT {
		sType                              = .PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
		pNext                              = nil,
		descriptorBuffer                   = true,
		descriptorBufferCaptureReplay      = true,
		descriptorBufferImageLayoutIgnored = true,
	}
	return caps, true
}
vulkan_create_device :: proc() -> bool {
	queues := VULKAN_STATE.queues
	priority: f32 = 1.0
	queue_infos: [dynamic]vk.DeviceQueueCreateInfo
	append_queue := proc(family: u32) {
		for info in queue_infos {if info.queueFamilyIndex == family do return}
		append(
			&queue_infos,
			vk.DeviceQueueCreateInfo {
				sType = .DEVICE_QUEUE_CREATE_INFO,
				queueFamilyIndex = family,
				queueCount = 1,
				pQueuePriorities = &priority,
			},
		)
	}

	append_queue(queues.graphics)
	append_queue(queues.compute)
	append_queue(queues.transfer)
	append_queue(queues.present)

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
		drawIndirectCount                        = true,
	}
	features12.pNext = &features13

	create_info := vk.DeviceCreateInfo {
		sType                   = .DEVICE_CREATE_INFO,
		queueCreateInfoCount    = u32(len(queue_infos)),
		pQueueCreateInfos       = raw_data(queue_infos),
		enabledExtensionCount   = u32(len(device_extensions)),
		ppEnabledExtensionNames = raw_data(device_extensions),
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
	log.infof(
		"[BF_GPU/Vulkan] Device queues: graphics=%d compute=%d transfer=%d present=%d",
		queues.graphics,
		queues.compute,
		queues.transfer,
		queues.present,
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
		physicalDevice  = VULKAN_STATE.physical_device,
		device           = VULKAN_STATE.device,
		pVulkanFunctions = &functions,
		vulkanApiVersion = vk.API_VERSION_1_3,
	}
	result := vma.CreateAllocator(create_info, &VULKAN_STATE.allocator)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] VMA allocator creation failed: %v", result)
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
	result := vk.CreateSemaphore(vULKAN_STATE.device, &create_info, nil, out)
	if result != .SUCCES {
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
