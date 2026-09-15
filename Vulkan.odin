package BF_GPU

import vma "../../dependencies/odin-vma"
import "core:log"
import "core:strings"
import sdl "vendor:sdl3"
import vk "vendor:vulkan"

//* EXT/CAPABILITIES listing.
BF_GPU_DEVICE_EXTENSION_COUNT :: 5
BF_GPU_VULKAN_API_VERSION :: vk.API_VERSION_1_3

BF_GPU_DEVICE_EXTENSIONS: [BF_GPU_DEVICE_EXTENSION_COUNT]cstring = {
	"VK_EXT_graphics_pipeline_library",
	"VK_EXT_memory_priority",
	"VK_EXT_memory_budget",
	"VK_KHR_fragment_shading_rate",
	"VK_EXT_descriptor_buffer",
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
	physical_device:         vk.PhysicalDevice,
	device:                  vk.Device,
	//
	queues:                  Vulkan_Queue_Family,
	//
	graphics_queue:          vk.Queue,
	compute_queue:           vk.Queue,
	transfer_queue:          vk.Queue,
	present_queue:           vk.Queue,
	//
	surface:                 vk.SurfaceKHR,
	//
	swapchain:               Vulkan_Swapchain,
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
//* LIFECYCLE CODE ------------------------------------------------------
vulkan_init :: proc() -> bool {
	if !vulkan_create_instance() do return false
	if !vulkan_create_surface() do return false
	if !vulkan_pick_physical_device() do return false
	if !vulkan_create_device() do return false
	if !vulkan_create_allocator() do return false
	if !vulkan_create_command_resources() do return false
	if !vulkan_create_sync_objects() do return false
	if !vulkan_create_swapchain() do return false
	if !vulkan_create_swapchain_image_views() do return false

	VULKAN_STATE.initialized = true
	log.info("[BF_GPU/Vulkan] Vulkan backend initialized")
	return true
}

vulkan_shutdown :: proc() {
	if !VULKAN_STATE.initialized && VULKAN_STATE.device == nil do return
	if VULKAN_STATE.device != nil {vk.DeviceWaitIdle(VULKAN_STATE.device)}

	vulkan_destroy_swapchain()

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		frame := &VULKAN_STATE.frames[i]
		if frame.image_available !=
		   cast(vk.Semaphore)0 {vk.DestroySemaphore(VULKAN_STATE.device, frame.image_available, nil)}
		if frame.render_finished !=
		   cast(vk.Semaphore)0 {vk.DestroySemaphore(VULKAN_STATE.device, frame.render_finished, nil)}
		if frame.command_pool !=
		   cast(vk.CommandPool)0 {vk.DestroyCommandPool(VULKAN_STATE.device, frame.command_pool, nil)}
	}
	if VULKAN_STATE.graphics_timeline != cast(vk.Semaphore)0 {
		vk.DestroySemaphore(VULKAN_STATE.device, VULKAN_STATE.graphics_timeline, nil)
	}
	if VULKAN_STATE.compute_timeline != cast(vk.Semaphore)0 {
		vk.DestroySemaphore(VULKAN_STATE.device, VULKAN_STATE.compute_timeline, nil)
	}
	if VULKAN_STATE.transfer_timeline != cast(vk.Semaphore)0 {
		vk.DestroySemaphore(VULKAN_STATE.device, VULKAN_STATE.transfer_timeline, nil)
	}
	if VULKAN_STATE.allocator != nil {
		vma.DestroyAllocator(VULKAN_STATE.allocator)
	}
	if VULKAN_STATE.device != nil {vk.DestroyDevice(VULKAN_STATE.device, nil)}
	if VULKAN_STATE.surface !=
	   cast(vk.SurfaceKHR)0 {vk.DestroySurfaceKHR(VULKAN_STATE.instance, VULKAN_STATE.surface, nil)}
	if VULKAN_STATE.instance != nil {vk.DestroyInstance(VULKAN_STATE.instance, nil)}

	VULKAN_STATE = {}
}
//* ---------------------------------------------------------------------

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
//* DEVICE QUERIES ========================================================
vulkan_device_is_suitable :: proc(device: vk.PhysicalDevice) -> bool {
	queues := vulkan_find_queue_families(device)
	if !queues.has_graphics do return false
	if !queues.has_present do return false
	if !queues.has_compute do return false
	if !queues.has_transfer do return false

	if !vulkan_check_required_extensions(device) do return false
	if !vulkan_check_required_features(device) do return false

	// if !vulkan_check_device_extensions(device) do return false
	VULKAN_STATE.queues = queues
	return true
}
vulkan_enumerate_device_extensions :: proc(device: vk.PhysicalDevice) -> ([dynamic]string, bool) {
	count: u32
	result := vk.EnumerateDeviceExtensionProperties(device, nil, &count, nil)
	if result != .SUCCESS || count == 0 do return nil, false
	properties := make([]vk.ExtensionProperties, count)
	defer delete(properties)

	result = vk.EnumerateDeviceExtensionProperties(device, nil, &count, raw_data(properties))
	if result != .SUCCESS do return nil, false

	names: [dynamic]string

	for i in 0 ..< len(properties) {
		cstr := cstring(raw_data(properties[i].extensionName[:]))
		append(&names, strings.clone(string(cstr)))
	}
	return names, true
}
vulkan_has_device_extension :: proc(device: vk.PhysicalDevice, required: cstring) -> bool {
	count: u32
	result := vk.EnumerateDeviceExtensionProperties(device, nil, &count, nil)
	if result != .SUCCESS || count == 0 do return false

	properties := make([]vk.ExtensionProperties, count)
	defer delete(properties)

	result = vk.EnumerateDeviceExtensionProperties(device, nil, &count, raw_data(properties))
	if result != .SUCCESS do return false

	for i in 0 ..< len(properties) {
		if extension_name_equal(properties[i].extensionName[:], required) do return true
	}
	return false
}
extension_name_equal :: proc(name: []byte, other: cstring) -> bool {
	a := name
	b := string(other)
	if len(b) > len(a) do return false
	for i in 0 ..< len(b) {
		if a[i] != b[i] do return false
	}
	return a[len(b)] == 0
}
vulkan_check_required_extensions :: proc(device: vk.PhysicalDevice) -> bool {
	for extension in BF_GPU_DEVICE_EXTENSIONS {
		if !vulkan_has_device_extension(device, extension) {
			log.errorf("[BF_GPU/Vulkan] Missing required device extension: %s", extension)
			return false
		}
	}
	return true
}
vulkan_check_required_features :: proc(device: vk.PhysicalDevice) -> bool {
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
	if !features13.dynamicRendering {
		log.error("[BF_GPU/Vulkan] Dynamic rendering is required but not supported")
		return false
	}
	if !features13.synchronization2 {
		log.error("[BF_GPU/Vulkan] Synchronization2 is required but not supported")
		return false
	}
	if !features12.bufferDeviceAddress {
		log.error("[BF_GPU/Vulkan] Buffer device address is required but not supported")
		return false
	}
	if !features12.descriptorIndexing {
		log.error("[BF_GPU/Vulkan] Descriptor indexing is required but not supported")
		return false
	}
	if !features12.runtimeDescriptorArray {
		log.error("[BF_GPU/Vulkan] Runtime descriptor array is required but not supported")
		return false
	}
	if !features12.descriptorBindingVariableDescriptorCount {
		log.error(
			"[BF_GPU/Vulkan] Descriptor binding variable descriptor count is required but not supported",
		)
		return false
	}
	if !features12.drawIndirectCount {
		log.error("[BF_GPU/Vulkan] Draw indirect count is required but not supported")
		return false
	}
	return true
}
//* =======================================================================

//* SWAPCHAIN =============================================================
vulkan_query_surface_capabilities :: proc(
	device: vk.PhysicalDevice,
) -> (
	vk.SurfaceCapabilitiesKHR,
	bool,
) {
	capabilities := vk.SurfaceCapabilitiesKHR{}
	result := vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(
		device,
		VULKAN_STATE.surface,
		&capabilities,
	)
	if result != .SUCCESS {
		log.error("[BF_GPU/Vulkan] vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: %v", result)
		return capabilities, false
	}
	return capabilities, true
}
vulkan_query_surface_formats :: proc(device: vk.PhysicalDevice) -> ([]vk.SurfaceFormatKHR, bool) {
	count: u32
	result := vk.GetPhysicalDeviceSurfaceFormatsKHR(device, VULKAN_STATE.surface, &count, nil)
	if result != .SUCCESS || count == 0 {
		log.error("[BF_GPU/Vulkan] no surface formats available")
		return nil, false
	}

	formats := make([]vk.SurfaceFormatKHR, count)
	result = vk.GetPhysicalDeviceSurfaceFormatsKHR(
		device,
		VULKAN_STATE.surface,
		&count,
		raw_data(formats),
	)

	if result != .SUCCESS {
		delete(formats)
		log.errorf("[BF_GPU/Vulkan] failed querying surface formats: %v", result)
		return nil, false
	}

	return formats, true
}
vulkan_query_present_modes :: proc(device: vk.PhysicalDevice) -> ([]vk.PresentModeKHR, bool) {
	count: u32
	result := vk.GetPhysicalDeviceSurfacePresentModesKHR(device, VULKAN_STATE.surface, &count, nil)
	if result != .SUCCESS || count == 0 {
		log.error("[BF_GPU/Vulkan] no present modes available")
		return nil, false
	}
	modes := make([]vk.PresentModeKHR, count)

	result = vk.GetPhysicalDeviceSurfacePresentModesKHR(
		device,
		VULKAN_STATE.surface,
		&count,
		raw_data(modes),
	)
	if result != .SUCCESS {
		delete(modes)
		log.errorf("[BF_GPU/Vulkan] failed querying present modes: %v", result)
		return nil, false
	}
	return modes, true
}
vulkan_choose_surface_format :: proc(
	formats: []vk.SurfaceFormatKHR,
) -> (
	vk.SurfaceFormatKHR,
	bool,
) {
	for format in formats {
		if format.format == .B8G8R8A8_SRGB &&
		   format.colorSpace == .SRGB_NONLINEAR {return format, true}
	}
	// TODO: make swapchain format configurable via renderer settings
	// Fall back to the first format supplied by the implementation.
	if len(formats) > 0 {return formats[0], true}

	return {}, false
}
vulkan_choose_present_mode :: proc(modes: []vk.PresentModeKHR) -> vk.PresentModeKHR {
	for mode in modes {
		if mode == .MAILBOX {return .MAILBOX}
	}
	// Mailbox if available, otherwise FIFO.
	return .FIFO
}
vulkan_choose_swapchain_extent :: proc(
	capabilities: vk.SurfaceCapabilitiesKHR,
	window: Window_Handle,
) -> vk.Extent2D {
	if capabilities.currentExtent.width != 0xFFFFFFFF {
		return capabilities.currentExtent}

	width := window.width
	height := window.height

	if width < capabilities.minImageExtent.width {
		width = capabilities.minImageExtent.width}
	if width > capabilities.maxImageExtent.width {
		width = capabilities.maxImageExtent.width}

	if height < capabilities.minImageExtent.height {
		height = capabilities.minImageExtent.height}
	if height > capabilities.maxImageExtent.height {
		height = capabilities.maxImageExtent.height}

	return vk.Extent2D{width = width, height = height}
}
vulkan_create_swapchain :: proc() -> bool {
	device := VULKAN_STATE.physical_device
	capabilities, ok := vulkan_query_surface_capabilities(device)
	if !ok do return false
	formats, ok2 := vulkan_query_surface_formats(device)
	if !ok2 do return false
	defer delete(formats)
	present_modes, ok3 := vulkan_query_present_modes(device)
	if !ok3 do return false
	defer delete(present_modes)
	surface_format, ok4 := vulkan_choose_surface_format(formats)
	if !ok4 {
		log.error("[BF_GPU/Vulkan] Failed to choose surface format")
		return false
	}
	present_mode := vulkan_choose_present_mode(present_modes)
	window := window_get_handle()
	extent := vulkan_choose_swapchain_extent(capabilities, window)
	image_count := capabilities.minImageCount + 1

	if capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount {
		image_count = capabilities.maxImageCount}

	queue_indices := [2]u32{VULKAN_STATE.queues.graphics, VULKAN_STATE.queues.present}

	create_info := vk.SwapchainCreateInfoKHR {
		sType            = .SWAPCHAIN_CREATE_INFO_KHR,
		surface          = VULKAN_STATE.surface,
		minImageCount    = image_count,
		imageFormat      = surface_format.format,
		imageColorSpace  = surface_format.colorSpace,
		imageExtent      = extent,
		imageArrayLayers = 1,
		imageUsage       = {.COLOR_ATTACHMENT},
		preTransform     = capabilities.currentTransform,
		compositeAlpha   = {.OPAQUE},
		presentMode      = present_mode,
		clipped          = true,
		oldSwapchain     = {},
	}

	if VULKAN_STATE.queues.graphics != VULKAN_STATE.queues.present {
		create_info.imageSharingMode = .CONCURRENT
		create_info.queueFamilyIndexCount = 2
		create_info.pQueueFamilyIndices = &queue_indices[0]
	} else {
		create_info.imageSharingMode = .EXCLUSIVE
	}

	result := vk.CreateSwapchainKHR(
		VULKAN_STATE.device,
		&create_info,
		nil,
		&VULKAN_STATE.swapchain.handle,
	)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkCreateSwapchainKHR failed: %v", result)
		return false
	}

	VULKAN_STATE.swapchain.format = surface_format.format
	VULKAN_STATE.swapchain.extent = extent

	result = vk.GetSwapchainImagesKHR(
		VULKAN_STATE.device,
		VULKAN_STATE.swapchain.handle,
		&image_count,
		nil,
	)
	if result != .SUCCESS || image_count == 0 {
		log.errorf("[BF_GPU/Vulkan] vkGetSwapchainImagesKHR failed: %v", result)
		vulkan_destroy_swapchain()
		return false
	}
	VULKAN_STATE.swapchain.images = make([]vk.Image, image_count)
	result = vk.GetSwapchainImagesKHR(
		VULKAN_STATE.device,
		VULKAN_STATE.swapchain.handle,
		&image_count,
		raw_data(VULKAN_STATE.swapchain.images),
	)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkGetSwapchainImagesKHR failed: %v", result)
		delete(VULKAN_STATE.swapchain.images)
		VULKAN_STATE.swapchain.images = nil
		vk.DestroySwapchainKHR(VULKAN_STATE.device, VULKAN_STATE.swapchain.handle, nil)
		VULKAN_STATE.swapchain.handle = {}
		return false
	}

	VULKAN_STATE.swapchain.image_count = image_count
	log.infof(
		"[BF_GPU/Vulkan] Swapchain created: %d%d, images=%d",
		extent.width,
		extent.height,
		image_count
	)

	return true
}
vulkan_create_swapchain_image_views :: proc() -> bool {
	swapchain := &VULKAN_STATE.swapchain
	swapchain.image_views = make([]vk.ImageView, swapchain.image_count)

	for i in 0 ..< swapchain.image_count {
		create_info := vk.ImageViewCreateInfo {
			sType = .IMAGE_VIEW_CREATE_INFO,
			image = swapchain.images[i],
			viewType = .D2,
			format = swapchain.format,
			components = vk.ComponentMapping {
				r = .IDENTITY,
				g = .IDENTITY,
				b = .IDENTITY,
				a = .IDENTITY,
			},
			subresourceRange = vk.ImageSubresourceRange {
				aspectMask = {.COLOR},
				baseMipLevel = 0,
				levelCount = 1,
				baseArrayLayer = 0,
				layerCount = 1,
			},
		}
		result := vk.CreateImageView(
			VULKAN_STATE.device,
			&create_info,
			nil,
			&swapchain.image_views[i],
		)
		if result != .SUCCESS {
			log.errorf("[BF_GPU/Vulkan] vkCreateImageView for swapchain image %d: %v", i, result)
			for j in 0..< i {
				vk.DestroyImageView(VULKAN_STATE.device, swapchain.image_views[j], nil)
			}
			delete(swapchain.image_views)
			swapchain.image_views = nil
			return false
		}
	}
	return true
}
vulkan_destroy_swapchain :: proc() {
	swapchain := &VULKAN_STATE.swapchain
	for view in swapchain.image_views {
		if view != {} {
			vk.DestroyImageView(VULKAN_STATE.device, view, nil)
		}
	}
	delete(swapchain.image_views)
	swapchain.image_views = nil

	if swapchain.handle != {} {
		vk.DestroySwapchainKHR(VULKAN_STATE.device, swapchain.handle, nil)
		swapchain.handle = {}
	}
	swapchain^ = {}
}
//* SWAPCHAIN RUNTIME
vulkan_acquire_next_image :: proc(frame: ^Vulkan_Frame) -> (u32, vk.Result) {
	image_index: u32
	result := vk.AcquireNextImageKHR(
		VULKAN_STATE.device,
		VULKAN_STATE.swapchain.handle,
		0xFFFFFFFFFFFFFFFF,
		frame.image_available,
		{},
		&image_index,
	)
	return image_index, result
}
//* =====================================================================

vulkan_find_queue_families :: proc(device: vk.PhysicalDevice) -> Vulkan_Queue_Family {
	result := Vulkan_Queue_Family{}
	count: u32

	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, nil)
	properties := make([]vk.QueueFamilyProperties, count)
	defer delete(properties)

	vk.GetPhysicalDeviceQueueFamilyProperties(device, &count, raw_data(properties))

	// First pass: prefer dedicated compute/transfer families.

	for props, i in properties {
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
	for props, i in properties {
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
vulkan_create_device :: proc() -> bool {
	queues := VULKAN_STATE.queues
	priority: f32 = 1.0
	queue_infos: [dynamic]vk.DeviceQueueCreateInfo

	vulkan_append_unique_queue_family(&queue_infos, queues.graphics, &priority)
	vulkan_append_unique_queue_family(&queue_infos, queues.compute, &priority)
	vulkan_append_unique_queue_family(&queue_infos, queues.transfer, &priority)
	vulkan_append_unique_queue_family(&queue_infos, queues.present, &priority)

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
		enabledExtensionCount   = u32(len(BF_GPU_DEVICE_EXTENSIONS)),
		ppEnabledExtensionNames = &BF_GPU_DEVICE_EXTENSIONS[0],
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
		physicalDevice   = VULKAN_STATE.physical_device,
		device           = VULKAN_STATE.device,
		pVulkanFunctions = &functions,
		vulkanApiVersion = BF_GPU_VULKAN_API_VERSION,
	}
	result := vma.CreateAllocator(create_info, &VULKAN_STATE.allocator)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] VMA allocator creation failed: %v", result)
		return false
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
	result := vk.CreateSemaphore(VULKAN_STATE.device, &create_info, nil, out)
	if result != .SUCCESS {
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
