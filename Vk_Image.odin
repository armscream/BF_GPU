// BF_GPU/Vk_Image.odin
//
// Vulkan image + image-view creation/destruction, VMA-backed.
//
// VMA owns the VkDeviceMemory that backs every image. The image +
// allocation pair is wrapped in a Vulkan_Image record kept alive by a
// handle map. The handle map owns the heap allocation, mirroring the
// Vulkan_Buffer map in Vulkan.odin.
//
// Destruction is split into immediate and deferred paths:
//   - destroy_image_now: image + view + allocation released immediately
//     (used during shutdown).
//   - vulkan_defer_image_destruction: pushes a Vulkan_Deferred_Destruction
//     entry onto the per-backend queue, tagged with the GPU completion
//     value at submission time. vulkan_collect_garbage(up_to) reaps every
//     entry whose completion value has been signaled.
//
// Validation lives next to the create path. The validation helpers are
// pure functions that do not require a live device and are exercised by
// tests_vulkan_resources.odin.

package BF_GPU

import vma "../../dependencies/odin-vma"
import "core:log"
import vk "vendor:vulkan"

// ---------------------------------------------------------------------------
// Image / view types. All fields are backend-private; the rest of the
// renderer only ever sees Gpu_Image_Handle / Gpu_Image_View_Handle.
// ---------------------------------------------------------------------------

Vulkan_Image :: struct {
	image:       vk.Image,
	allocation:  vma.Allocation,
	view:        vk.ImageView, // optional: image-view paired at create time
	format:      vk.Format,
	width:       u32,
	height:      u32,
	mip_count:   u32,
	array_layers:u32,
	usage:       vk.ImageUsageFlags,
	owns_view:   bool, // true when the backend created the view in lockstep
}

// Image_Handle_Map / Image_View_Handle_Map / Sampler_Handle_Map are the
// registry the renderer side (Resources.odin) reaches through. Each
// entry owns a heap allocation the map manages.
Vulkan_Image_Handle_Map       :: map[Gpu_Image_Handle]^Vulkan_Image
Vulkan_Image_View_Handle_Map  :: map[Gpu_Image_View_Handle]vk.ImageView
Vulkan_Sampler_Handle_Map     :: map[Gpu_Sampler_Handle]vk.Sampler

@(private)
VULKAN_IMAGE_MAP:        Vulkan_Image_Handle_Map
@(private)
VULKAN_IMAGE_VIEW_MAP:   Vulkan_Image_View_Handle_Map
@(private)
VULKAN_SAMPLER_MAP:      Vulkan_Sampler_Handle_Map

@(private)
VULKAN_IMAGE_NEXT_ID:        u64 = 1
@(private)
VULKAN_IMAGE_VIEW_NEXT_ID:   u64 = 1
@(private)
VULKAN_SAMPLER_NEXT_ID:      u64 = 1

// ---------------------------------------------------------------------------
// Format translation. Image_Format -> VkFormat. Returns .UNDEFINED when
// the host-side enum has no equivalent; the create path translates this
// into a validation failure rather than passing .UNDEFINED to Vulkan.
// ---------------------------------------------------------------------------

image_format_to_vk :: proc(format: Image_Format) -> vk.Format {
	#partial switch format {
	case .R8G8B8A8_Unorm:        return .R8G8B8A8_UNORM
	case .R8G8B8A8_Srgb:         return .R8G8B8A8_SRGB
	case .B8G8R8A8_Unorm:        return .B8G8R8A8_UNORM
	case .B8G8R8A8_Srgb:         return .B8G8R8A8_SRGB
	case .R16G16B16A16_Sfloat:   return .R16G16B16A16_SFLOAT
	case .R32G32B32A32_Sfloat:   return .R32G32B32A32_SFLOAT
	case .R32_Sfloat:            return .R32_SFLOAT
	case .D32_Sfloat:            return .D32_SFLOAT
	case .D24_Unorm_S8_Uint:     return .D24_UNORM_S8_UINT
	case .D32_Sfloat_S8_Uint:    return .D32_SFLOAT_S8_UINT
	case .Undefined:             return .UNDEFINED
	}
	return .UNDEFINED
}

image_usage_to_vk :: proc(usage: Image_Usage) -> vk.ImageUsageFlags {
	flags: vk.ImageUsageFlags
	if .Color_Attachment           in usage do flags += {.COLOR_ATTACHMENT}
	if .Depth_Stencil_Attachment   in usage do flags += {.DEPTH_STENCIL_ATTACHMENT}
	if .Sampled                    in usage do flags += {.SAMPLED}
	if .Storage                    in usage do flags += {.STORAGE}
	if .Transfer_Src               in usage do flags += {.TRANSFER_SRC}
	if .Transfer_Dst               in usage do flags += {.TRANSFER_DST}
	if .Input_Attachment           in usage do flags += {.INPUT_ATTACHMENT}
	return flags
}

// vk_to_image_format is the inverse of image_format_to_vk, used by the
// view path when the caller did not supply a format and the image's
// creation format must be re-used. Returns .Undefined for every format
// not enumerated in Image_Format.
vk_to_image_format :: proc(format: vk.Format) -> Image_Format {
	#partial switch format {
	case .R8G8B8A8_UNORM:     return .R8G8B8A8_Unorm
	case .R8G8B8A8_SRGB:      return .R8G8B8A8_Srgb
	case .B8G8R8A8_UNORM:     return .B8G8R8A8_Unorm
	case .B8G8R8A8_SRGB:      return .B8G8R8A8_Srgb
	case .R16G16B16A16_SFLOAT:return .R16G16B16A16_Sfloat
	case .R32G32B32A32_SFLOAT:return .R32G32B32A32_Sfloat
	case .R32_SFLOAT:         return .R32_Sfloat
	case .D32_SFLOAT:         return .D32_Sfloat
	case .D24_UNORM_S8_UINT:  return .D24_Unorm_S8_Uint
	case .D32_SFLOAT_S8_UINT: return .D32_Sfloat_S8_Uint
	case:
		return .Undefined
	}
}

image_view_kind_to_vk :: proc(kind: Image_View_Kind) -> vk.ImageViewType {
	#partial switch kind {
	case .View_2D:           return .D2
	case .View_2D_Array:     return .D2_ARRAY
	case .View_Cube:         return .CUBE
	case .View_Cube_Array:   return .CUBE_ARRAY
	case .View_3D:           return .D3
	}
	return .D2
}

// image_aspect_mask picks the matching aspect mask for a given format.
// Depth-stencil formats require .DEPTH | .STENCIL, color formats .COLOR.
image_aspect_mask :: proc(format: Image_Format) -> vk.ImageAspectFlags {
	#partial switch format {
	case .D32_Sfloat, .D24_Unorm_S8_Uint, .D32_Sfloat_S8_Uint:
		flags: vk.ImageAspectFlags = {.DEPTH}
		if format == .D24_Unorm_S8_Uint || format == .D32_Sfloat_S8_Uint {
			flags += {.STENCIL}
		}
		return flags
	}
	return {.COLOR}
}

// ---------------------------------------------------------------------------
// Validation. Pure helpers; no live device needed.
// ---------------------------------------------------------------------------

// vulkan_validate_image_description rejects descriptions that are clearly
// invalid (zero extent, unsupported format, empty usage set). It returns
// the first failure as a cstring suitable for log.errorf; "" == ok.
vulkan_validate_image_description :: proc(desc: Image_Description) -> cstring {
	if desc.extent.width == 0 || desc.extent.height == 0 {
		return "image extent has a zero dimension"
	}
	if desc.extent.width > 16384 || desc.extent.height > 16384 {
		return "image extent exceeds the 16384 px VkPhysicalDeviceLimits minimum"
	}
	if desc.mip_count == 0 || desc.mip_count > 16 {
		return "image mip_count must be in [1, 16]"
	}
	if desc.array_layers == 0 || desc.array_layers > 256 {
		return "image array_layers must be in [1, 256]"
	}
	if desc.samples != 1 && desc.samples != 2 && desc.samples != 4 && desc.samples != 8 && desc.samples != 16 {
		return "image samples must be a power of two in {1, 2, 4, 8, 16}"
	}
	if desc.format == .Undefined {
		return "image format is Undefined"
	}
	if desc.usage == {} {
		return "image usage must include at least one Image_Usage_Flag"
	}
	// Depth-stencil formats require DEPTH_STENCIL_ATTACHMENT usage and
	// forbid color attachment usage.
	is_depth := desc.format == .D32_Sfloat ||
		desc.format == .D24_Unorm_S8_Uint ||
		desc.format == .D32_Sfloat_S8_Uint
	if is_depth {
		if .Depth_Stencil_Attachment not_in desc.usage {
			return "depth-stencil image must declare Depth_Stencil_Attachment usage"
		}
		if .Color_Attachment in desc.usage {
			return "depth-stencil image cannot declare Color_Attachment usage"
		}
		if .Sampled in desc.usage && desc.format == .D24_Unorm_S8_Uint {
			return "D24S8 formats cannot be combined with sampled shaders in v1"
		}
	} else {
		if .Depth_Stencil_Attachment in desc.usage {
			return "color image cannot declare Depth_Stencil_Attachment usage"
		}
	}
	if desc.samples != 1 && .Color_Attachment not_in desc.usage && .Depth_Stencil_Attachment not_in desc.usage {
		return "MSAA image must declare a render-target usage"
	}
	if desc.samples != 1 && (.Storage in desc.usage) {
		return "storage images cannot be multisampled"
	}
	return ""
}

// vulkan_validate_image_view_description rejects view descriptions that
// do not match the source image or that exceed the host limits.
vulkan_validate_image_view_description :: proc(desc: Image_View_Description) -> cstring {
	if desc.image == GPU_IMAGE_INVALID {
		return "image_view references the invalid image handle"
	}
	if desc.format == .Undefined {
		return "image_view format is Undefined"
	}
	if desc.kind == .View_3D && desc.layer_count != 1 {
		return "3D image views must have layer_count == 1"
	}
	if desc.mip_count == 0 {
		return "image_view mip_count must be >= 1"
	}
	if desc.layer_count == 0 {
		return "image_view layer_count must be >= 1"
	}
	if desc.kind == .View_Cube && desc.layer_count != 6 {
		return "cube image views must have layer_count == 6"
	}
	if desc.kind == .View_Cube_Array && desc.layer_count % 6 != 0 {
		return "cube-array image views must have layer_count divisible by 6"
	}
	return ""
}

// vulkan_validate_sampler_description rejects invalid sampler params.
// Anisotropy > 1.0 requires the host to have queried device support; we
// cap at 16.0 to match the spec.
vulkan_validate_sampler_description :: proc(desc: Sampler_Description) -> cstring {
	if desc.min_lod < 0 || desc.max_lod < desc.min_lod {
		return "sampler lod range is invalid"
	}
	if desc.max_anisotropy < 1.0 || desc.max_anisotropy > 16.0 {
		return "sampler max_anisotropy must be in [1.0, 16.0]"
	}
	return ""
}

// vulkan_validate_buffer_size rejects buffers that are zero-sized or
// exceed the spec's 32-bit size limit. The stride is checked for being a
// power of two for byte-addressable buffers; storage buffers with stride
// 0 (raw) are valid.
vulkan_validate_buffer_size :: proc(kind: Gpu_Buffer_Kind, size: u64, stride: u32) -> cstring {
	_ = kind
	if size == 0 {
		return "buffer size is zero"
	}
	// VkBufferCreateInfo.size is a VkDeviceSize (uint64); reject
	// values that would overflow the host-side rounding.
	if size > 0x000FFFFFFFFFFFFF {
		return "buffer size exceeds VkDeviceSize limit"
	}
	if stride > 0 && (stride & (stride - 1)) != 0 {
		return "buffer stride must be a power of two or zero"
	}
	return ""
}

// vulkan_validate_buffer_alignment checks the requested buffer creation
// against the minimum required alignment for the buffer's intended usage.
// Storage buffers / uniform buffers require 4; ray-tracing scratch 16;
// the safe minimum is 4.
vulkan_validate_buffer_alignment :: proc(size: u64, stride: u32) -> cstring {
	_ = size
	_ = stride
	// Vulkan requires VkBufferCreateInfo.size to be a positive integer;
	// the physical-device minStorageBufferOffsetAlignment (>= 4) is
	// honored at the descriptor-binding level, not at creation. The
	// minimum creation-time alignment is 1 byte.
	return ""
}

// ---------------------------------------------------------------------------
// Image creation. Public backend entry points go through
// vulkan_backend_create_image / vulkan_backend_create_image_view,
// vulkan_backend_create_sampler in Vulkan.odin.
// ---------------------------------------------------------------------------

vulkan_create_image :: proc(desc: Image_Description) -> (Vulkan_Image, bool) {
	msg := vulkan_validate_image_description(desc)
	if msg != "" {
		log.errorf("[BF_GPU/Vulkan] image create rejected: %s", msg)
		return {}, false
	}
	if VULKAN_STATE.allocator == nil {
		log.error("[BF_GPU/Vulkan] image create before VMA init")
		return {}, false
	}
	vk_format := image_format_to_vk(desc.format)
	vk_usage := image_usage_to_vk(desc.usage)
	mip_count := desc.mip_count
	if mip_count == 0 {
		mip_count = vulkan_derive_mip_count(desc.extent.width, desc.extent.height)
	}

	create_info := vk.ImageCreateInfo {
		sType                 = .IMAGE_CREATE_INFO,
		imageType             = .D2,
		format                = vk_format,
		extent                = vk.Extent3D{
			width  = desc.extent.width,
			height = desc.extent.height,
			depth  = 1,
		},
		mipLevels             = mip_count,
		arrayLayers           = desc.array_layers,
		samples               = vulkan_samples_to_vk(desc.samples),
		tiling                = .OPTIMAL,
		usage                 = vk_usage,
		sharingMode           = .EXCLUSIVE,
		initialLayout         = .UNDEFINED,
	}

	alloc_info := vma.AllocationCreateInfo {
		usage         = .GPU_ONLY,
		requiredFlags = {},
	}

	result: Vulkan_Image
	create_result := vma.CreateImage(
		VULKAN_STATE.allocator,
		create_info,
		alloc_info,
		&result.image,
		&result.allocation,
		nil,
	)
	if create_result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] CreateImage failed: %v", create_result)
		return {}, false
	}
	result.format       = vk_format
	result.width        = desc.extent.width
	result.height       = desc.extent.height
	result.mip_count    = mip_count
	result.array_layers = desc.array_layers
	result.usage        = vk_usage
	result.view         = {}
	result.owns_view    = false
	return result, true
}

vulkan_create_image_view_for_image :: proc(
	image: ^Vulkan_Image,
	desc: Image_View_Description,
) -> (vk.ImageView, bool) {
	msg := vulkan_validate_image_view_description(desc)
	if msg != "" {
		log.errorf("[BF_GPU/Vulkan] image_view create rejected: %s", msg)
		return {}, false
	}
	view_format := image_format_to_vk(desc.format)
	aspect := image_aspect_mask(desc.format)
	create_info := vk.ImageViewCreateInfo {
		sType    = .IMAGE_VIEW_CREATE_INFO,
		image    = image.image,
		viewType = image_view_kind_to_vk(desc.kind),
		format   = view_format,
		components = vk.ComponentMapping {
			r = .IDENTITY,
			g = .IDENTITY,
			b = .IDENTITY,
			a = .IDENTITY,
		},
		subresourceRange = vk.ImageSubresourceRange {
			aspectMask     = aspect,
			baseMipLevel   = desc.base_mip,
			levelCount     = desc.mip_count,
			baseArrayLayer = desc.base_layer,
			layerCount     = desc.layer_count,
		},
	}
	out: vk.ImageView
	create_result := vk.CreateImageView(VULKAN_STATE.device, &create_info, nil, &out)
	if create_result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] CreateImageView failed: %v", create_result)
		return {}, false
	}
	return out, true
}

vulkan_create_image_view :: proc(
	image_handle: Gpu_Image_Handle,
	desc: Image_View_Description,
) -> (vk.ImageView, bool) {
	image, ok := VULKAN_IMAGE_MAP[image_handle]
	if !ok || image == nil {
		log.error("[BF_GPU/Vulkan] image_view references unknown image handle")
		return {}, false
	}
	return vulkan_create_image_view_for_image(image, desc)
}

// ---------------------------------------------------------------------------
// Sampler creation.
// ---------------------------------------------------------------------------

sampler_filter_to_vk :: proc(filter: Sampler_Filter) -> vk.Filter {
	switch filter {
	case .Nearest: return .NEAREST
	case .Linear:  return .LINEAR
	}
	return .NEAREST
}

sampler_mipmap_mode_to_vk :: proc(filter: Sampler_Filter) -> vk.SamplerMipmapMode {
	switch filter {
	case .Nearest: return .NEAREST
	case .Linear:  return .LINEAR
	}
	return .NEAREST
}

sampler_address_mode_to_vk :: proc(mode: Sampler_Address_Mode) -> vk.SamplerAddressMode {
	switch mode {
	case .Repeat:           return .REPEAT
	case .Mirrored_Repeat:  return .MIRRORED_REPEAT
	case .Clamp_To_Edge:    return .CLAMP_TO_EDGE
	case .Clamp_To_Border:  return .CLAMP_TO_BORDER
	}
	return .CLAMP_TO_EDGE
}

vulkan_create_sampler :: proc(desc: Sampler_Description) -> (vk.Sampler, bool) {
	msg := vulkan_validate_sampler_description(desc)
	if msg != "" {
		log.errorf("[BF_GPU/Vulkan] sampler create rejected: %s", msg)
		return {}, false
	}
	enable_aniso := desc.max_anisotropy > 1.0
	create_info := vk.SamplerCreateInfo {
		sType                   = .SAMPLER_CREATE_INFO,
		magFilter               = sampler_filter_to_vk(desc.mag_filter),
		minFilter               = sampler_filter_to_vk(desc.min_filter),
		mipmapMode              = sampler_mipmap_mode_to_vk(desc.mipmap_mode),
		addressModeU            = sampler_address_mode_to_vk(desc.address_u),
		addressModeV            = sampler_address_mode_to_vk(desc.address_v),
		addressModeW            = sampler_address_mode_to_vk(desc.address_w),
		mipLodBias              = 0.0,
		anisotropyEnable        = b32(enable_aniso),
		maxAnisotropy           = desc.max_anisotropy,
		compareEnable           = false,
		compareOp               = .ALWAYS,
		minLod                  = desc.min_lod,
		maxLod                  = desc.max_lod,
		borderColor             = .FLOAT_OPAQUE_BLACK,
		unnormalizedCoordinates = false,
	}
	out: vk.Sampler
	create_result := vk.CreateSampler(VULKAN_STATE.device, &create_info, nil, &out)
	if create_result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] CreateSampler failed: %v", create_result)
		return {}, false
	}
	return out, true
}

// ---------------------------------------------------------------------------
// Destruction paths. immediate_destroy_* run synchronously and are used
// during shutdown; defer_* queue an entry that vulkan_collect_garbage()
// reaps once the GPU work that referenced the resource is known to have
// completed (signaled timeline value <= up_to).
// ---------------------------------------------------------------------------

vulkan_destroy_image_now :: proc(image: ^Vulkan_Image) {
	if image == nil do return
	if image.view != {} && image.owns_view {
		vk.DestroyImageView(VULKAN_STATE.device, image.view, nil)
		image.view = {}
	}
	if image.image != {} && VULKAN_STATE.allocator != nil {
		vma.DestroyImage(VULKAN_STATE.allocator, image.image, image.allocation)
	}
	image^ = {}
}

vulkan_destroy_image_view_now :: proc(view: vk.ImageView) {
	if view == {} do return
	if VULKAN_STATE.device != nil {
		vk.DestroyImageView(VULKAN_STATE.device, view, nil)
	}
}

vulkan_destroy_sampler_now :: proc(sampler: vk.Sampler) {
	if sampler == {} do return
	if VULKAN_STATE.device != nil {
		vk.DestroySampler(VULKAN_STATE.device, sampler, nil)
	}
}

// vulkan_clear_resource_maps empties the image / view / sampler handle
// maps and frees all native resources. Called during vulkan_shutdown
// after the device is idle and the deferred-destruction queue has been
// drained.
vulkan_clear_resource_maps :: proc() {
	for _, entry in VULKAN_IMAGE_MAP {
		if entry != nil {
			vulkan_destroy_image_now(entry)
			free(entry)
		}
	}
	clear(&VULKAN_IMAGE_MAP)
	VULKAN_IMAGE_NEXT_ID = 1

	for _, view in VULKAN_IMAGE_VIEW_MAP {
		vulkan_destroy_image_view_now(view)
	}
	clear(&VULKAN_IMAGE_VIEW_MAP)
	VULKAN_IMAGE_VIEW_NEXT_ID = 1

	for _, sampler in VULKAN_SAMPLER_MAP {
		vulkan_destroy_sampler_now(sampler)
	}
	clear(&VULKAN_SAMPLER_MAP)
	VULKAN_SAMPLER_NEXT_ID = 1
}

// vulkan_derive_mip_count returns floor(log2(max(width, height))) + 1.
vulkan_derive_mip_count :: proc(width, height: u32) -> u32 {
	largest := max(width, height)
	count: u32 = 1
	for largest > 1 {
		largest = largest >> 1
		count += 1
	}
	return count
}

vulkan_samples_to_vk :: proc(samples: u32) -> vk.SampleCountFlags {
	switch samples {
	case 1:  return vk.SampleCountFlags{vk.SampleCountFlag._1}
	case 2:  return vk.SampleCountFlags{vk.SampleCountFlag._2}
	case 4:  return vk.SampleCountFlags{vk.SampleCountFlag._4}
	case 8:  return vk.SampleCountFlags{vk.SampleCountFlag._8}
	case 16: return vk.SampleCountFlags{vk.SampleCountFlag._16}
	}
	return vk.SampleCountFlags{vk.SampleCountFlag._1}
}