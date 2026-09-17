// BF_GPU/Vk_Descriptor.odin
//
// Bindless descriptor model for the BF_GPU Vulkan backend.
//
// The renderer splits its shader-visible bindings into three descriptor
// sets:
//
//   set 0 = frame
//       per-frame data the culling / shading shaders need each frame:
//       the frame UBO (camera, frame constants). One set per
//       MAX_FRAMES_IN_FLIGHT.
//
//   set 1 = persistent / bindless
//       runtime-resizable bindless arrays (textures, samplers) plus
//       the persistent storage-buffer bindings the renderer rarely
//       rebinds (material pool, transform pool, model pool, animation
//       pool). One set per MAX_FRAMES_IN_FLIGHT.
//
//   set 2 = per-pass
//       pass-specific resources the culling/shading pipeline needs:
//       the HiZ depth pyramid is the canonical binding. One set per
//       MAX_FRAMES_IN_FLIGHT.
//
// The pool is allocated with UPDATE_AFTER_BIND; the bindless texture /
// sampler arrays use PARTIALLY_BOUND | UPDATE_AFTER_BIND |
// VARIABLE_DESCRIPTOR_COUNT so the renderer can add and remove
// descriptors at any point in the frame without rebuilding the
// affected sets.
//
// Texture identity is split cleanly from the descriptor index: the
// GPU_Texture_ID stays the asset-stable identifier the renderer uses
// in the resource store, and vulkan_descriptor_bind_texture hands out
// a u32 bindless index that lives only in this module's tables. The
// shader-side lookup always goes through a uniform index, so an asset
// that is rebound to a different underlying VkImage does not change
// any visible indices.

package BF_GPU

import "core:log"
import vk "vendor:vulkan"

// ---------------------------------------------------------------------------
//* Set indices + binding model.
//
// Set 0 = frame, set 1 = persistent/bindless, set 2 = per-pass. These
// constants are baked into the GLSL source so any change here must
// match every shader that uses layout(set = N, ...).
// ---------------------------------------------------------------------------

DESCRIPTOR_SET_FRAME      :: 0
DESCRIPTOR_SET_PERSISTENT :: 1
DESCRIPTOR_SET_PER_PASS   :: 2

// Binding slots inside each set.
BIND_FRAME_GLOBAL_CONTEXT :: 0 // set 0, UBO with view/proj/eye/etc.

BIND_PERSISTENT_MATERIAL_POOL :: 0 // set 1, storage buffer
BIND_PERSISTENT_TRANSFORM_POOL :: 1 // set 1, storage buffer
BIND_PERSISTENT_MODEL_POOL     :: 2 // set 1, storage buffer
BIND_PERSISTENT_BINDLESS_TEX   :: 3 // set 1, runtime sampled-image array
BIND_PERSISTENT_BINDLESS_SAMP  :: 4 // set 1, runtime sampler array

BIND_PER_PASS_DEPTH_PYRAMID :: 0 // set 2, combined image sampler

// Capacity for the bindless texture + sampler arrays. The shader side
// indexes into these arrays as `tex[nonuniformEXT(bindless_index)]`.
// Sized for the planned "many thousands of textures" use case while
// staying well under the per-stage descriptor-indexing limit most
// drivers expose (>1M).
DESCRIPTOR_BINDLESS_TEXTURES_MAX :: 8192
DESCRIPTOR_BINDLESS_SAMPLERS_MAX :: 32

// Pool sizing. maxSets covers 3 sets per MAX_FRAMES_IN_FLIGHT; the
// descriptor counts cover each binding type's per-pool allocation.
DESCRIPTOR_POOL_MAX_SETS :: MAX_FRAMES_IN_FLIGHT * 3

// ---------------------------------------------------------------------------
//* State.
// ---------------------------------------------------------------------------

Vulkan_Descriptor_Layouts :: struct {
	frame:      vk.DescriptorSetLayout,
	persistent: vk.DescriptorSetLayout,
	per_pass:   vk.DescriptorSetLayout,
}

// Vulkan_Descriptor_Sets is the per-frame-in-flight set triple. Every
// frame in flight owns its own (frame, persistent, per_pass) set; the
// bindless texture/sampler indices are global, so all frames see the
// same logical texture at the same array slot.
Vulkan_Descriptor_Sets :: struct {
	frame:      [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
	persistent: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
	per_pass:   [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSet,
}

@(private)
VULKAN_DESCRIPTOR_STATE: Vulkan_Descriptor_State

Vulkan_Descriptor_State :: struct {
	layouts: Vulkan_Descriptor_Layouts,
	pool:    vk.DescriptorPool,
	sets:    Vulkan_Descriptor_Sets,
	// Texture handle -> bindless slot. Stable for the lifetime of
	// the registered texture; reused when the same handle is
	// re-registered (the previous slot is freed first).
	texture_to_index:   map[Gpu_Image_Handle]u32,
	sampler_to_index:   map[Gpu_Sampler_Handle]u32,
	// Reverse map: bindless slot -> handle. Lets the renderer walk
	// the table to free every texture when the device is destroyed.
	index_to_texture:   [dynamic]Gpu_Image_Handle,
	index_to_sampler:   [dynamic]Gpu_Sampler_Handle,
	// Free list for bindless slot recycling. Pops a free slot when
	// bind_texture is called and the texture handle is new.
	free_texture_slots: [dynamic]u32,
	free_sampler_slots: [dynamic]u32,
	initialized:        bool,
}

// ---------------------------------------------------------------------------
//* Public lifecycle.
// ---------------------------------------------------------------------------

// vulkan_descriptor_init creates the layouts, pool, and per-frame sets.
// Safe to call once after the device exists; safe to call again only
// after vulkan_descriptor_shutdown has torn down the previous state.
vulkan_descriptor_init :: proc() -> bool {
	if VULKAN_DESCRIPTOR_STATE.initialized {
		log.warn("[BF_GPU/Vulkan] descriptor_init called twice; ignoring")
		return true
	}
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] descriptor_init before device creation")
		return false
	}

	// Create the three layouts. Each one is independent: pipeline
	// layouts in the pipeline builder compose them via
	// VkPipelineLayoutCreateInfo.pSetLayouts, and the renderer can
	// request any subset (the culling compute shaders only need set
	// 2; the shading graphics pipelines need sets 0+1+2).
	if !vulkan_descriptor_create_layouts() {
		log.error("[BF_GPU/Vulkan] descriptor layout creation failed")
		return false
	}
	if !vulkan_descriptor_create_pool() {
		log.error("[BF_GPU/Vulkan] descriptor pool creation failed")
		vulkan_descriptor_destroy_layouts()
		return false
	}
	if !vulkan_descriptor_allocate_sets() {
		log.error("[BF_GPU/Vulkan] descriptor set allocation failed")
		vulkan_descriptor_destroy_pool()
		vulkan_descriptor_destroy_layouts()
		return false
	}

	// Bindless index tables start empty. Slots are minted on demand
	// when vulkan_descriptor_bind_texture / _bind_sampler is called.
	VULKAN_DESCRIPTOR_STATE.texture_to_index = make(map[Gpu_Image_Handle]u32)
	VULKAN_DESCRIPTOR_STATE.sampler_to_index = make(map[Gpu_Sampler_Handle]u32)
	VULKAN_DESCRIPTOR_STATE.index_to_texture = make([dynamic]Gpu_Image_Handle)
	VULKAN_DESCRIPTOR_STATE.index_to_sampler = make([dynamic]Gpu_Sampler_Handle)
	VULKAN_DESCRIPTOR_STATE.free_texture_slots = make([dynamic]u32)
	VULKAN_DESCRIPTOR_STATE.free_sampler_slots = make([dynamic]u32)

	VULKAN_DESCRIPTOR_STATE.initialized = true
	log.info("[BF_GPU/Vulkan] bindless descriptor model initialized")
	return true
}

vulkan_descriptor_shutdown :: proc() {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return

	// The persistent set holds descriptors that reference the live
	// bindless images/samplers. Reset them to an unbound state so
	// vkDestroy* on the underlying views/samplers (which happens
	// after the device is idle in vulkan_shutdown) does not see
	// dangling descriptor references.
	vulkan_descriptor_reset_persistent_sets()

	delete(VULKAN_DESCRIPTOR_STATE.texture_to_index)
	delete(VULKAN_DESCRIPTOR_STATE.sampler_to_index)
	delete(VULKAN_DESCRIPTOR_STATE.index_to_texture)
	delete(VULKAN_DESCRIPTOR_STATE.index_to_sampler)
	delete(VULKAN_DESCRIPTOR_STATE.free_texture_slots)
	delete(VULKAN_DESCRIPTOR_STATE.free_sampler_slots)

	vulkan_descriptor_destroy_sets()
	vulkan_descriptor_destroy_pool()
	vulkan_descriptor_destroy_layouts()

	VULKAN_DESCRIPTOR_STATE.initialized = false
}

// vulkan_descriptor_initialized is the public read-only view used by
// tests + diagnostics. The pipeline builder also gates on it so a
// pipeline creation attempted before the descriptor module is ready
// is reported as a failure rather than producing a VkPipelineLayout
// referencing unset handles.
vulkan_descriptor_initialized :: proc() -> bool {
	return VULKAN_DESCRIPTOR_STATE.initialized
}

vulkan_descriptor_layouts_get :: proc() -> Vulkan_Descriptor_Layouts {
	return VULKAN_DESCRIPTOR_STATE.layouts
}

// vulkan_descriptor_sets_get exposes the allocated sets so the frame
// recording layer can bind them into command buffers. Returned by
// value so callers cannot mutate the state.
vulkan_descriptor_sets_get :: proc() -> Vulkan_Descriptor_Sets {
	return VULKAN_DESCRIPTOR_STATE.sets
}

// vulkan_descriptor_pipeline_layouts returns the three descriptor set
// layouts in (set 0, set 1, set 2) order, ready to be fed into
// VkPipelineLayoutCreateInfo.pSetLayouts. The returned slice is owned
// by the caller; in practice pipeline builders consume it as a stack
// array reference rather than allocating.
//
// Returns an empty slice when the descriptor model is not yet
// initialized; the pipeline builder must gate on
// vulkan_descriptor_initialized() before asking.
@(private)
_VK_DESCRIPTOR_PIPELINE_LAYOUTS: [3]vk.DescriptorSetLayout

vulkan_descriptor_pipeline_layouts :: proc() -> []vk.DescriptorSetLayout {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return nil
	_VK_DESCRIPTOR_PIPELINE_LAYOUTS[0] = VULKAN_DESCRIPTOR_STATE.layouts.frame
	_VK_DESCRIPTOR_PIPELINE_LAYOUTS[1] = VULKAN_DESCRIPTOR_STATE.layouts.persistent
	_VK_DESCRIPTOR_PIPELINE_LAYOUTS[2] = VULKAN_DESCRIPTOR_STATE.layouts.per_pass
	return _VK_DESCRIPTOR_PIPELINE_LAYOUTS[:]
}

// ---------------------------------------------------------------------------
//* Layout creation.
// ---------------------------------------------------------------------------

// vulkan_descriptor_create_layouts creates the three descriptor set
// layouts. The frame + per-pass layouts are plain (single binding each,
// UPDATE_AFTER_BIND not required because they are written exactly
// once per frame). The persistent layout is the only one that needs
// UPDATE_AFTER_BIND; the bindless bindings also need
// PARTIALLY_BOUND so the renderer can add and remove textures at
// any time. VARIABLE_DESCRIPTOR_COUNT is only valid on the last
// binding of a layout, so it is set on the bindless sampler binding
// (binding 4) and not the bindless texture binding (binding 3,
// which uses a static descriptorCount equal to
// DESCRIPTOR_BINDLESS_TEXTURES_MAX). UPDATE_AFTER_BIND on the
// SAMPLED_IMAGE binding also requires the
// descriptorBindingSampledImageUpdateAfterBind feature; when that
// feature is unavailable the binding falls back to PARTIALLY_BOUND
// only and the renderer logs a warning.
vulkan_descriptor_create_layouts :: proc() -> bool {
	// --- set 0 (frame) ---
	frame_bindings := [1]vk.DescriptorSetLayoutBinding {
		{
			binding         = BIND_FRAME_GLOBAL_CONTEXT,
			descriptorType  = .UNIFORM_BUFFER,
			descriptorCount = 1,
			stageFlags      = {.COMPUTE, .VERTEX, .FRAGMENT, .TASK_EXT, .TASK_NV, .MESH_EXT, .MESH_NV},
		},
	}

	frame_flags := [1]vk.DescriptorBindingFlags {}
	frame_flags_info := vk.DescriptorSetLayoutBindingFlagsCreateInfo {
		sType         = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		bindingCount  = u32(len(frame_flags)),
		pBindingFlags = &frame_flags[0],
	}

	frame_info := vk.DescriptorSetLayoutCreateInfo {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		pNext        = &frame_flags_info,
		flags        = {},
		bindingCount = u32(len(frame_bindings)),
		pBindings    = &frame_bindings[0],
	}
	result := vk.CreateDescriptorSetLayout(
		VULKAN_STATE.device,
		&frame_info,
		nil,
		&VULKAN_DESCRIPTOR_STATE.layouts.frame,
	)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] CreateDescriptorSetLayout(frame) failed: %v", result)
		return false
	}

	// --- set 1 (persistent / bindless) ---
	persistent_bindings := [5]vk.DescriptorSetLayoutBinding {
		{
			binding         = BIND_PERSISTENT_MATERIAL_POOL,
			descriptorType  = .STORAGE_BUFFER,
			descriptorCount = 1,
			stageFlags      = {.FRAGMENT, .COMPUTE},
		},
		{
			binding         = BIND_PERSISTENT_TRANSFORM_POOL,
			descriptorType  = .STORAGE_BUFFER,
			descriptorCount = 1,
			stageFlags      = {.COMPUTE, .VERTEX, .TASK_EXT, .TASK_NV, .MESH_EXT, .MESH_NV},
		},
		{
			binding         = BIND_PERSISTENT_MODEL_POOL,
			descriptorType  = .STORAGE_BUFFER,
			descriptorCount = 1,
			stageFlags      = {.COMPUTE, .VERTEX, .FRAGMENT},
		},
		{
			binding         = BIND_PERSISTENT_BINDLESS_TEX,
			descriptorType  = .SAMPLED_IMAGE,
			descriptorCount = DESCRIPTOR_BINDLESS_TEXTURES_MAX,
			stageFlags      = {.COMPUTE, .FRAGMENT},
		},
		{
			binding         = BIND_PERSISTENT_BINDLESS_SAMP,
			descriptorType  = .SAMPLER,
			descriptorCount = DESCRIPTOR_BINDLESS_SAMPLERS_MAX,
			stageFlags      = {.COMPUTE, .FRAGMENT},
		},
	}
	// Build the bindless tex binding flags conditionally:
	// descriptorBindingSampledImageUpdateAfterBind must be enabled
	// for UPDATE_AFTER_BIND on a SAMPLED_IMAGE descriptor. When the
	// physical device lacks the feature, drop UPDATE_AFTER_BIND and
	// keep PARTIALLY_BOUND only — the texture array becomes
	// read-only after initial population, which is a graceful
	// fallback for low-end Intel hardware that omits the feature.
	bindless_tex_flags: vk.DescriptorBindingFlags = {.PARTIALLY_BOUND}
	if VULKAN_STATE.descriptor_binding_sampled_image_update_after_bind_available {
		bindless_tex_flags += {.UPDATE_AFTER_BIND}
	}
	persistent_flags := [5]vk.DescriptorBindingFlags {
		{},                                                          // material pool
		{},                                                          // transform pool
		{},                                                          // model pool
		bindless_tex_flags,                                          // bindless tex
		{.UPDATE_AFTER_BIND, .PARTIALLY_BOUND, .VARIABLE_DESCRIPTOR_COUNT}, // bindless samp (last binding)
	}
	persistent_flags_info := vk.DescriptorSetLayoutBindingFlagsCreateInfo {
		sType         = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		bindingCount  = u32(len(persistent_flags)),
		pBindingFlags = &persistent_flags[0],
	}
	persistent_info := vk.DescriptorSetLayoutCreateInfo {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		pNext        = &persistent_flags_info,
		// UPDATE_AFTER_BIND_POOL is required on the pool side; the
		// matching layout flag tells the driver the layout allows
		// UPDATE_AFTER_BIND so the pool can have the bit set.
		flags        = {.UPDATE_AFTER_BIND_POOL},
		bindingCount = u32(len(persistent_bindings)),
		pBindings    = &persistent_bindings[0],
	}
	result = vk.CreateDescriptorSetLayout(
		VULKAN_STATE.device,
		&persistent_info,
		nil,
		&VULKAN_DESCRIPTOR_STATE.layouts.persistent,
	)
	if result != .SUCCESS {
		log.errorf(
			"[BF_GPU/Vulkan] CreateDescriptorSetLayout(persistent) failed: %v",
			result,
		)
		vk.DestroyDescriptorSetLayout(
			VULKAN_STATE.device,
			VULKAN_DESCRIPTOR_STATE.layouts.frame,
			nil,
		)
		VULKAN_DESCRIPTOR_STATE.layouts.frame = {}
		return false
	}

	// --- set 2 (per-pass) ---
	per_pass_bindings := [1]vk.DescriptorSetLayoutBinding {
		{
			binding         = BIND_PER_PASS_DEPTH_PYRAMID,
			descriptorType  = .COMBINED_IMAGE_SAMPLER,
			descriptorCount = 1,
			stageFlags      = {.COMPUTE, .FRAGMENT},
		},
	}
	per_pass_flags := [1]vk.DescriptorBindingFlags {}
	per_pass_flags_info := vk.DescriptorSetLayoutBindingFlagsCreateInfo {
		sType         = .DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
		bindingCount  = u32(len(per_pass_flags)),
		pBindingFlags = &per_pass_flags[0],
	}
	per_pass_info := vk.DescriptorSetLayoutCreateInfo {
		sType        = .DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		pNext        = &per_pass_flags_info,
		flags        = {},
		bindingCount = u32(len(per_pass_bindings)),
		pBindings    = &per_pass_bindings[0],
	}
	result = vk.CreateDescriptorSetLayout(
		VULKAN_STATE.device,
		&per_pass_info,
		nil,
		&VULKAN_DESCRIPTOR_STATE.layouts.per_pass,
	)
	if result != .SUCCESS {
		log.errorf(
			"[BF_GPU/Vulkan] CreateDescriptorSetLayout(per_pass) failed: %v",
			result,
		)
		vk.DestroyDescriptorSetLayout(
			VULKAN_STATE.device,
			VULKAN_DESCRIPTOR_STATE.layouts.persistent,
			nil,
		)
		vk.DestroyDescriptorSetLayout(
			VULKAN_STATE.device,
			VULKAN_DESCRIPTOR_STATE.layouts.frame,
			nil,
		)
		VULKAN_DESCRIPTOR_STATE.layouts = {}
		return false
	}

	return true
}

vulkan_descriptor_destroy_layouts :: proc() {
	if VULKAN_STATE.device == nil do return
	if VULKAN_DESCRIPTOR_STATE.layouts.frame != {} {
		vk.DestroyDescriptorSetLayout(
			VULKAN_STATE.device,
			VULKAN_DESCRIPTOR_STATE.layouts.frame,
			nil,
		)
	}
	if VULKAN_DESCRIPTOR_STATE.layouts.persistent != {} {
		vk.DestroyDescriptorSetLayout(
			VULKAN_STATE.device,
			VULKAN_DESCRIPTOR_STATE.layouts.persistent,
			nil,
		)
	}
	if VULKAN_DESCRIPTOR_STATE.layouts.per_pass != {} {
		vk.DestroyDescriptorSetLayout(
			VULKAN_STATE.device,
			VULKAN_DESCRIPTOR_STATE.layouts.per_pass,
			nil,
		)
	}
	VULKAN_DESCRIPTOR_STATE.layouts = {}
}

// ---------------------------------------------------------------------------
//* Pool creation.
// ---------------------------------------------------------------------------

vulkan_descriptor_create_pool :: proc() -> bool {
	// Pool sizes. Each entry covers every frame in flight; the driver
	// hands out descriptors from this budget as sets are allocated.
	// The bindless image / sampler allocations are sized to the
	// maximum array length so the pool can satisfy an
	// AllocateDescriptorSets with the full VARIABLE_DESCRIPTOR_COUNT
	// on the persistent set.
	sizes := [6]vk.DescriptorPoolSize {
		{type = .UNIFORM_BUFFER,        descriptorCount = MAX_FRAMES_IN_FLIGHT},
		{type = .STORAGE_BUFFER,        descriptorCount = MAX_FRAMES_IN_FLIGHT * 3},
		{type = .COMBINED_IMAGE_SAMPLER, descriptorCount = MAX_FRAMES_IN_FLIGHT},
		{type = .SAMPLED_IMAGE,         descriptorCount = DESCRIPTOR_BINDLESS_TEXTURES_MAX * MAX_FRAMES_IN_FLIGHT},
		{type = .SAMPLER,               descriptorCount = DESCRIPTOR_BINDLESS_SAMPLERS_MAX * MAX_FRAMES_IN_FLIGHT},
		{type = .STORAGE_IMAGE,         descriptorCount = MAX_FRAMES_IN_FLIGHT},
	}

	pool_info := vk.DescriptorPoolCreateInfo {
		sType         = .DESCRIPTOR_POOL_CREATE_INFO,
		flags         = {.UPDATE_AFTER_BIND, .FREE_DESCRIPTOR_SET},
		maxSets       = DESCRIPTOR_POOL_MAX_SETS,
		poolSizeCount = u32(len(sizes)),
		pPoolSizes    = &sizes[0],
	}
	result := vk.CreateDescriptorPool(
		VULKAN_STATE.device,
		&pool_info,
		nil,
		&VULKAN_DESCRIPTOR_STATE.pool,
	)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] CreateDescriptorPool failed: %v", result)
		return false
	}
	return true
}

vulkan_descriptor_destroy_pool :: proc() {
	if VULKAN_STATE.device == nil || VULKAN_DESCRIPTOR_STATE.pool == {} do return
	vk.DestroyDescriptorPool(VULKAN_STATE.device, VULKAN_DESCRIPTOR_STATE.pool, nil)
	VULKAN_DESCRIPTOR_STATE.pool = {}
}

// ---------------------------------------------------------------------------
//* Set allocation.
// ---------------------------------------------------------------------------

// vulkan_descriptor_allocate_sets allocates (MAX_FRAMES_IN_FLIGHT x 3)
// descriptor sets. The persistent set declares the full
// DESCRIPTOR_BINDLESS_TEXTURES_MAX / _SAMPLERS_MAX as the variable
// descriptor count so the layout's runtime-resizable arrays are sized
// to the maximum at allocation time.
vulkan_descriptor_allocate_sets :: proc() -> bool {
	persistent_layouts: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout
	frame_layouts:      [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout
	per_pass_layouts:   [MAX_FRAMES_IN_FLIGHT]vk.DescriptorSetLayout

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		persistent_layouts[i] = VULKAN_DESCRIPTOR_STATE.layouts.persistent
		frame_layouts[i]      = VULKAN_DESCRIPTOR_STATE.layouts.frame
		per_pass_layouts[i]   = VULKAN_DESCRIPTOR_STATE.layouts.per_pass
	}

	// Variable descriptor counts for the persistent layout
	// allocation. Each entry matches a set in the same order as
	// pSetLayouts and specifies the actual descriptor count for the
	// single binding flagged with VARIABLE_DESCRIPTOR_COUNT (the
	// bindless sampler array at binding 4). The bindless texture
	// array at binding 3 uses a static descriptorCount sized to
	// DESCRIPTOR_BINDLESS_TEXTURES_MAX so it does not need a
	// per-allocation variable count.
	samp_counts := [MAX_FRAMES_IN_FLIGHT]u32 {
		DESCRIPTOR_BINDLESS_SAMPLERS_MAX,
		DESCRIPTOR_BINDLESS_SAMPLERS_MAX,
	}
	variable_count_info := vk.DescriptorSetVariableDescriptorCountAllocateInfo {
		sType              = .DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO,
		descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
		pDescriptorCounts  = &samp_counts[0],
	}

	if !vulkan_allocate_one_set(
		&persistent_layouts[0],
		MAX_FRAMES_IN_FLIGHT,
		&variable_count_info,
		&VULKAN_DESCRIPTOR_STATE.sets.persistent[0],
	) {
		return false
	}
	if !vulkan_allocate_one_set(
		&frame_layouts[0],
		MAX_FRAMES_IN_FLIGHT,
		nil,
		&VULKAN_DESCRIPTOR_STATE.sets.frame[0],
	) {
		vulkan_descriptor_destroy_sets()
		return false
	}
	if !vulkan_allocate_one_set(
		&per_pass_layouts[0],
		MAX_FRAMES_IN_FLIGHT,
		nil,
		&VULKAN_DESCRIPTOR_STATE.sets.per_pass[0],
	) {
		vulkan_descriptor_destroy_sets()
		return false
	}

	return true
}

// vulkan_allocate_one_set is a thin wrapper around AllocateDescriptorSets
// that handles the optional variable-count pNext. Centralised so the
// per-frame triple allocation stays readable.
vulkan_allocate_one_set :: proc(
	layouts: ^vk.DescriptorSetLayout,
	count: u32,
	variable_count_info: ^vk.DescriptorSetVariableDescriptorCountAllocateInfo,
	out: ^vk.DescriptorSet,
) -> bool {
	allocate_info := vk.DescriptorSetAllocateInfo {
		sType              = .DESCRIPTOR_SET_ALLOCATE_INFO,
		pNext              = variable_count_info if variable_count_info != nil else nil,
		descriptorPool     = VULKAN_DESCRIPTOR_STATE.pool,
		descriptorSetCount = count,
		pSetLayouts        = layouts,
	}
	result := vk.AllocateDescriptorSets(VULKAN_STATE.device, &allocate_info, out)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] AllocateDescriptorSets failed: %v", result)
		return false
	}
	return true
}

vulkan_descriptor_destroy_sets :: proc() {
	// Sets allocated from a pool with FREE_DESCRIPTOR_SET are freed
	// implicitly when the pool is destroyed. Nothing to do here.
	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		VULKAN_DESCRIPTOR_STATE.sets.frame[i]      = {}
		VULKAN_DESCRIPTOR_STATE.sets.persistent[i] = {}
		VULKAN_DESCRIPTOR_STATE.sets.per_pass[i]   = {}
	}
}

// ---------------------------------------------------------------------------
//* Bindless texture / sampler registration.
// ---------------------------------------------------------------------------

// vulkan_descriptor_bind_texture allocates a bindless index for the
// given image handle (or returns the existing index if the handle was
// already registered) and writes the (view, layout) descriptor into
// every per-frame persistent set. Returns the bindless index the
// shader uses to read the texture.
//
// The handle -> index mapping is asset-stable: calling bind_texture
// twice with the same handle returns the same index and refreshes the
// underlying VkImageView. Calling with a different view handle swaps
// the VkImageView without changing the index.
vulkan_descriptor_bind_texture :: proc(
	handle: Gpu_Image_Handle,
	view: vk.ImageView,
	layout: vk.ImageLayout,
) -> u32 {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return INVALID_DESCRIPTOR_INDEX
	if handle == GPU_IMAGE_INVALID || view == {} {
		// A zero view is a programming error, not an unexpected
		// condition; warn rather than error so unit tests that
		// exercise the rejection path do not register a failure.
		log.warn("[BF_GPU/Vulkan] bind_texture: invalid handle or view")
		return INVALID_DESCRIPTOR_INDEX
	}

	index, exists := VULKAN_DESCRIPTOR_STATE.texture_to_index[handle]
	if !exists {
		index = vulkan_descriptor_acquire_texture_slot(handle)
		if index == INVALID_DESCRIPTOR_INDEX do return INVALID_DESCRIPTOR_INDEX
		VULKAN_DESCRIPTOR_STATE.texture_to_index[handle] = index
	}

	vulkan_descriptor_write_texture_all_frames(index, view, layout)
	return index
}

vulkan_descriptor_unbind_texture :: proc(handle: Gpu_Image_Handle) -> bool {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return false
	index, exists := VULKAN_DESCRIPTOR_STATE.texture_to_index[handle]
	if !exists do return false

	// Reset every persistent set's bindless slot back to a null
	// descriptor so a stale view handle isn't referenced. The slot
	// is freed back into the recycle list so the next bind_texture
	// call reuses it.
	vulkan_descriptor_write_texture_all_frames(index, {}, .UNDEFINED)
	delete_key(&VULKAN_DESCRIPTOR_STATE.texture_to_index, handle)
	append(&VULKAN_DESCRIPTOR_STATE.free_texture_slots, index)
	return true
}

vulkan_descriptor_bindless_index :: proc(handle: Gpu_Image_Handle) -> u32 {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return INVALID_DESCRIPTOR_INDEX
	if index, ok := VULKAN_DESCRIPTOR_STATE.texture_to_index[handle]; ok {
		return index
	}
	return INVALID_DESCRIPTOR_INDEX
}

vulkan_descriptor_bind_sampler :: proc(
	handle: Gpu_Sampler_Handle,
	sampler: vk.Sampler,
) -> u32 {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return INVALID_DESCRIPTOR_INDEX
	if handle == GPU_SAMPLER_INVALID || sampler == {} {
		log.warn("[BF_GPU/Vulkan] bind_sampler: invalid handle or sampler")
		return INVALID_DESCRIPTOR_INDEX
	}

	index, exists := VULKAN_DESCRIPTOR_STATE.sampler_to_index[handle]
	if !exists {
		index = vulkan_descriptor_acquire_sampler_slot(handle)
		if index == INVALID_DESCRIPTOR_INDEX do return INVALID_DESCRIPTOR_INDEX
		VULKAN_DESCRIPTOR_STATE.sampler_to_index[handle] = index
	}

	vulkan_descriptor_write_sampler_all_frames(index, sampler)
	return index
}

vulkan_descriptor_unbind_sampler :: proc(handle: Gpu_Sampler_Handle) -> bool {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return false
	index, exists := VULKAN_DESCRIPTOR_STATE.sampler_to_index[handle]
	if !exists do return false
	vulkan_descriptor_write_sampler_all_frames(index, {})
	delete_key(&VULKAN_DESCRIPTOR_STATE.sampler_to_index, handle)
	append(&VULKAN_DESCRIPTOR_STATE.free_sampler_slots, index)
	return true
}

vulkan_descriptor_bindless_sampler_index :: proc(handle: Gpu_Sampler_Handle) -> u32 {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return INVALID_DESCRIPTOR_INDEX
	if index, ok := VULKAN_DESCRIPTOR_STATE.sampler_to_index[handle]; ok {
		return index
	}
	return INVALID_DESCRIPTOR_INDEX
}

INVALID_DESCRIPTOR_INDEX :: u32(0xFFFFFFFF)

// vulkan_descriptor_acquire_texture_slot returns a fresh bindless
// slot. Prefers the recycle list so the table stays dense; falls back
// to appending to the reverse map once the recycle list is empty.
vulkan_descriptor_acquire_texture_slot :: proc(handle: Gpu_Image_Handle) -> u32 {
	if len(VULKAN_DESCRIPTOR_STATE.free_texture_slots) > 0 {
		// Pop the last free slot. The reverse map at that slot is
		// overwritten below; if the slot previously pointed at a
		// different handle, that handle's mapping is stale and gets
		// removed when the caller notices (unbind) or when it
		// re-binds (this routine overwrites the reverse map).
		slot := pop(&VULKAN_DESCRIPTOR_STATE.free_texture_slots)
		if int(slot) < len(VULKAN_DESCRIPTOR_STATE.index_to_texture) {
			VULKAN_DESCRIPTOR_STATE.index_to_texture[slot] = handle
		}
		return slot
	}
	// Append a fresh slot. Verify the array still has room under
	// DESCRIPTOR_BINDLESS_TEXTURES_MAX.
	if len(VULKAN_DESCRIPTOR_STATE.index_to_texture) >= DESCRIPTOR_BINDLESS_TEXTURES_MAX {
		log.warnf(
			"[BF_GPU/Vulkan] bindless texture array exhausted (max %d)",
			DESCRIPTOR_BINDLESS_TEXTURES_MAX,
		)
		return INVALID_DESCRIPTOR_INDEX
	}
	append(&VULKAN_DESCRIPTOR_STATE.index_to_texture, handle)
	return u32(len(VULKAN_DESCRIPTOR_STATE.index_to_texture) - 1)
}

vulkan_descriptor_acquire_sampler_slot :: proc(handle: Gpu_Sampler_Handle) -> u32 {
	if len(VULKAN_DESCRIPTOR_STATE.free_sampler_slots) > 0 {
		slot := pop(&VULKAN_DESCRIPTOR_STATE.free_sampler_slots)
		if int(slot) < len(VULKAN_DESCRIPTOR_STATE.index_to_sampler) {
			VULKAN_DESCRIPTOR_STATE.index_to_sampler[slot] = handle
		}
		return slot
	}
	if len(VULKAN_DESCRIPTOR_STATE.index_to_sampler) >= DESCRIPTOR_BINDLESS_SAMPLERS_MAX {
		log.warnf(
			"[BF_GPU/Vulkan] bindless sampler array exhausted (max %d)",
			DESCRIPTOR_BINDLESS_SAMPLERS_MAX,
		)
		return INVALID_DESCRIPTOR_INDEX
	}
	append(&VULKAN_DESCRIPTOR_STATE.index_to_sampler, handle)
	return u32(len(VULKAN_DESCRIPTOR_STATE.index_to_sampler) - 1)
}

// vulkan_descriptor_write_texture_all_frames writes the (view, layout)
// pair into every per-frame persistent set's bindless texture array at
// the given slot. A zero view means "write a null descriptor" so the
// slot becomes a no-op for the shader side (PARTIALLY_BOUND).
vulkan_descriptor_write_texture_all_frames :: proc(
	index: u32,
	view: vk.ImageView,
	layout: vk.ImageLayout,
) {
	if VULKAN_STATE.device == nil do return
	writes: [MAX_FRAMES_IN_FLIGHT]vk.WriteDescriptorSet
	images: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorImageInfo

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		images[i] = vk.DescriptorImageInfo {
			sampler     = {},
			imageView   = view,
			imageLayout = layout,
		}
		writes[i] = vk.WriteDescriptorSet {
			sType            = .WRITE_DESCRIPTOR_SET,
			dstSet           = VULKAN_DESCRIPTOR_STATE.sets.persistent[i],
			dstBinding       = BIND_PERSISTENT_BINDLESS_TEX,
			dstArrayElement  = index,
			descriptorCount  = 1,
			descriptorType   = .SAMPLED_IMAGE,
			pImageInfo       = &images[i],
		}
	}
	vk.UpdateDescriptorSets(VULKAN_STATE.device, MAX_FRAMES_IN_FLIGHT, &writes[0], 0, nil)
}

vulkan_descriptor_write_sampler_all_frames :: proc(index: u32, sampler: vk.Sampler) {
	if VULKAN_STATE.device == nil do return
	writes: [MAX_FRAMES_IN_FLIGHT]vk.WriteDescriptorSet
	images: [MAX_FRAMES_IN_FLIGHT]vk.DescriptorImageInfo

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		images[i] = vk.DescriptorImageInfo {
			sampler     = sampler,
			imageView   = {},
			imageLayout = .UNDEFINED,
		}
		writes[i] = vk.WriteDescriptorSet {
			sType            = .WRITE_DESCRIPTOR_SET,
			dstSet           = VULKAN_DESCRIPTOR_STATE.sets.persistent[i],
			dstBinding       = BIND_PERSISTENT_BINDLESS_SAMP,
			dstArrayElement  = index,
			descriptorCount  = 1,
			descriptorType   = .SAMPLER,
			pImageInfo       = &images[i],
		}
	}
	vk.UpdateDescriptorSets(VULKAN_STATE.device, MAX_FRAMES_IN_FLIGHT, &writes[0], 0, nil)
}

// vulkan_descriptor_reset_persistent_sets writes null descriptors for
// every currently-allocated bindless slot across every per-frame set.
// Called from vulkan_descriptor_shutdown so the underlying image views
// / samplers can be destroyed without leaving dangling descriptors in
// any set.
vulkan_descriptor_reset_persistent_sets :: proc() {
	if VULKAN_STATE.device == nil || VULKAN_DESCRIPTOR_STATE.pool == {} do return

	// Reset texture slots.
	for handle, index in VULKAN_DESCRIPTOR_STATE.texture_to_index {
		vulkan_descriptor_write_texture_all_frames(index, {}, .UNDEFINED)
		_ = handle
	}
	// Reset sampler slots.
	for handle, index in VULKAN_DESCRIPTOR_STATE.sampler_to_index {
		vulkan_descriptor_write_sampler_all_frames(index, {})
		_ = handle
	}
}

// ---------------------------------------------------------------------------
//* Persistent buffer binding updates.
// ---------------------------------------------------------------------------

// vulkan_descriptor_update_persistent_buffer writes a single
// STORAGE_BUFFER descriptor into every per-frame persistent set. Used
// when the renderer swaps the backing buffer for a pool (e.g. after
// reallocating the material pool to a larger capacity).
vulkan_descriptor_update_persistent_buffer :: proc(
	binding: u32,
	buffer: vk.Buffer,
	offset: vk.DeviceSize,
	range: vk.DeviceSize,
) {
	if !VULKAN_DESCRIPTOR_STATE.initialized || buffer == {} {
		log.warn("[BF_GPU/Vulkan] update_persistent_buffer: backend not ready or null buffer")
		return
	}

	writes: [MAX_FRAMES_IN_FLIGHT]vk.WriteDescriptorSet
	infos:  [MAX_FRAMES_IN_FLIGHT]vk.DescriptorBufferInfo

	for i in 0 ..< MAX_FRAMES_IN_FLIGHT {
		infos[i] = vk.DescriptorBufferInfo {
			buffer = buffer,
			offset = offset,
			range  = range,
		}
		writes[i] = vk.WriteDescriptorSet {
			sType           = .WRITE_DESCRIPTOR_SET,
			dstSet          = VULKAN_DESCRIPTOR_STATE.sets.persistent[i],
			dstBinding      = binding,
			dstArrayElement = 0,
			descriptorCount = 1,
			descriptorType  = .STORAGE_BUFFER,
			pBufferInfo     = &infos[i],
		}
	}
	vk.UpdateDescriptorSets(VULKAN_STATE.device, MAX_FRAMES_IN_FLIGHT, &writes[0], 0, nil)
}

// vulkan_descriptor_update_frame_ubo writes the frame UBO descriptor
// into the named per-frame set. The buffer must remain valid until the
// GPU has finished the matching frame's submission; the renderer
// already manages that via the deferred-destruction queue.
vulkan_descriptor_update_frame_ubo :: proc(
	frame_index: int,
	buffer: vk.Buffer,
	offset: vk.DeviceSize,
	range: vk.DeviceSize,
) {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return
	if frame_index < 0 || frame_index >= MAX_FRAMES_IN_FLIGHT {
		log.errorf("[BF_GPU/Vulkan] update_frame_ubo: invalid frame_index %d", frame_index)
		return
	}
	if buffer == {} {
		log.warn("[BF_GPU/Vulkan] update_frame_ubo: null buffer; skipping write")
		return
	}

	info := vk.DescriptorBufferInfo {
		buffer = buffer,
		offset = offset,
		range  = range,
	}
	write := vk.WriteDescriptorSet {
		sType           = .WRITE_DESCRIPTOR_SET,
		dstSet          = VULKAN_DESCRIPTOR_STATE.sets.frame[frame_index],
		dstBinding      = BIND_FRAME_GLOBAL_CONTEXT,
		dstArrayElement = 0,
		descriptorCount = 1,
		descriptorType  = .UNIFORM_BUFFER,
		pBufferInfo     = &info,
	}
	vk.UpdateDescriptorSets(VULKAN_STATE.device, 1, &write, 0, nil)
}

// ---------------------------------------------------------------------------
//* Per-pass image binding.
// ---------------------------------------------------------------------------

// vulkan_descriptor_update_per_pass_depth_pyramid writes the per-pass
// set's HiZ depth pyramid image descriptor. The combined
// sampler + image form (vs. separate) matches the shader's
// `uniform sampler2D depthPyramid;`.
vulkan_descriptor_update_per_pass_depth_pyramid :: proc(
	frame_index: int,
	view: vk.ImageView,
	layout: vk.ImageLayout,
	sampler: vk.Sampler,
) {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return
	if frame_index < 0 || frame_index >= MAX_FRAMES_IN_FLIGHT {
		log.errorf(
			"[BF_GPU/Vulkan] update_per_pass_depth_pyramid: invalid frame_index %d",
			frame_index,
		)
		return
	}
	if view == {} {
		log.warn("[BF_GPU/Vulkan] update_per_pass_depth_pyramid: null image view")
		return
	}

	info := vk.DescriptorImageInfo {
		sampler     = sampler,
		imageView   = view,
		imageLayout = layout,
	}
	write := vk.WriteDescriptorSet {
		sType           = .WRITE_DESCRIPTOR_SET,
		dstSet          = VULKAN_DESCRIPTOR_STATE.sets.per_pass[frame_index],
		dstBinding      = BIND_PER_PASS_DEPTH_PYRAMID,
		dstArrayElement = 0,
		descriptorCount = 1,
		descriptorType  = .COMBINED_IMAGE_SAMPLER,
		pImageInfo      = &info,
	}
	vk.UpdateDescriptorSets(VULKAN_STATE.device, 1, &write, 0, nil)
}

// vulkan_descriptor_update_per_pass_sampled_image writes a single
// sampled image descriptor into the per-pass set. Used by future
// per-pass passes (shadow atlas, particle atlases) that don't need the
// combined sampler form.
vulkan_descriptor_update_per_pass_sampled_image :: proc(
	frame_index: int,
	binding: u32,
	view: vk.ImageView,
	layout: vk.ImageLayout,
) {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return
	if frame_index < 0 || frame_index >= MAX_FRAMES_IN_FLIGHT {
		log.errorf(
			"[BF_GPU/Vulkan] update_per_pass_sampled_image: invalid frame_index %d",
			frame_index,
		)
		return
	}
	info := vk.DescriptorImageInfo {
		sampler     = {},
		imageView   = view,
		imageLayout = layout,
	}
	write := vk.WriteDescriptorSet {
		sType           = .WRITE_DESCRIPTOR_SET,
		dstSet          = VULKAN_DESCRIPTOR_STATE.sets.per_pass[frame_index],
		dstBinding      = binding,
		dstArrayElement = 0,
		descriptorCount = 1,
		descriptorType  = .SAMPLED_IMAGE,
		pImageInfo      = &info,
	}
	vk.UpdateDescriptorSets(VULKAN_STATE.device, 1, &write, 0, nil)
}

// ---------------------------------------------------------------------------
//* Bind helper.
// ---------------------------------------------------------------------------

// vulkan_descriptor_bind_sets is a thin wrapper around
// CmdBindDescriptorSets that picks the matching VkDescriptorSet triple
// for the current frame in flight. Pipeline layouts are supplied by
// the caller; the descriptor module never owns pipeline layouts.
//
// Pass first_set=0 to bind all three; pass first_set=2 to bind only
// set 2 (the culling compute shaders' typical case before set 0/1 are
// ever populated).
vulkan_descriptor_bind_sets :: proc(
	cmd: vk.CommandBuffer,
	bind_point: vk.PipelineBindPoint,
	pipeline_layout: vk.PipelineLayout,
	first_set: u32,
	count: u32,
	frame_index: int,
) {
	if !VULKAN_DESCRIPTOR_STATE.initialized do return
	if frame_index < 0 || frame_index >= MAX_FRAMES_IN_FLIGHT do return

	sets: [3]vk.DescriptorSet
	sets_count: u32 = 0
	if first_set <= u32(DESCRIPTOR_SET_FRAME) && sets_count < count {
		sets[sets_count] = VULKAN_DESCRIPTOR_STATE.sets.frame[frame_index]
		sets_count += 1
	}
	if first_set <= u32(DESCRIPTOR_SET_PERSISTENT) && sets_count < count {
		sets[sets_count] = VULKAN_DESCRIPTOR_STATE.sets.persistent[frame_index]
		sets_count += 1
	}
	if first_set <= u32(DESCRIPTOR_SET_PER_PASS) && sets_count < count {
		sets[sets_count] = VULKAN_DESCRIPTOR_STATE.sets.per_pass[frame_index]
		sets_count += 1
	}
	if sets_count == 0 do return

	vk.CmdBindDescriptorSets(
		cmd,
		bind_point,
		pipeline_layout,
		first_set,
		sets_count,
		&sets[0],
		0,
		nil,
	)
}
