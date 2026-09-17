// BF_GPU/Vk_Pipeline.odin
//
// Graphics + compute pipeline creation backed by VkShaderModule entries
// produced by Vk_Shader.odin.
//
// The pipeline builder is the only consumer of Vulkan_Shader rawptr
// handles the renderer hands out: it unwraps each handle, assembles
// the matching VkPipelineShaderStageCreateInfo, configures the dynamic
// rendering state (since the engine never creates a VkRenderPass),
// and submits the build to the device with a shared VkPipelineCache.
//
// Descriptor set layouts are owned by the descriptor model. The default
// renderer-facing bindings still match the "set 0 = frame, set 1 =
// persistent/bindless, set 2 = per-pass" plan, but the layouts themselves
// are created and supplied by the caller; this file only stitches them
// into VkPipelineLayout objects.
//
// All entry points return an opaque rawptr compatible with
// GPU_Backend.create_pipeline / destroy_pipeline. The internal
// Vulkan_Pipeline struct tracks the VkPipeline + VkPipelineLayout
// pair so destroy_pipeline releases both.

package BF_GPU

import "core:log"
import "core:mem"
import vk "vendor:vulkan"

@(private)
PIPELINE_KEY_HASHER_SEED: u64 = 0x9E3779B97F4A7C15

@(private)
Pipeline_Stage_Flags :: bit_set[Shader_Stage]
@(private)
PIPELINE_STAGE_ALL_GRAPHICS: Pipeline_Stage_Flags = {.Vertex, .Fragment, .Task, .Mesh}
@(private)
PIPELINE_STAGE_COMPUTE:     Pipeline_Stage_Flags = {.Compute}

// Pipeline_Handle is the opaque identity the renderer uses to refer
// to a compiled pipeline. 0 is reserved as the invalid handle so the
// destroy path can early-out without dereferencing a nil pointer.
Pipeline_Handle :: distinct u64
PIPELINE_HANDLE_INVALID :: Pipeline_Handle(0)

@(private)
VULKAN_PIPELINE_NEXT_ID: u64 = 1

@(private)
Vulkan_Pipeline_Map :: map[u64]^Vulkan_Pipeline

@(private)
VULKAN_PIPELINE_MAP: Vulkan_Pipeline_Map

// Vulkan_Pipeline is the backend-private record attached to every
// successfully created pipeline. The struct is heap-allocated and
// tracked through VULKAN_PIPELINE_MAP so destroy_pipeline can find
// the VkPipelineLayout that must also be released.
Vulkan_Pipeline :: struct {
	handle:        Pipeline_Handle,
	pipeline:      vk.Pipeline,
	layout:        vk.PipelineLayout,
	stages:        Pipeline_Stage_Flags,
	vertex_shader: rawptr,
	fragment_shader: rawptr,
	task_shader:   rawptr,
	mesh_shader:   rawptr,
	compute_shader: rawptr,
}

// Graphics_Pipeline_Description is the host-side descriptor for a
// traditional graphics pipeline. Mesh-shader paths are configured via
// the same struct: when task_shader + mesh_shader are non-nil the
// builder omits the vertex shader and supplies the mesh stages
// instead.
//
// All rawptr fields are opaque Vulkan_Shader handles returned by
// vulkan_backend_compile_shader; the renderer does not know or care
// that they point at Vulkan_Shader structs.
Graphics_Pipeline_Description :: struct {
	name:               string,
	vertex_shader:      rawptr,
	fragment_shader:    rawptr,
	task_shader:        rawptr,
	mesh_shader:        rawptr,
	// Push constants shared across the bound stages.
	push_constant_size:  u32,
	// Descriptor set layouts for the pipeline layout. Owned by the
	// caller; the pipeline only borrows references.
	descriptor_set_layouts: []vk.DescriptorSetLayout,
	// Color attachments the dynamic-rendering pass will target. The
	// builder forwards them verbatim to the pipeline's rendering
	// state; an empty slice implies the pipeline does not write
	// colour attachments (e.g. depth-only pre-passes).
	color_formats: []vk.Format,
	depth_format:  vk.Format,
	// Dynamic state is always enabled. The listed states are the
	// minimum the renderer requires for its frame loop; passing in
	// extra states is allowed.
	depth_test:        bool,
	depth_write:       bool,
	cull_mode:         vk.CullModeFlags,
	front_face:        vk.FrontFace,
	samples:           u32,
}

// Compute_Pipeline_Description is the host-side descriptor for a
// compute pipeline. The single shader must have been compiled with
// Shader_Stage.Compute; the builder validates that and rejects
// otherwise.
Compute_Pipeline_Description :: struct {
	name:                  string,
	compute_shader:        rawptr,
	push_constant_size:    u32,
	descriptor_set_layouts: []vk.DescriptorSetLayout,
}

// ---------------------------------------------------------------------------
//* Hashing for the pipeline map.

// pipeline_key derives a stable u64 key from a graphics descriptor so
// duplicate pipeline_create calls return the existing entry instead
// of rebuilding the pipeline. The shader rawptrs fold into the key
// directly (no dereferencing); a duplicate request means the renderer
// asked twice for the same shader / state combination.
//
// For dynamic state that can change every frame (viewport, scissor,
// ...) the key only covers the static structural parameters. Pipeline
// instances sharing the same structural key share a VkPipeline; their
// dynamic state is set per-bind via vkCmdSetViewport / etc.
pipeline_key :: proc(d: ^Graphics_Pipeline_Description) -> u64 {
	h := PIPELINE_KEY_HASHER_SEED
	h = hash_combine_u64(h, u64(cast(uintptr)d.vertex_shader))
	h = hash_combine_u64(h, u64(cast(uintptr)d.fragment_shader))
	h = hash_combine_u64(h, u64(cast(uintptr)d.task_shader))
	h = hash_combine_u64(h, u64(cast(uintptr)d.mesh_shader))
	h = hash_combine_u64(h, u64(d.push_constant_size))
	h = hash_combine_u64(h, u64(d.depth_test))
	h = hash_combine_u64(h, u64(d.depth_write))
	// bit_set / enum values are 4-byte backed; promote through u64
	// to fold into the key without per-field byte-level fiddling.
	h = hash_combine_u64(h, u64(transmute(u32)d.cull_mode))
	h = hash_combine_u64(h, u64(transmute(u32)d.front_face))
	h = hash_combine_u64(h, u64(d.samples))
	h = hash_combine_u64(h, u64(transmute(u32)d.depth_format))
	for f in d.color_formats {
		h = hash_combine_u64(h, u64(transmute(u32)f))
	}
	for layout in d.descriptor_set_layouts {
		h = hash_combine_u64(h, u64(cast(uintptr)layout))
	}
	h = hash_combine_u64(h, fnv1a_string(d.name))
	return h
}

pipeline_key_compute :: proc(d: ^Compute_Pipeline_Description) -> u64 {
	h := PIPELINE_KEY_HASHER_SEED
	h = hash_combine_u64(h, u64(cast(uintptr)d.compute_shader))
	h = hash_combine_u64(h, u64(d.push_constant_size))
	for layout in d.descriptor_set_layouts {
		h = hash_combine_u64(h, u64(cast(uintptr)layout))
	}
	h = hash_combine_u64(h, fnv1a_string(d.name))
	return h
}

hash_combine_u64 :: proc(h, v_in: u64) -> u64 {
	v := v_in
	v = (v ~ (v >> 30)) * 0xBF58476D1CE4E5B9
	v = (v ~ (v >> 27)) * 0x94D049BB133111EB
	v = v ~ (v >> 31)
	return h ~ v
}

// ---------------------------------------------------------------------------
//* Validation helpers.

// vulkan_validate_graphics_descriptor rejects descriptors that are
// obviously malformed: missing both vertex and task+mesh stages, no
// shader pairs to draw, or shaders of the wrong stage.
vulkan_validate_graphics_descriptor :: proc(d: ^Graphics_Pipeline_Description) -> cstring {
	if d == nil {
		return "graphics descriptor is nil"
	}
	if d.push_constant_size > 128 {
		return "graphics push_constant_size exceeds the 128-byte Vulkan minimum"
	}
	if d.samples != 1 && d.samples != 2 && d.samples != 4 && d.samples != 8 && d.samples != 16 {
		return "graphics samples must be a power of two in {1, 2, 4, 8, 16}"
	}
	has_vert := d.vertex_shader != nil
	has_mesh := d.task_shader != nil && d.mesh_shader != nil
	has_task := d.task_shader != nil && d.mesh_shader == nil
	has_mesh_only := d.task_shader == nil && d.mesh_shader != nil
	if !has_vert && !has_mesh {
		return "graphics pipeline must declare either vertex_shader or task+mesh shader pair"
	}
	if has_task || has_mesh_only {
		return "graphics pipeline must declare both task and mesh shaders (or neither)"
	}
	if has_vert && d.fragment_shader == nil && len(d.color_formats) > 0 {
		return "graphics pipeline writes colour but is missing a fragment shader"
	}
	if !has_vert && d.fragment_shader == nil {
		return "graphics mesh pipeline is missing a fragment shader"
	}
	if d.vertex_shader != nil && vulkan_shader_stage(d.vertex_shader) != .Vertex {
		return "vertex shader stage mismatch"
	}
	if d.fragment_shader != nil && vulkan_shader_stage(d.fragment_shader) != .Fragment {
		return "fragment shader stage mismatch"
	}
	if d.task_shader != nil && vulkan_shader_stage(d.task_shader) != .Task {
		return "task shader stage mismatch"
	}
	if d.mesh_shader != nil && vulkan_shader_stage(d.mesh_shader) != .Mesh {
		return "mesh shader stage mismatch"
	}
	return ""
}

vulkan_validate_compute_descriptor :: proc(d: ^Compute_Pipeline_Description) -> cstring {
	if d == nil {
		return "compute descriptor is nil"
	}
	if d.compute_shader == nil {
		return "compute descriptor is missing the compute shader"
	}
	if vulkan_shader_stage(d.compute_shader) != .Compute {
		return "compute shader stage mismatch"
	}
	if d.push_constant_size > 128 {
		return "compute push_constant_size exceeds the 128-byte Vulkan minimum"
	}
	return ""
}

// ---------------------------------------------------------------------------
//* Pipeline layout creation.

// vulkan_create_pipeline_layout wraps VkCreatePipelineLayout. The
// caller owns the descriptor set layouts; the layout only borrows
// references. Push constants are declared once for every stage that
// shares them.
vulkan_create_pipeline_layout :: proc(
	set_layouts: []vk.DescriptorSetLayout,
	push_constant_size: u32,
	push_stages: vk.ShaderStageFlags,
) -> (
	vk.PipelineLayout,
	bool,
) {
	ranges: [1]vk.PushConstantRange
	if push_constant_size > 0 {
		ranges[0] = vk.PushConstantRange {
			stageFlags = push_stages,
			offset     = 0,
			size       = push_constant_size,
		}
	}

	create_info := vk.PipelineLayoutCreateInfo {
		sType                  = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount         = u32(len(set_layouts)),
		pSetLayouts            = raw_data(set_layouts) if len(set_layouts) > 0 else nil,
		pushConstantRangeCount = u32(1) if push_constant_size > 0 else 0,
		pPushConstantRanges     = &ranges[0] if push_constant_size > 0 else nil,
	}

	layout: vk.PipelineLayout
	result := vk.CreatePipelineLayout(VULKAN_STATE.device, &create_info, nil, &layout)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkCreatePipelineLayout failed: %v", result)
		return {}, false
	}
	return layout, true
}

// ---------------------------------------------------------------------------
//* Graphics pipeline creation.

// vulkan_create_graphics_pipeline builds the VkPipeline + layout pair
// and inserts the result into the cache keyed by the descriptor hash.
// A duplicate request returns the cached entry with no further work.
vulkan_create_graphics_pipeline :: proc(d: ^Graphics_Pipeline_Description) -> rawptr {
	if d == nil do return nil
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] create_graphics_pipeline before Vulkan device creation")
		return nil
	}
	if msg := vulkan_validate_graphics_descriptor(d); msg != "" {
		log.errorf("[BF_GPU/Vulkan] graphics pipeline rejected: %s", msg)
		return nil
	}

	key := pipeline_key(d)
	if entry, found := VULKAN_PIPELINE_MAP[key]; found && entry != nil {
		return rawptr(entry)
	}

	stages: Pipeline_Stage_Flags = {}
	push_stages: vk.ShaderStageFlags = {}
	if d.vertex_shader != nil {
		stages += {.Vertex}
		push_stages += {.VERTEX}
	}
	if d.fragment_shader != nil {
		stages += {.Fragment}
		push_stages += {.FRAGMENT}
	}
	if d.task_shader != nil {
		stages += {.Task}
		// Vulkan 1.3 exposes mesh / task stages under EXT names; the
		// driver falls back to NV when EXT is unavailable. Both are
		// required when VK_EXT_mesh_shader is enabled, which the
		// renderer only does when the BF_GPU_Mesh extension is
		// attached.
		push_stages += {.TASK_EXT}
		push_stages += {.TASK_NV}
	}
	if d.mesh_shader != nil {
		stages += {.Mesh}
		push_stages += {.MESH_EXT}
		push_stages += {.MESH_NV}
	}

	layout, layout_ok := vulkan_create_pipeline_layout(
		d.descriptor_set_layouts,
		d.push_constant_size,
		push_stages,
	)
	if !layout_ok {
		return nil
	}

	pipeline, pipeline_ok := vulkan_build_graphics_pipeline(d, layout, stages)
	if !pipeline_ok {
		vk.DestroyPipelineLayout(VULKAN_STATE.device, layout, nil)
		return nil
	}

	entry := new(Vulkan_Pipeline)
	entry^ = Vulkan_Pipeline {
		handle         = Pipeline_Handle(VULKAN_PIPELINE_NEXT_ID),
		pipeline       = pipeline,
		layout         = layout,
		stages         = stages,
		vertex_shader  = d.vertex_shader,
		fragment_shader = d.fragment_shader,
		task_shader    = d.task_shader,
		mesh_shader    = d.mesh_shader,
	}
	VULKAN_PIPELINE_NEXT_ID += 1
	VULKAN_PIPELINE_MAP[key] = entry

	log.infof(
		"[BF_GPU/Vulkan] graphics pipeline %q built (stages=%v, handle=%d)",
		d.name,
		stages,
		u64(entry.handle),
	)
	return rawptr(entry)
}

// vulkan_build_graphics_pipeline assembles the VkGraphicsPipelineCreateInfo
// with every dynamic state the renderer needs. Viewport + scissor +
// depth bias + blend constants are always dynamic so the per-frame
// state can change without rebuilding the pipeline.
vulkan_build_graphics_pipeline :: proc(
	d: ^Graphics_Pipeline_Description,
	layout: vk.PipelineLayout,
	stages: Pipeline_Stage_Flags,
) -> (
	vk.Pipeline,
	bool,
) {
	dynamic_states := []vk.DynamicState {
		.VIEWPORT,
		.SCISSOR,
		.DEPTH_BIAS,
		.BLEND_CONSTANTS,
		.STENCIL_COMPARE_MASK,
		.STENCIL_WRITE_MASK,
		.STENCIL_REFERENCE,
	}

	dynamic_state_info := vk.PipelineDynamicStateCreateInfo {
		sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = u32(len(dynamic_states)),
		pDynamicStates    = raw_data(dynamic_states),
	}

	// Shader stages.
	stage_infos: [4]vk.PipelineShaderStageCreateInfo
	stage_count: u32 = 0
	if d.vertex_shader != nil {
		stage_infos[stage_count] = vulkan_make_shader_stage(.Vertex, d.vertex_shader)
		stage_count += 1
	}
	if d.task_shader != nil {
		stage_infos[stage_count] = vulkan_make_shader_stage(.Task, d.task_shader)
		stage_count += 1
	}
	if d.mesh_shader != nil {
		stage_infos[stage_count] = vulkan_make_shader_stage(.Mesh, d.mesh_shader)
		stage_count += 1
	}
	if d.fragment_shader != nil {
		stage_infos[stage_count] = vulkan_make_shader_stage(.Fragment, d.fragment_shader)
		stage_count += 1
	}

	// Vertex input state. The culling shaders consume gl_VertexIndex
	// / gl_InstanceIndex so the vertex buffer binding is empty; the
	// renderer forwards per-mesh vertex data through buffer device
	// address SSBOs. The meshlet / task pipeline path also skips the
	// vertex input state.
	vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
		sType                     = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount = 0,
		vertexAttributeDescriptionCount = 0,
	}

	input_assembly_info := vk.PipelineInputAssemblyStateCreateInfo {
		sType                  = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology               = .TRIANGLE_LIST,
		primitiveRestartEnable = false,
	}

	tessellation_info := vk.PipelineTessellationStateCreateInfo {
		sType                = .PIPELINE_TESSELLATION_STATE_CREATE_INFO,
		patchControlPoints   = 0,
	}

	viewport_info := vk.PipelineViewportStateCreateInfo {
		sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		scissorCount  = 1,
	}

	rasterization_info := vk.PipelineRasterizationStateCreateInfo {
		sType                    = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		depthClampEnable         = false,
		rasterizerDiscardEnable  = false,
		polygonMode              = .FILL,
		cullMode                 = d.cull_mode,
		frontFace                = d.front_face,
		depthBiasEnable          = false,
		depthBiasConstantFactor  = 0.0,
		depthBiasClamp           = 0.0,
		depthBiasSlopeFactor     = 0.0,
		lineWidth                = 1.0,
	}

	multisample_info := vk.PipelineMultisampleStateCreateInfo {
		sType                 = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		rasterizationSamples  = vulkan_samples_to_vk(d.samples),
		sampleShadingEnable   = false,
		minSampleShading      = 1.0,
	}

	depth_stencil_info := vk.PipelineDepthStencilStateCreateInfo {
		sType                 = .PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		depthTestEnable       = b32(d.depth_test),
		depthWriteEnable      = b32(d.depth_write),
		depthCompareOp        = .LESS_OR_EQUAL if d.depth_test else .ALWAYS,
		depthBoundsTestEnable = false,
		stencilTestEnable     = false,
		front                 = {},
		back                  = {},
		minDepthBounds        = 0.0,
		maxDepthBounds        = 1.0,
	}

	color_blend_attachment := vk.PipelineColorBlendAttachmentState {
		blendEnable         = false,
		srcColorBlendFactor = .SRC_ALPHA,
		dstColorBlendFactor = .ONE_MINUS_SRC_ALPHA,
		colorBlendOp        = .ADD,
		srcAlphaBlendFactor = .ONE,
		dstAlphaBlendFactor = .ONE_MINUS_SRC_ALPHA,
		alphaBlendOp        = .ADD,
		colorWriteMask      = {.R, .G, .B, .A},
	}

	color_blend_info := vk.PipelineColorBlendStateCreateInfo {
		sType             = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		logicOpEnable     = false,
		logicOp           = .COPY,
		attachmentCount   = u32(len(d.color_formats)),
		pAttachments      = &color_blend_attachment if len(d.color_formats) > 0 else nil,
		blendConstants    = {0, 0, 0, 0},
	}

	// Dynamic rendering state. VUID 07878 requires format lists to be
	// supplied via pColorAttachmentFormats for VkPipelineRenderingCreateInfo;
	// VK_FORMAT_UNDEFINED signals "this pipeline is not used for colour".
	rendering_formats: [16]vk.Format
	rendering_format_count := u32(len(d.color_formats))
	if rendering_format_count > u32(len(rendering_formats)) {
		log.error("[BF_GPU/Vulkan] graphics pipeline exceeds 16 colour attachments")
		return {}, false
	}
	for fmt, i in d.color_formats {
		rendering_formats[i] = fmt
	}

	rendering_info := vk.PipelineRenderingCreateInfo {
		sType                        = .PIPELINE_RENDERING_CREATE_INFO,
		viewMask                     = 0,
		colorAttachmentCount         = rendering_format_count,
		pColorAttachmentFormats      = &rendering_formats[0] if rendering_format_count > 0 else nil,
		depthAttachmentFormat        = d.depth_format,
		stencilAttachmentFormat      = vk.Format.UNDEFINED,
	}

	create_info := vk.GraphicsPipelineCreateInfo {
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		pNext               = &rendering_info,
		flags               = {},
		stageCount          = stage_count,
		pStages             = &stage_infos[0],
		pVertexInputState   = &vertex_input_info,
		pInputAssemblyState = &input_assembly_info,
		pTessellationState  = &tessellation_info,
		pViewportState      = &viewport_info,
		pRasterizationState = &rasterization_info,
		pMultisampleState   = &multisample_info,
		pDepthStencilState  = &depth_stencil_info,
		pColorBlendState    = &color_blend_info,
		pDynamicState       = &dynamic_state_info,
		layout              = layout,
		renderPass          = {},
		subpass             = 0,
		basePipelineHandle  = {},
		basePipelineIndex   = -1,
	}

	cache := vulkan_pipeline_cache_get()
	pipeline: vk.Pipeline
	result := vk.CreateGraphicsPipelines(
		VULKAN_STATE.device,
		cache,
		1,
		&create_info,
		nil,
		&pipeline,
	)
	if result != .SUCCESS {
		log.errorf(
			"[BF_GPU/Vulkan] vkCreateGraphicsPipelines failed for %q: %v",
			d.name,
			result,
		)
		return {}, false
	}
	return pipeline, true
}

// vulkan_make_shader_stage builds the VkPipelineShaderStageCreateInfo
// for a single shader module. The entry point name "main" matches the
// GLSL convention every renderer-owned shader follows.
vulkan_make_shader_stage :: proc(stage: Shader_Stage, shader: rawptr) -> vk.PipelineShaderStageCreateInfo {
	return vk.PipelineShaderStageCreateInfo {
		sType  = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage  = vulkan_shader_stage_to_vk(stage),
		module = vulkan_shader_module(shader),
		pName  = "main",
		pSpecializationInfo = nil,
	}
}

vulkan_shader_stage_to_vk :: proc(s: Shader_Stage) -> vk.ShaderStageFlags {
	#partial switch s {
	case .Vertex:   return {.VERTEX}
	case .Fragment: return {.FRAGMENT}
	case .Compute:  return {.COMPUTE}
	case .Task:     return {.TASK_EXT} | {.TASK_NV}
	case .Mesh:     return {.MESH_EXT} | {.MESH_NV}
	}
	return {.VERTEX}
}

// ---------------------------------------------------------------------------
//* Compute pipeline creation.

// vulkan_create_compute_pipeline builds the VkPipeline + layout pair
// for a single compute shader and inserts it into the cache.
vulkan_create_compute_pipeline :: proc(d: ^Compute_Pipeline_Description) -> rawptr {
	if d == nil do return nil
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] create_compute_pipeline before Vulkan device creation")
		return nil
	}
	if msg := vulkan_validate_compute_descriptor(d); msg != "" {
		log.errorf("[BF_GPU/Vulkan] compute pipeline rejected: %s", msg)
		return nil
	}

	key := pipeline_key_compute(d)
	if entry, found := VULKAN_PIPELINE_MAP[key]; found && entry != nil {
		return rawptr(entry)
	}

	layout, layout_ok := vulkan_create_pipeline_layout(
		d.descriptor_set_layouts,
		d.push_constant_size,
		{.COMPUTE},
	)
	if !layout_ok {
		return nil
	}

	stage := vulkan_make_shader_stage(.Compute, d.compute_shader)
	pipeline, pipeline_ok := vulkan_build_compute_pipeline(d, layout, &stage)
	if !pipeline_ok {
		vk.DestroyPipelineLayout(VULKAN_STATE.device, layout, nil)
		return nil
	}

	entry := new(Vulkan_Pipeline)
	entry^ = Vulkan_Pipeline {
		handle         = Pipeline_Handle(VULKAN_PIPELINE_NEXT_ID),
		pipeline       = pipeline,
		layout         = layout,
		stages         = {.Compute},
		compute_shader = d.compute_shader,
	}
	VULKAN_PIPELINE_NEXT_ID += 1
	VULKAN_PIPELINE_MAP[key] = entry

	log.infof(
		"[BF_GPU/Vulkan] compute pipeline %q built (handle=%d)",
		d.name,
		u64(entry.handle),
	)
	return rawptr(entry)
}

vulkan_build_compute_pipeline :: proc(
	d: ^Compute_Pipeline_Description,
	layout: vk.PipelineLayout,
	stage: ^vk.PipelineShaderStageCreateInfo,
) -> (
	vk.Pipeline,
	bool,
) {
	create_info := vk.ComputePipelineCreateInfo {
		sType              = .COMPUTE_PIPELINE_CREATE_INFO,
		stage              = stage^,
		layout             = layout,
		basePipelineHandle = {},
		basePipelineIndex  = -1,
	}

	cache := vulkan_pipeline_cache_get()
	pipeline: vk.Pipeline
	result := vk.CreateComputePipelines(
		VULKAN_STATE.device,
		cache,
		1,
		&create_info,
		nil,
		&pipeline,
	)
	if result != .SUCCESS {
		log.errorf(
			"[BF_GPU/Vulkan] vkCreateComputePipelines failed for %q: %v",
			d.name,
			result,
		)
		return {}, false
	}
	return pipeline, true
}

// ---------------------------------------------------------------------------
//* Backend hook + destruction.

// vulkan_backend_create_pipeline_impl is the GPU_Backend.create_pipeline
// implementation. It dispatches on the descriptor's tag field (the
// first u64 of the descriptor pointer) to either the graphics or the
// compute builder. The renderer-side callers know which descriptor
// type they hold.
vulkan_backend_create_pipeline_impl :: proc(shader: rawptr, descriptor: rawptr) -> rawptr {
	// The renderer's pipeline_descriptors struct is a tagged union of
	// (graphics_descriptor, compute_descriptor) followed by a tag.
	// We read the tag through the descriptor pointer rather than
	// from `shader`; shader is the legacy single-shader input that
	// the existing GPU_Backend surface accepts.
	if descriptor == nil do return nil
	desc_tag := vulkan_pipeline_descriptor_tag(descriptor)
	switch desc_tag {
	case .Graphics:
		gd := vulkan_pipeline_descriptor_graphics(descriptor)
		if gd == nil {
			log.error("[BF_GPU/Vulkan] create_pipeline: graphics descriptor is nil")
			return nil
		}
		// The legacy single-shader argument supplies the fragment
		// shader for callers that pass only one. The descriptor's
		// fragment_shader field takes precedence when non-nil.
		if gd.fragment_shader == nil && shader != nil &&
		   vulkan_shader_stage(shader) == .Fragment {
			gd.fragment_shader = shader
		}
		return vulkan_create_graphics_pipeline(gd)
	case .Compute:
		cd := vulkan_pipeline_descriptor_compute(descriptor)
		if cd == nil {
			log.error("[BF_GPU/Vulkan] create_pipeline: compute descriptor is nil")
			return nil
		}
		if cd.compute_shader == nil && shader != nil &&
		   vulkan_shader_stage(shader) == .Compute {
			cd.compute_shader = shader
		}
		return vulkan_create_compute_pipeline(cd)
	}
	log.error("[BF_GPU/Vulkan] create_pipeline: unknown descriptor tag")
	return nil
}

// vulkan_backend_destroy_pipeline_impl is the GPU_Backend.destroy_pipeline
// implementation. Tears down the VkPipeline + VkPipelineLayout and
// drops the cache slot. Shader modules referenced by the pipeline are
// not destroyed here; the caller is expected to manage those lifetimes
// explicitly so a pipeline and its shaders can be released in any
// order without a use-after-free.
vulkan_backend_destroy_pipeline_impl :: proc(p: rawptr) {
	if p == nil do return
	entry := cast(^Vulkan_Pipeline)p

	// Find and remove the cache slot. The map is keyed by descriptor
	// hash, so we walk it to find the matching entry; pipelines are
	// typically destroyed once per session so this is cheap enough.
	removed_key: u64
	removed := false
	for k, v in VULKAN_PIPELINE_MAP {
		if v == entry {
			removed_key = k
			removed = true
			break
		}
	}
	if removed {
		delete_key(&VULKAN_PIPELINE_MAP, removed_key)
	}

	if VULKAN_STATE.device != nil {
		if entry.pipeline != {} {
			vk.DestroyPipeline(VULKAN_STATE.device, entry.pipeline, nil)
		}
		if entry.layout != {} {
			vk.DestroyPipelineLayout(VULKAN_STATE.device, entry.layout, nil)
		}
	}
	free(entry)
}

// vulkan_pipeline_map_clear empties the map and destroys every entry.
// Called from vulkan_shutdown after the device is idle.
vulkan_pipeline_map_clear :: proc() {
	for _, entry in VULKAN_PIPELINE_MAP {
		if entry == nil do continue
		if VULKAN_STATE.device != nil {
			if entry.pipeline != {} {
				vk.DestroyPipeline(VULKAN_STATE.device, entry.pipeline, nil)
			}
			if entry.layout != {} {
				vk.DestroyPipelineLayout(VULKAN_STATE.device, entry.layout, nil)
			}
		}
		free(entry)
	}
	clear(&VULKAN_PIPELINE_MAP)
	VULKAN_PIPELINE_NEXT_ID = 1
}

// ---------------------------------------------------------------------------
//* Pipeline lookup helpers.

// vulkan_pipeline_handle returns the pipeline's identity handle. Used
// by the frame-recording layer when the renderer wants a numeric id
// rather than the opaque rawptr.
vulkan_pipeline_handle :: proc(p: rawptr) -> Pipeline_Handle {
	if p == nil do return PIPELINE_HANDLE_INVALID
	entry := cast(^Vulkan_Pipeline)p
	return entry.handle
}

// vulkan_pipeline_vk returns the VkPipeline + VkPipelineLayout pair
// for binding. Returns (zero, zero) when the handle is nil so callers
// can early-out without nil-checking.
vulkan_pipeline_vk :: proc(p: rawptr) -> (vk.Pipeline, vk.PipelineLayout) {
	if p == nil do return {}, {}
	entry := cast(^Vulkan_Pipeline)p
	return entry.pipeline, entry.layout
}

vulkan_pipeline_map_len :: proc() -> int {
	if VULKAN_PIPELINE_MAP == nil do return 0
	return len(VULKAN_PIPELINE_MAP)
}

// ---------------------------------------------------------------------------
//* Descriptor tagging.
//
// The renderer's GPU_Backend.create_pipeline contract is `(shader,
// descriptor) -> rawptr`. Because the descriptor can be either a
// graphics or compute layout, the descriptor pointer embeds a small
// header the backend reads to dispatch to the right builder. The
// renderer-side wrappers (in Renderer.odin / Resource plumbing) wrap
// their typed descriptor in vulkan_make_pipeline_descriptor so the
// backend does not need a parallel pair of entry points.

Pipeline_Descriptor_Tag :: enum u8 {
	Graphics,
	Compute,
}

Pipeline_Descriptor_Header :: struct {
	tag:    Pipeline_Descriptor_Tag,
	// Total byte length of the descriptor block (including this
	// header). Vulkan_Free_Pipeline_Descriptor uses this value to
	// recover the original []byte slice and hand it back to
	// runtime.delete without a side-table.
	// Storing total here rather than only the payload size keeps
	// the frees safe even when multiple descriptor wrappers live
	// concurrently (one graphics, one compute).
	total_size: int,
}

vulkan_pipeline_descriptor_tag :: proc(descriptor: rawptr) -> Pipeline_Descriptor_Tag {
	header := cast(^Pipeline_Descriptor_Header)descriptor
	return header.tag
}

vulkan_pipeline_descriptor_graphics :: proc(descriptor: rawptr) -> ^Graphics_Pipeline_Description {
	header := cast(^Pipeline_Descriptor_Header)descriptor
	if header.tag != .Graphics do return nil
	ptr := cast(rawptr)(cast(uintptr)descriptor + size_of(Pipeline_Descriptor_Header))
	return cast(^Graphics_Pipeline_Description)ptr
}

vulkan_pipeline_descriptor_compute :: proc(descriptor: rawptr) -> ^Compute_Pipeline_Description {
	header := cast(^Pipeline_Descriptor_Header)descriptor
	if header.tag != .Compute do return nil
	ptr := cast(rawptr)(cast(uintptr)descriptor + size_of(Pipeline_Descriptor_Header))
	return cast(^Compute_Pipeline_Description)ptr
}

// vulkan_make_pipeline_descriptor wraps a typed descriptor in the
// header the backend expects. The returned memory owns the descriptor
// fields by copy and must be freed with the matching
// vulkan_free_pipeline_descriptor proc when the renderer drops the
// pipeline.
vulkan_make_graphics_descriptor :: proc(d: Graphics_Pipeline_Description) -> rawptr {
	// copy `d` into a local so the address-of expression is valid.
	local := d
	return vulkan_make_pipeline_descriptor(.Graphics, &local, size_of(Graphics_Pipeline_Description))
}

vulkan_make_compute_descriptor :: proc(d: Compute_Pipeline_Description) -> rawptr {
	local := d
	return vulkan_make_pipeline_descriptor(.Compute, &local, size_of(Compute_Pipeline_Description))
}

vulkan_make_pipeline_descriptor :: proc(
	tag: Pipeline_Descriptor_Tag,
	payload: rawptr,
	payload_size: int,
) -> rawptr {
	total_size := size_of(Pipeline_Descriptor_Header) + payload_size
	block := make([]byte, total_size)
	header := cast(^Pipeline_Descriptor_Header)raw_data(block)
	header.tag = tag
	header.total_size = total_size
	payload_dst := rawptr(uintptr(raw_data(block)) + size_of(Pipeline_Descriptor_Header))
	mem.copy(payload_dst, payload, payload_size)
	return raw_data(block)
}

// vulkan_free_pipeline_descriptor releases the wrapper returned by
// vulkan_make_pipeline_descriptor. Safe to pass nil. The wrapper's
// size is encoded in the header so the free can recover the original
// block even when several descriptors of different kinds coexist.
vulkan_free_pipeline_descriptor :: proc(descriptor: rawptr) {
	if descriptor == nil do return
	header := cast(^Pipeline_Descriptor_Header)descriptor
	total := header.total_size
	if total <= 0 {
		// Unknown / already freed; nothing to do.
		return
	}
	block := ([^]byte)(descriptor)[:total]
	delete(block)
}