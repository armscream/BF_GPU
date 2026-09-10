// BF_GPU/Frame_Context.odin
//
// Host-side owner of every persistent GPU buffer the culling pipeline
// reads from. Each persistent buffer is created with
// VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT (or the equivalent on the
// backend) and its device address is captured at creation time. The
// address is then stored in Gpu_Frame_Global_Context and uploaded as a
// single SSBO at the start of each frame.
//
// This module is the integration point between the GPU backend (the
// Vulkan/MoltenVK calls in Pipeline.odin) and the shader side (the
// std430 structs in Gpu_Types.odin). Everything below is intentionally
// backend-agnostic: the only thing it owns is u64 addresses, counts and
// flags. The actual VkBuffer / VkImage handles live in the GPU backend.

package BF_GPU

import "base:runtime"
import "core:log"

// ---------------------------------------------------------------------------
// Persistent GPU buffer registry.
//
// One entry per buffer the culling pipeline reads or writes. The host
// tracks the size, capacity, and device address; the GPU backend owns
// the actual handle and is responsible for keeping it alive.
//
// Buffer_Kind is the same as the FrameGlobalContext field name so that
// the per-frame upload loop is mechanical.
// ---------------------------------------------------------------------------

Gpu_Buffer_Handle :: distinct u64 // backend-defined (opaque); 0 == invalid

Gpu_Buffer_Kind :: enum {
	Global_Instance_Index,
	Global_Indirect_Command,
	Global_Indirect_Command_Descriptor,
	Global_Draw_Count,
	Global_Model_Allocation,
	Global_Mesh_Allocation,

	Camera_Visible_Index,
	Camera_Pool,
	Camera_Sparse_Map,

	Tag_Sparse_Map,
	Tag_Data,

	Transform_Pool,
	Transform_Sparse_Map,
	Transform_Model_Link,

	Static_Chunk_Data,
	Static_Chunk_Visible_Index,
	Static_Chunk_Count,

	Model_Address,
	Model_Pool,
	Model_Sparse_Map,
	Model_Count,
	Model_Visible_Index,

	Animation_Address,
	Animation_Pool,
	Animation_Sparse_Map,

	Material_Pool,
	Material_Lookup,
	Pipeline_Lookup,

	Scene_AABB,
	Morton_Keys,
	Morton_Values,
	Morton_Chunk_Data,
	Morton_Chunk_Indirect_Dispatch,
	Morton_Chunk_Indirect_Draw,
	Morton_Chunk_Visible_Indirect_Dispatch,
	Morton_Chunk_Visible_Index,
	Morton_Chunk_Transforms_Index,

	COUNT,
}

Gpu_Buffer_Entry :: struct {
	handle:    Gpu_Buffer_Handle, // 0 until the GPU backend creates it
	device_addr: u64,             // 0 until the backend resolves it
	size:      u64,               // current host-side allocation in bytes
	capacity:  u64,               // current GPU-side capacity in elements
	stride:    u32,               // sizeof(struct), used by shader-side indexing
}

// ---------------------------------------------------------------------------
// Frame_Context_State - the host-side mirror.
//
// Owns the buffer registry and the per-frame FrameGlobalContext snapshot.
// The snapshot is updated each frame from the render scene and pushed to
// the GPU at renderer_record_frame start.
// ---------------------------------------------------------------------------

Frame_Context_State :: struct {
	allocator:  runtime.Allocator,
	buffers:    [Gpu_Buffer_Kind.COUNT]Gpu_Buffer_Entry,
	frame_ctx:  Gpu_Frame_Global_Context, // mirrored host state, uploaded each frame
	frame_idx:  u64,
}

// frame_context_init initializes the host-side state. The GPU backend
// hooks are not invoked here - that happens during module_register once
// the backend service is available.
frame_context_init :: proc(state: ^Frame_Context_State, allocator := context.allocator) {
	state.allocator = allocator
	state.frame_idx = 0
	for &b in state.buffers {
		b = {}
	}
}

// frame_context_destroy frees any host-side bookkeeping. The GPU backend
// tears down its own resources during its own unload path.
frame_context_destroy :: proc(state: ^Frame_Context_State) {
	state^ = {}
}

// ---------------------------------------------------------------------------
// Per-buffer address lookup. Used by Scene.odin after extraction to
// populate the matching frame_ctx field.
// ---------------------------------------------------------------------------

frame_context_buffer_address :: #force_inline proc(state: ^Frame_Context_State, kind: Gpu_Buffer_Kind) -> u64 {
	return state.buffers[kind].device_addr
}

// frame_context_buffer_entry returns the full entry so callers can also
// read the current capacity (e.g. to know how many dense indices are
// valid on the GPU side).
frame_context_buffer_entry :: #force_inline proc(state: ^Frame_Context_State, kind: Gpu_Buffer_Kind) -> Gpu_Buffer_Entry {
	return state.buffers[kind]
}

// ---------------------------------------------------------------------------
// Frame-global write helpers.
//
// Scene.odin calls these after each frame's extraction to update the
// FrameGlobalContext mirror. Every field that maps 1:1 to a shader-side
// address lives here; scalar flags live in update_frame_scalar_state().
// ---------------------------------------------------------------------------

@(private = "file")
gctx_buffer_addrs := []Gpu_Buffer_Kind {
	.Global_Instance_Index,
	.Global_Indirect_Command,
	.Global_Indirect_Command_Descriptor,
	.Global_Draw_Count,
	.Global_Model_Allocation,
	.Global_Mesh_Allocation,
	.Camera_Visible_Index,
	.Camera_Pool,
	.Camera_Sparse_Map,
	.Tag_Sparse_Map,
	.Tag_Data,
	.Transform_Pool,
	.Transform_Sparse_Map,
	.Transform_Model_Link,
	.Static_Chunk_Data,
	.Static_Chunk_Visible_Index,
	.Static_Chunk_Count,
	.Model_Address,
	.Model_Pool,
	.Model_Sparse_Map,
	.Model_Count,
	.Model_Visible_Index,
	.Animation_Address,
	.Animation_Pool,
	.Animation_Sparse_Map,
	.Material_Pool,
	.Material_Lookup,
	.Pipeline_Lookup,
	.Scene_AABB,
	.Morton_Keys,
	.Morton_Values,
	.Morton_Chunk_Data,
	.Morton_Chunk_Indirect_Dispatch,
	.Morton_Chunk_Indirect_Draw,
	.Morton_Chunk_Visible_Indirect_Dispatch,
	.Morton_Chunk_Visible_Index,
	.Morton_Chunk_Transforms_Index,
}

// refresh_frame_addresses walks the buffer registry and copies every
// device address into the matching Gpu_Frame_Global_Context field. Call
// after the GPU backend creates its buffers (typically once at startup)
// or whenever the backend swaps an allocation.
refresh_frame_addresses :: proc(state: ^Frame_Context_State) {
	fc := &state.frame_ctx
	for kind, idx in gctx_buffer_addrs {
		addr := state.buffers[kind].device_addr
		switch idx {
		case 0:  fc.global_instance_index_buffer_addr = addr
		case 1:  fc.global_indirect_command_buffer_addr = addr
		case 2:  fc.global_indirect_command_desc_buffer_addr = addr
		case 3:  fc.global_draw_count_buffer_addr = addr
		case 4:  fc.global_model_allocation_buffer_addr = addr
		case 5:  fc.global_mesh_allocation_buffer_addr = addr
		case 6:  fc.camera_visible_index_buffer_addr = addr
		case 7:  fc.camera_buffer_addr = addr
		case 8:  fc.camera_sparse_map_buffer_addr = addr
		case 9:  fc.tag_sparse_map_buffer_addr = addr
		case 10: fc.tag_data_buffer_addr = addr
		case 11: fc.transform_buffer_addr = addr
		case 12: fc.transform_sparse_map_buffer_addr = addr
		case 13: fc.transform_model_link_buffer_addr = addr
		case 14: fc.static_chunk_data_buffer_addr = addr
		case 15: fc.static_chunk_visible_index_buffer_addr = addr
		case 16: fc.static_chunk_count_buffer_addr = addr
		case 17: fc.model_address_buffer_addr = addr
		case 18: fc.model_buffer_addr = addr
		case 19: fc.model_sparse_map_buffer_addr = addr
		case 20: fc.model_count_buffer_addr = addr
		case 21: fc.model_visible_index_buffer_addr = addr
		case 22: fc.animation_address_buffer_addr = addr
		case 23: fc.animation_buffer_addr = addr
		case 24: fc.animation_sparse_map_buffer_addr = addr
		case 25: fc.material_buffer_addr = addr
		case 26: fc.material_lookup_buffer_addr = addr
		case 27: fc.pipeline_lookup_buffer_addr = addr
		case 28: fc.scene_aabb_buffer_addr = addr
		case 29: fc.morton_keys_buffer_addr = addr
		case 30: fc.morton_values_buffer_addr = addr
		case 31: fc.morton_chunk_data_buffer_addr = addr
		case 32: fc.morton_chunk_indirect_dispatch_addr = addr
		case 33: fc.morton_chunk_indirect_draw_addr = addr
		case 34: fc.morton_chunk_visible_indirect_dispatch_addr = addr
		case 35: fc.morton_chunk_visible_index_buffer_addr = addr
		case 36: fc.morton_chunk_transforms_index_buffer_addr = addr
		}
	}
}

// ---------------------------------------------------------------------------
// Per-frame scalar-state writer.
//
// Called once per frame after the scene extraction finishes. Translates
// the host-side runtime settings + this frame's counts into the
// FrameGlobalContext scalar fields the culling shaders read.
// ---------------------------------------------------------------------------

update_frame_scalar_state :: proc(
	state: ^Frame_Context_State,
	settings: ^GPU_Runtime_Settings,
	main_camera, active_camera: u32,
	traditional_cmd_count, meshlet_cmd_count, indirect_cmd_count: u32,
	static_chunk_count, model_count, all_transforms, static_transforms: u32,
	screen_w, screen_h: f32,
) {
	fc := &state.frame_ctx

	fc.screen_width  = screen_w
	fc.screen_height = screen_h

	fc.main_camera_entity   = main_camera
	fc.active_camera_entity = active_camera

	fc.global_indirect_command_count     = indirect_cmd_count
	fc.global_traditional_commands_count = traditional_cmd_count
	fc.global_meshlet_commands_count     = meshlet_cmd_count

	fc.static_chunk_count = static_chunk_count
	fc.model_count        = model_count

	fc.all_transform_count    = all_transforms
	fc.static_transform_count = static_transforms

	// enable flags: project settings -> cull flags
	enable_flag :: #force_inline proc(b: bool) -> u32 { return b ? 1 : 0 }

	fc.enable_meshlet_cone_culling    = enable_flag(settings.cone_culling)
	fc.enable_chunk_frustum_culling   = enable_flag(settings.frustum_culling)
	fc.enable_model_frustum_culling   = enable_flag(settings.frustum_culling)
	fc.enable_mesh_frustum_culling    = enable_flag(settings.frustum_culling)
	fc.enable_meshlet_frustum_culling = enable_flag(settings.frustum_culling)

	fc.enable_chunk_occlusion_culling   = enable_flag(settings.hiz_occlusion)
	fc.enable_model_occlusion_culling   = enable_flag(settings.hiz_occlusion)
	fc.enable_mesh_occlusion_culling    = enable_flag(settings.hiz_occlusion)
	fc.enable_meshlet_occlusion_culling = enable_flag(settings.hiz_occlusion)
}

// frame_context_advance bumps the per-frame counter.
frame_context_advance :: proc(state: ^Frame_Context_State) {
	state.frame_idx += 1
}

// ---------------------------------------------------------------------------
// Debug-only sanity check. Logs (without aborting) if a culling-critical
// buffer is still missing a device address, which usually means the GPU
// backend has not initialised.
// ---------------------------------------------------------------------------

@(private = "file")
CRITICAL_BUFFERS := []Gpu_Buffer_Kind {
	.Global_Indirect_Command,
	.Global_Indirect_Command_Descriptor,
	.Global_Instance_Index,
	.Transform_Pool,
	.Transform_Sparse_Map,
	.Transform_Model_Link,
	.Model_Pool,
	.Model_Sparse_Map,
	.Model_Address,
	.Material_Pool,
	.Material_Lookup,
	.Camera_Pool,
	.Camera_Sparse_Map,
	.Static_Chunk_Data,
	.Static_Chunk_Count,
	.Static_Chunk_Visible_Index,
}

frame_context_validate :: proc(state: ^Frame_Context_State) -> bool {
	ok := true
	for kind in CRITICAL_BUFFERS {
		if state.buffers[kind].device_addr == 0 {
			log.warnf("[BF_GPU] missing device address for %v", kind)
			ok = false
		}
	}
	return ok
}
