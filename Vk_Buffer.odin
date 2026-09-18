// BF_GPU/Vk_Buffer.odin
//
// Persistent GPU buffer lifecycle for the renderer.
//
// vulkan_buffer_layout_kind enumerates every Gpu_Buffer_Kind the culling
// pipeline needs along with the stride (bytes per element) and the
// initial element capacity. The shader side reads each kind through a
// buffer_reference whose stride matches these values, so a layout drift
// between the Odin struct and the GLSL declaration is a hard error.
//
// vulkan_init_renderer_buffers iterates every Gpu_Buffer_Kind, creates
// a STORAGE_BUFFER + SHADER_DEVICE_ADDRESS VMA allocation, stores the
// handle in VULKAN_BUFFER_MAP (the same map the GPU_Backend vtable
// reaches through), and copies the device address into the matching
// Frame_Context_State entry. refresh_frame_addresses() then copies the
// addresses into the Gpu_Frame_Global_Context mirror that the shaders
// read every frame.
//
// vulkan_upload_scene runs once per frame after gpu_scene_update:
//   - For each pool (transforms, models, materials, cameras, static
//     chunks, ...) the host-side slice is copied into a staging buffer
//     and vkCmdCopyBuffer'd into the persistent buffer. A buffer memory
//     barrier ensures the GPU-side reads see the new data.
//   - Capacity grows trigger a resize: the persistent buffer is
//     destroyed, recreated with the next power-of-two capacity, and the
//     frame address mirror is refreshed.
//   - Dirty ranges (added/updated slots from Render_Scene) are honored
//     where possible; the first upload after a resize copies the full
//     range.

#+feature dynamic-literals
package BF_GPU

import vma "../../dependencies/odin-vma"
import "core:log"
import "core:mem"
import vk "vendor:vulkan"


// Vulkan_Buffer is the backend-side buffer handle. The
// Gpu_Buffer_Handle -> ^Vulkan_Buffer map (VULKAN_BUFFER_MAP in
// Vulkan.odin) reaches these through the GPU_Backend vtable. Persistent
// renderer buffers (per-kind) plus arbitrary user buffers both live
// here.
//
// When `arena != nil` the entry is a suballocation of the named arena:
// `buffer == arena.buffer`, `allocation == arena.allocation`, and
// `device_address == arena.device_address + arena_offset`. The upload
// path issues `vkCmdCopyBuffer(..., arena.buffer, arena_offset, ...)`;
// destroy calls gpu_arena_free instead of vma.DestroyBuffer.
Vulkan_Buffer :: struct {
	buffer:         vk.Buffer,
	allocation:     vma.Allocation,
	size:           vk.DeviceSize,
	device_address: vk.DeviceAddress,
	// Host-side usage flags from the create_asset_buffer call. Stored
	// so the diagnostics layer can attribute per-buffer uploads to
	// the right Upload_Kind (vertex / index / other) without forcing
	// the upload path to plumb the kind through. zero for buffers
	// created via vulkan_create_buffer directly (no host-side kind
	// attribution then).
	usage_flags:    Gpu_Buffer_Usage,
	// Arena backref (nil for buffers with their own VkBuffer).
	arena:          ^GPU_Arena,
	arena_offset:   vk.DeviceSize,
	arena_size:     vk.DeviceSize,
}

Gpu_Buffer_Description :: struct {
	size:         vk.DeviceSize,
	capacity:     u64,
	stride:       u32,
	usage:        vk.BufferUsageFlags,
	memory_usage: vma.MemoryUsage,
}

// ---------------------------------------------------------------------------
// Layout table.
//
// vulkan_buffer_layout returns the per-kind stride + initial capacity.
// The shader side reads each pool as a buffer_reference whose stride
// matches this value; see tests_scene_upload.odin for layout vs shader
// parity assertions.
//
// Per-pool sources (the `kind` -> "what I upload from" mapping):
//
//   Transform_Pool             gpu.transforms.data   (Gpu_Transform_Component)
//   Transform_Sparse_Map       gpu.sparse.transforms.entity_to_dense  (u32)
//   Transform_Model_Link       per (entity, model slot) -> Gpu_Transform_Model_Link
//   Model_Pool                 gpu.models.data
//   Model_Sparse_Map           gpu.sparse.models.entity_to_dense
//   Model_Address              gpu.model_addresses.data
//   Camera_Pool                gpu.cameras.data
//   Camera_Sparse_Map          gpu.sparse.cameras.entity_to_dense
//   Material_Pool              gpu.materials.data
//   Material_Lookup            gpu.material_lookup.data
//   Pipeline_Lookup            gpu.pipeline_lookup.data
//   Static_Chunk_Data          gpu.static_chunks
//
// Buffers the renderer does not yet populate (visibility, indirect,
// counts, morton) get a fixed-size scratch allocation sized to the
// per-frame maxima.
// ---------------------------------------------------------------------------

@(private)
VULKAN_BUFFER_LAYOUT: map[Gpu_Buffer_Kind]Gpu_Buffer_Description

@(private)
vulkan_buffer_layout_init :: proc() {
	// Re-init safe: clear the underlying storage so a repeated
	// dynamic-literal assignment doesn't leak the previous hash map.
	clear(&VULKAN_BUFFER_LAYOUT)
	VULKAN_BUFFER_LAYOUT = {
		.Global_Instance_Index                  = {capacity = 4096, stride = 4},
		.Global_Indirect_Command                = {capacity = 1024, stride = size_of(GPU_Indirect_Command)},
		.Global_Indirect_Command_Descriptor     = {capacity = 1024, stride = 4},
		.Global_Draw_Count                      = {capacity = 1,    stride = 4},

		.Global_Model_Allocation                = {capacity = 1024, stride = size_of(Gpu_Model_Allocation)},
		.Global_Mesh_Allocation                 = {capacity = 4096, stride = size_of(Gpu_Mesh_Allocation)},

		.Camera_Visible_Index                  = {capacity = 64,   stride = 4},
		.Camera_Pool                            = {capacity = 16,   stride = size_of(Gpu_Camera)},
		.Camera_Sparse_Map                      = {capacity = 4096, stride = 4},

		.Transform_Pool                        = {capacity = 4096, stride = size_of(Gpu_Transform_Component)},
		.Transform_Sparse_Map                   = {capacity = 4096, stride = 4},
		.Transform_Model_Link                   = {capacity = 4096, stride = size_of(Gpu_Transform_Model_Link)},

		.Static_Chunk_Data                      = {capacity = 256,  stride = size_of(Gpu_Static_Chunk)},
		.Static_Chunk_Visible_Index             = {capacity = 256,  stride = 4},
		.Static_Chunk_Count                     = {capacity = 1,    stride = 4},

		.Model_Address                         = {capacity = 1024, stride = size_of(Gpu_Model_Addresses)},
		.Model_Pool                             = {capacity = 4096, stride = size_of(Gpu_Model_Component)},
		.Model_Sparse_Map                       = {capacity = 4096, stride = 4},
		.Model_Count                           = {capacity = 1,    stride = 4},
		.Model_Visible_Index                    = {capacity = 4096, stride = 4},

		.Animation_Address                     = {capacity = 64,   stride = size_of(Gpu_Animation_Addresses)},
		.Animation_Pool                         = {capacity = 64,   stride = size_of(Gpu_Animation_Component)},
		.Animation_Sparse_Map                   = {capacity = 4096, stride = 4},

		.Material_Pool                          = {capacity = 1024, stride = size_of(Gpu_Material)},
		.Material_Lookup                        = {capacity = 4096, stride = 4},
		.Pipeline_Lookup                        = {capacity = 4096, stride = 4},

		.Scene_AABB                             = {capacity = 1,    stride = size_of(Gpu_Scene_AABB)},

		.Morton_Keys                            = {capacity = 4096, stride = 4},
		.Morton_Values                          = {capacity = 4096, stride = 4},
		.Morton_Chunk_Data                      = {capacity = 128,  stride = size_of(Gpu_Static_Chunk)},
		.Morton_Chunk_Indirect_Dispatch         = {capacity = 1,    stride = 12},
		.Morton_Chunk_Indirect_Draw             = {capacity = 128,  stride = size_of(GPU_Indirect_Command)},
		.Morton_Chunk_Visible_Indirect_Dispatch = {capacity = 1,    stride = 12},
		.Morton_Chunk_Visible_Index             = {capacity = 128,  stride = 4},
		.Morton_Chunk_Transforms_Index          = {capacity = 1024, stride = 4},

		.Frame_Global_Context                   = {capacity = 1, stride = size_of(Gpu_Frame_Global_Context)},
	}
}

// gpu_buffer_descriptions now returns the real per-kind layout. The
// previous stub returned the same value for every kind and made every
// persistent buffer the wrong size; the renderer could not detect this
// because no persistent buffers were ever created. The shader-side
// stride must match .stride; see tests_scene_upload.odin for the parity
// assertions.
gpu_buffer_descriptions :: proc(kind: Gpu_Buffer_Kind) -> Gpu_Buffer_Description {
	desc, ok := VULKAN_BUFFER_LAYOUT[kind]
	if !ok || desc.stride == 0 {
		// Defensive: a missing entry defaults to 4-byte scratch so a
		// future Gpu_Buffer_Kind addition still produces a usable
		// (if small) buffer instead of silently zeroing size.
		desc = Gpu_Buffer_Description{capacity = 16, stride = 4}
	}
	desc.size = vk.DeviceSize(desc.capacity * u64(desc.stride))
	return desc
}

// Resize the persistent buffer for a single Gpu_Buffer_Kind.
//
//   vbuf, ok := vulkan_resize_buffer(.Transform_Pool, new_capacity_in_elements)
//
// Allocates a new VMA buffer at least new_capacity_in_elements * stride
// bytes (rounded up to a power of two to keep allocator behaviour
// predictable), destroys the old buffer in place (the entry's VMA
// allocation is released synchronously; the caller is responsible for
// GPU-idle guards), and refreshes the matching Frame_Context_State
// entry. Returns the new buffer pair; ok=false leaves the existing
// buffer untouched.
//
// Used by vulkan_upload_scene when GPU pool capacity grows.
vulkan_resize_buffer :: proc(kind: Gpu_Buffer_Kind, new_capacity: u64) -> (Vulkan_Buffer, bool) {
	if !VULKAN_STATE.initialized || VULKAN_STATE.allocator == nil do return {}, false
	desc := gpu_buffer_descriptions(kind)
	stride := u64(desc.stride)
	if stride == 0 do stride = 4

	// Round up to the next power of two so allocator behaviour is
	// predictable and reallocations after small growths don't churn.
	capacity := max(new_capacity, 16)
	if (capacity & (capacity - 1)) != 0 {
		capacity *= 2
		for (capacity & (capacity - 1)) != 0 do capacity += 1
	}

	new_size := vk.DeviceSize(capacity * stride)
	if msg := vulkan_validate_buffer_size(kind, u64(new_size), desc.stride); msg != "" {
		log.errorf("[BF_GPU/Vulkan] resize_buffer rejected: %s", msg)
		return {}, false
	}

	vbuf, ok := vulkan_create_buffer(
		new_size,
		{.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
		.GPU_ONLY,
	)
	if !ok do return {}, false

	// Swap: the caller (vulkan_upload_scene) holds the OLD buffer
	// handle in VULKAN_BUFFER_MAP and updates it after it has flushed
	// the old contents to the new buffer. We just return the new
	// allocation; the swap step lives outside this proc.
	desc.size = new_size
	_ = desc
	return vbuf, true
}

// ---------------------------------------------------------------------------
// Lifecycle.
//
// vulkan_init_renderer_buffers creates every per-kind buffer described
// by VULKAN_BUFFER_LAYOUT, populates the matching Frame_Context_State
// entry, and refreshes the device-address mirror. It is invoked once
// from vulkan_init after VMA is up.
//
// vulkan_destroy_renderer_buffers releases every buffer the function
// above created. Called from vulkan_shutdown with a vkDeviceWaitIdle
// already issued.
// ---------------------------------------------------------------------------

// Dirty_Range is a half-open element range [first, last) inside a
// pool. The upload path records one vkCmdCopyBuffer per range so
// unchanged elements stay GPU-resident.
Dirty_Range :: struct {
	first: u32, // inclusive element index
	last:  u32, // exclusive element index
}

// Per-pool upload cursor. The host-side copy only writes slots
// [0, dense_used), but the GPU buffer is sized to capacity elements.
// On the first upload (last_uploaded_to == 0) we copy the full range;
// subsequent uploads only touch the dirty slots/entities accumulated
// since the previous frame plus the freshly-grown tail
// [last_uploaded_to, dense_used) when the pool extended. On a resize
// the cursor is reset to 0 so the next upload repopulates the new
// buffer from scratch.
//
// dirty_slots is the per-kind source of "this slot was added / updated
// / removed since the previous upload" markers. Transform_Pool and
// Model_Pool use it. dirty_entities is the parallel list for
// entity-indexed pools (Transform_Sparse_Map, Model_Sparse_Map). Each
// cursor only populates the list that matches its pool kind.
//
// ranges is the result of merging / sorting the dirty list plus the
// tail into minimal disjoint Dirty_Range entries after the last
// upload. bytes is the corresponding total bytes for diagnostics.
@(private)
Vulkan_Upload_Cursor :: struct {
	dense_used:      u32, // last GPU-side element count
	capacity:        u64, // current GPU buffer element capacity (== frame.buffers[kind].capacity)
	last_uploaded_to: u32, // exclusive end of the last per-frame range
	dirty_slots:     [dynamic]u32, // slot indices needing upload
	dirty_entities:  [dynamic]u32, // entity indices needing upload
	ranges:          [dynamic]Dirty_Range, // last computed ranges (diagnostics)
	bytes:           u64, // bytes uploaded by the last call (diagnostics)
}

@(private)
VULKAN_UPLOAD_CURSORS: [Gpu_Buffer_Kind.COUNT]Vulkan_Upload_Cursor

// Dirty-range table mirrors the per-frame added/updated/removed sets
// from Render_Scene. The actual upload loop iterates these ranges so
// unchanged slots stay GPU-resident. The cursor above keeps the
// invariant: after each upload last_uploaded_to <= dense_used.
//
// Range table is allocated lazily once Render_Scene starts having
// additions. For the first frame -- when every slot is "added" -- the
// full range is uploaded in one copy.
@(private)
vulkan_init_renderer_buffers :: proc(frame: ^Frame_Context_State) -> bool {
	if frame == nil do return false

	for kind in Gpu_Buffer_Kind {
		if kind == .COUNT {break}
		desc := gpu_buffer_descriptions(kind)

		// Storage + device-address usage. The compute / mesh / vertex
		// pipelines consume these buffers, so SHADER_DEVICE_ADDRESS
		// must be in the creation flags or the culling shaders can't
		// read through buffer_reference.
		vbuf, ok := vulkan_create_buffer(
			desc.size,
			{.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS},
			.GPU_ONLY,
		)
		if !ok {
			log.errorf("[BF_GPU/Vulkan] failed creating persistent buffer %v", kind)
			return false
		}

		handle := Gpu_Buffer_Handle(VULKAN_BUFFER_NEXT_ID)
		VULKAN_BUFFER_NEXT_ID += 1
		entry := new(Vulkan_Buffer)
		entry^ = vbuf
		VULKAN_BUFFER_MAP[handle] = entry

		frame.buffers[kind] = Gpu_Buffer_Entry {
			handle      = handle,
			device_addr = u64(vbuf.device_address),
			size        = u64(desc.size),
			capacity    = desc.capacity,
			stride      = desc.stride,
		}
		VULKAN_UPLOAD_CURSORS[kind] = Vulkan_Upload_Cursor {
			dense_used      = 0,
			capacity        = desc.capacity,
			last_uploaded_to= 0,
			dirty_slots     = make([dynamic]u32, 0, 256, context.allocator),
			dirty_entities  = make([dynamic]u32, 0, 256, context.allocator),
			ranges          = make([dynamic]Dirty_Range, 0, 64, context.allocator),
			bytes           = 0,
		}
	}
	refresh_frame_addresses(frame)
	log.infof("[BF_GPU/Vulkan] initialised %d persistent buffers", int(Gpu_Buffer_Kind.COUNT))
	return true
}

vulkan_destroy_renderer_buffers :: proc(frame: ^Frame_Context_State) {
	// Destroy in reverse order so the upload cursors can be cleared
	// safely (they reference nothing the buffers own).
	for i := int(Gpu_Buffer_Kind.COUNT) - 1; i >= 0; i -= 1 {
		kind := Gpu_Buffer_Kind(i)
		entry_handle := frame.buffers[kind].handle
		if entry_handle == Gpu_Buffer_Handle(0) {continue}
		vulkan_backend_destroy_buffer(entry_handle)
		frame.buffers[kind] = {}
		cursor := &VULKAN_UPLOAD_CURSORS[kind]
		delete(cursor.dirty_slots)
		delete(cursor.dirty_entities)
		delete(cursor.ranges)
		cursor^ = {}
	}
}

// vulkan_mark_slot_dirty records a slot index that needs to be re-uploaded
// for `kind`. The caller (Scene.odin's gpu_scene_update) feeds this from
// Render_Scene's added/updated/removed change sets. Use only for kinds
// whose pool is slot-indexed (Transform_Pool, Model_Pool).
vulkan_mark_slot_dirty :: proc(kind: Gpu_Buffer_Kind, slot: u32) {
	cursor := &VULKAN_UPLOAD_CURSORS[kind]
	if int(kind) < 0 || int(kind) >= int(Gpu_Buffer_Kind.COUNT) do return
	ensure_cursor_dirty_alloc(cursor)
	append(&cursor.dirty_slots, slot)
}

// vulkan_mark_entity_dirty records an entity index that needs to be
// re-uploaded for `kind`. Use only for kinds whose pool is
// entity-indexed (Transform_Sparse_Map, Model_Sparse_Map).
vulkan_mark_entity_dirty :: proc(kind: Gpu_Buffer_Kind, entity: u32) {
	cursor := &VULKAN_UPLOAD_CURSORS[kind]
	if int(kind) < 0 || int(kind) >= int(Gpu_Buffer_Kind.COUNT) do return
	ensure_cursor_dirty_alloc(cursor)
	append(&cursor.dirty_entities, entity)
}

// ensure_cursor_dirty_alloc installs context.allocator on the cursor's
// dirty-tracking dynamic arrays when they were never initialised (the
// test paths and the first-frame-before-vulkan_init path). Odin's
// zero-value dynamic array defaults to context.allocator for `append`,
// but the memory tracker treats the resulting allocation as a leak
// because the underlying call site never recorded a matching free.
// Initialising the allocator explicitly keeps the tracker happy and
// keeps the allocators consistent with the rest of the renderer.
@(private)
ensure_cursor_dirty_alloc :: proc(cursor: ^Vulkan_Upload_Cursor) {
	if cursor.dirty_slots.allocator.procedure == nil {
		cursor.dirty_slots.allocator = context.allocator
	}
	if cursor.dirty_entities.allocator.procedure == nil {
		cursor.dirty_entities.allocator = context.allocator
	}
	if cursor.ranges.allocator.procedure == nil {
		cursor.ranges.allocator = context.allocator
	}
}

// vulkan_reset_dirty clears every dirty mark on a single kind without
// flushing anything. Used by tests + the asset-upload path when a pool
// is wholesale-repopulated outside the per-frame change sets.
vulkan_reset_dirty :: proc(kind: Gpu_Buffer_Kind) {
	cursor := &VULKAN_UPLOAD_CURSORS[kind]
	clear(&cursor.dirty_slots)
	clear(&cursor.dirty_entities)
	clear(&cursor.ranges)
	cursor.bytes = 0
}

// vulkan_last_upload_ranges returns the most recent ranges[] the upload
// path emitted for `kind`. Empty when the last upload was a no-op.
vulkan_last_upload_ranges :: proc(kind: Gpu_Buffer_Kind) -> []Dirty_Range {
	cursor := &VULKAN_UPLOAD_CURSORS[kind]
	return cursor.ranges[:]
}

// vulkan_last_upload_bytes returns the byte count of the most recent
// upload for `kind`.
vulkan_last_upload_bytes :: proc(kind: Gpu_Buffer_Kind) -> u64 {
	return VULKAN_UPLOAD_CURSORS[kind].bytes
}

// ---------------------------------------------------------------------------
// Per-frame scene upload.
//
// vulkan_upload_scene runs once per frame after gpu_scene_update. It
// walks the CPU pools (transforms, models, materials, cameras, ...),
// records vkCmdCopyBuffer commands into the current frame's command
// buffer, and inserts a buffer memory barrier so the GPU-side culling
// shaders see the new data.
//
// The command buffer must already be in the recording state; caller
// (vulkan_frame) is responsible for begin / end / submit.
//
// Capacity-grows trigger vulkan_resize_buffer; a resize resets the
// cursor so the next upload copies the full range.
// ---------------------------------------------------------------------------

// vulkan_record_pool_upload records a copy of `data[0..count)` into
// the persistent `kind` buffer. Returns false on resize failure (the
// caller is expected to abort and retry next frame).
vulkan_record_pool_upload :: proc(
	cmd_buffer: vk.CommandBuffer,
	frame: ^Frame_Context_State,
	kind: Gpu_Buffer_Kind,
	data: rawptr,
	count: u32,
	stride: u32,
) -> bool {
	if cmd_buffer == nil || data == nil || count == 0 || stride == 0 do return true
	if frame == nil do return false

	cursor := &VULKAN_UPLOAD_CURSORS[kind]
	cursor.dense_used = max(cursor.dense_used, count)

	// Resize if the persistent buffer no longer fits. The new buffer
	// is created GPU_ONLY; we copy from a fresh staging buffer so the
	// old contents aren't required to survive.
	if u64(count) > cursor.capacity {
		new_capacity := u64(count) * 2
		vbuf, ok := vulkan_resize_buffer(kind, new_capacity)
		if !ok do return false

		// Tear down the old buffer through the deferred-destruction
		// queue so any in-flight GPU work (a frame N-1 still using
		// the old device address through its frame mirror) is allowed
		// to retire before VMA releases the underlying VkDeviceMemory.
		// The previous path synchronously destroyed, which was a
		// use-after-free hazard on resize-during-render. Tag with the
		// current graphics_timeline_value: vulkan_collect_garbage()
		// reaps everything <= the value the GPU has reached, and the
		// next frame submit signals that value.
		old_handle := frame.buffers[kind].handle
		old_entry := VULKAN_BUFFER_MAP[old_handle]
		if old_entry != nil {
			delete_key(&VULKAN_BUFFER_MAP, old_handle)
			vulkan_defer_buffer_destruction(
				old_entry,
				VULKAN_STATE.graphics_timeline_value,
			)
		}

		new_handle := Gpu_Buffer_Handle(VULKAN_BUFFER_NEXT_ID)
		VULKAN_BUFFER_NEXT_ID += 1
		new_entry := new(Vulkan_Buffer)
		new_entry^ = vbuf
		VULKAN_BUFFER_MAP[new_handle] = new_entry

		frame.buffers[kind] = Gpu_Buffer_Entry {
			handle      = new_handle,
			device_addr = u64(vbuf.device_address),
			size        = u64(vbuf.size),
			capacity    = cursor.capacity + (cursor.capacity >> 1), // approx; real value lives in resizer
			stride      = stride,
		}
		// The resizer records the real capacity via the buffer entry,
		// not the cursor. Sync the cursor so subsequent resize checks
		// use the correct floor.
		cursor.capacity = u64(count) * 2
		cursor.last_uploaded_to = 0 // full-range copy next
		refresh_frame_addresses(frame)
		return vulkan_record_pool_upload(cmd_buffer, frame, kind, data, count, stride)
	}

	// Build the dirty range set. Three sources contribute:
	//
	//   1. Dirty slots the host side marked via vulkan_mark_slot_dirty
	//      (slot-parallel pools: Transform_Pool, Model_Pool).
	//   2. Dirty entities the host side marked via
	//      vulkan_mark_entity_dirty (entity-indexed pools:
	//      Transform_Sparse_Map, Model_Sparse_Map).
	//   3. The freshly-grown tail [last_uploaded_to, count), so newly
	//      reserved memory is uploaded with at least its zero state.
	//
	// First-frame uploads have last_uploaded_to == 0; the tail covers
	// the whole pool and the dirty marks contribute zero extra bytes.
	ranges_buf: [dynamic]Dirty_Range
	ranges_buf.allocator = context.allocator
	defer delete(ranges_buf)

	if count > cursor.last_uploaded_to {
		append(
			&ranges_buf,
			Dirty_Range{first = cursor.last_uploaded_to, last = count},
		)
	}

	slot_or_entity_dirty := false
	if len(cursor.dirty_slots) > 0 {
		slot_or_entity_dirty = true
		append_ranges_from_dirty_indices(&ranges_buf, cursor.dirty_slots[:])
	}
	if len(cursor.dirty_entities) > 0 {
		slot_or_entity_dirty = true
		append_ranges_from_dirty_indices(&ranges_buf, cursor.dirty_entities[:])
	}
	_ = slot_or_entity_dirty

	// Cap to the live pool so a stale slot index in the dirty list
	// (e.g. a removal that landed on a slot the cursor has already
	// forgotten about) cannot push us past the buffer.
	for &r in ranges_buf {
		if r.last > count do r.last = count
	}

	if len(ranges_buf) > 1 {
		merge_overlapping_ranges(&ranges_buf)
	}

	if len(ranges_buf) == 0 {
		// Nothing changed this frame. Clear dirty markers (they may
		// have been emptied by the cap pass) and exit without a copy
		// or a barrier.
		clear(&cursor.dirty_slots)
		clear(&cursor.dirty_entities)
		clear(&cursor.ranges)
		cursor.bytes = 0
		cursor.last_uploaded_to = max(cursor.last_uploaded_to, count)
		frame.buffers[kind].capacity = cursor.capacity
		return true
	}

	dst_entry := VULKAN_BUFFER_MAP[frame.buffers[kind].handle]
	if dst_entry == nil || dst_entry.buffer == {} {
		log.errorf("[BF_GPU/Vulkan] missing destination buffer for %v", kind)
		return false
	}

	// One (buffer, base_offset) pair per range. Each range produces one
	// vkCmdCopyBuffer so we keep the granularity of "dirty" at the
	// range level instead of staging the whole pool.
	total_bytes: u64 = 0
	for r in ranges_buf {
		bc := u64(r.last - r.first) * u64(stride)
		if bc == 0 do continue
		total_bytes += bc

		ring_size := align_up_u64(bc, 4)
		ring_slot := u32(VULKAN_STATE.frame_index)
		src_buffer, src_offset, src_ptr, ring_ok := vulkan_upload_ring_alloc(
			ring_slot,
			ring_size,
			4,
		)
		if ring_ok {
			src_base := rawptr(uintptr(data) + uintptr(r.first) * uintptr(stride))
			mem.copy(src_ptr, src_base, int(bc))
		} else {
			// Cold path: allocate a one-shot staging buffer for this
			// range only (not the whole pool).
			src_base := rawptr(uintptr(data) + uintptr(r.first) * uintptr(stride))
			staging, staging_alloc, staging_ok := vulkan_create_staging_buffer(
				src_base,
				bc,
			)
			if !staging_ok do return false
			defer vma.DestroyBuffer(VULKAN_STATE.allocator, staging, staging_alloc)
			src_buffer = staging
			src_offset = 0
		}

		copy := vk.BufferCopy {
			srcOffset = src_offset,
			dstOffset = vk.DeviceSize(u64(r.first) * u64(stride)),
			size      = vk.DeviceSize(bc),
		}
		vk.CmdCopyBuffer(cmd_buffer, src_buffer, dst_entry.buffer, 1, &copy)
	}

	// Single combined barrier covering every dirty range. The whole
	// dirty-set is treated as one dependency edge for the downstream
	// compute / vertex stages - the barrier cost is paid once per pool
	// per frame regardless of how many ranges participated.
	barrier := vk.BufferMemoryBarrier2 {
		sType               = .BUFFER_MEMORY_BARRIER_2,
		srcStageMask        = {.COPY},
		srcAccessMask       = {.TRANSFER_WRITE},
		dstStageMask        = {.COMPUTE_SHADER, .VERTEX_SHADER, .FRAGMENT_SHADER},
		dstAccessMask       = {.SHADER_READ},
		srcQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		dstQueueFamilyIndex = vk.QUEUE_FAMILY_IGNORED,
		buffer              = dst_entry.buffer,
		offset              = 0,
		size                = vk.DeviceSize(u64(count) * u64(stride)),
	}
	dep := vk.DependencyInfo {
		sType                    = .DEPENDENCY_INFO,
		bufferMemoryBarrierCount = 1,
		pBufferMemoryBarriers    = &barrier,
	}
	vk.CmdPipelineBarrier2(cmd_buffer, &dep)

	// Diagnostics + cursor housekeeping.
	clear(&cursor.ranges)
	for r in ranges_buf do append(&cursor.ranges, r)
	cursor.bytes = total_bytes
	cursor.last_uploaded_to = max(cursor.last_uploaded_to, count)
	clear(&cursor.dirty_slots)
	clear(&cursor.dirty_entities)

	// Update live metric on the entry; reflect the current dense count.
	frame.buffers[kind].capacity = cursor.capacity
	_ = slot_or_entity_dirty
	diag_record_upload_bandwidth_with_kind(
		.Frame_Transient,
		MODULE_STATE_VALUE.frame_ctx.frame_idx,
		total_bytes,
	)
	return true
}

// vulkan_record_scalar_upload records a copy of a single 32-bit value
// into a small GPU buffer (count = 1). Used for the *_Count buffers
// whose capacity is a single slot.
vulkan_record_scalar_upload :: proc(
	cmd_buffer: vk.CommandBuffer,
	frame: ^Frame_Context_State,
	kind: Gpu_Buffer_Kind,
	value: u32,
) -> bool {
	local := value
	return vulkan_record_pool_upload(
		cmd_buffer,
		frame,
		kind,
		&local,
		1,
		4,
	)
}

// vulkan_upload_scene flushes every populated pool from the Render_Scene
// mirror into the persistent GPU buffers. The command buffer must
// already be recording; the caller passes the per-frame Frame_Context_State
// so the upload can refresh the address mirror on resize.
//
// Returns false on any allocation failure (caller treats that as a
// soft error: log + skip this frame).
vulkan_upload_scene :: proc(
	cmd_buffer: vk.CommandBuffer,
	frame: ^Frame_Context_State,
	gpu: ^GPU_Scene,
) -> bool {
	if cmd_buffer == nil || frame == nil || gpu == nil do return false
	if !VULKAN_STATE.initialized do return false

	// Snapshot every arena's used / free into the diagnostics layer.
	// Cheap; one record per arena. Does not change command-buffer
	// contents.
	gpu_arena_record_used_to_diagnostics()

	ok := true

	// Transform pool + sparse map.
	if len(gpu.transforms.data) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Transform_Pool,
			raw_data(gpu.transforms.data),
			u32(len(gpu.transforms.data)),
			size_of(Gpu_Transform_Component),
		)
	}
	if len(gpu.sparse.transforms.entity_to_dense) > 0 {
		// Sparse maps hold a u32 per entity id, sized to the entity
		// table. We upload the full slice every frame; the shader
		// side treats slots >= entity_count as empty.
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Transform_Sparse_Map,
			raw_data(gpu.sparse.transforms.entity_to_dense),
			u32(len(gpu.sparse.transforms.entity_to_dense)),
			4,
		)
	}

	// Model pool + sparse map.
	if len(gpu.models.data) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Model_Pool,
			raw_data(gpu.models.data),
			u32(len(gpu.models.data)),
			size_of(Gpu_Model_Component),
		)
	}
	if len(gpu.sparse.models.entity_to_dense) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Model_Sparse_Map,
			raw_data(gpu.sparse.models.entity_to_dense),
			u32(len(gpu.sparse.models.entity_to_dense)),
			4,
		)
	}

	// Cameras.
	if len(gpu.cameras.data) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Camera_Pool,
			raw_data(gpu.cameras.data),
			u32(len(gpu.cameras.data)),
			size_of(Gpu_Camera),
		)
	}
	if len(gpu.sparse.cameras.entity_to_dense) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Camera_Sparse_Map,
			raw_data(gpu.sparse.cameras.entity_to_dense),
			u32(len(gpu.sparse.cameras.entity_to_dense)),
			4,
		)
	}

	// Static chunks. Re-uploaded wholesale; chunks are small (<= 256
	// entries) so the cost is bounded.
	if len(gpu.static_chunks) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Static_Chunk_Data,
			raw_data(gpu.static_chunks),
			u32(len(gpu.static_chunks)),
			size_of(Gpu_Static_Chunk),
		)
	}
	ok &= vulkan_record_scalar_upload(
		cmd_buffer,
		frame,
		.Static_Chunk_Count,
		u32(len(gpu.static_chunks)),
	)
	ok &= vulkan_record_scalar_upload(
		cmd_buffer,
		frame,
		.Model_Count,
		gpu.frame_model_count,
	)

	// Materials / pipeline lookup.
	if len(gpu.materials.data) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Material_Pool,
			raw_data(gpu.materials.data),
			u32(len(gpu.materials.data)),
			size_of(Gpu_Material),
		)
	}
	if len(gpu.material_lookup.data) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Material_Lookup,
			raw_data(gpu.material_lookup.data),
			u32(len(gpu.material_lookup.data)),
			4,
		)
	}
	if len(gpu.pipeline_lookup.data) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Pipeline_Lookup,
			raw_data(gpu.pipeline_lookup.data),
			u32(len(gpu.pipeline_lookup.data)),
			4,
		)
	}

	// Model addresses. The (entity) sparse + slot index already live in
	// the model sparse map; the addresses buffer carries the per-model
	// pointer slab populated when the asset upload resolved.
	if len(gpu.model_addresses.data) > 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Model_Address,
			raw_data(gpu.model_addresses.data),
			u32(len(gpu.model_addresses.data)),
			size_of(Gpu_Model_Addresses),
		)
	}

	// Frame global context. Single-element SSBO the culling / HiZ /
	// shading shaders read every frame through push constant
	// (pc.frameGlobalContextBufferAddr). Always uploaded when its
	// buffer exists so a zero-element frame still gets a valid
	// (zeroed) GPU-side context, which keeps the address stable.
	if frame.buffers[Gpu_Buffer_Kind.Frame_Global_Context].handle != 0 {
		ok &= vulkan_record_pool_upload(
			cmd_buffer,
			frame,
			.Frame_Global_Context,
			&frame.frame_ctx,
			1,
			size_of(Gpu_Frame_Global_Context),
		)
	}

	return ok
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// vulkan_begin_command_buffer / vulkan_end_command_buffer are the
// minimal command buffer lifecycle the upload + render passes share.
// Kept here so Vk_Frame.odin can import them without dragging the
// layout table along.

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

// ---------------------------------------------------------------------------
// Dirty-range helpers.
//
// append_ranges_from_dirty_indices consumes an unsorted, deduplicated
// list of u32 indices (slot or entity) and pushes one Dirty_Range per
// element. Sorting and merging into disjoint ranges happens later in
// merge_overlapping_ranges; this proc just translates the "dirty set"
// into a flat range list the merge step can reason about.
//
// merge_overlapping_ranges sorts + dedupes a range list in place and
// coalesces any touching or overlapping ranges into the minimal
// disjoint set. Stable on already-sorted input (O(N)) in that case.
// ---------------------------------------------------------------------------

@(private)
append_ranges_from_dirty_indices :: proc(out: ^[dynamic]Dirty_Range, dirty: []u32) {
	for slot in dirty {
		append(out, Dirty_Range{first = slot, last = slot + 1})
	}
}

@(private)
merge_overlapping_ranges :: proc(ranges: ^[dynamic]Dirty_Range) {
	if len(ranges) <= 1 do return

	// Sort by .first.
	sort_dirty_ranges(ranges)

	// Coalesce.
	write := 1
	for read in 1 ..< len(ranges) {
		prev := &ranges[write - 1]
		cur := &ranges[read]
		if cur.first <= prev.last {
			if cur.last > prev.last do prev.last = cur.last
		} else {
			if write != read do ranges[write] = cur^
			write += 1
		}
	}

	// Trim.
	resize(ranges, write)
}

@(private)
sort_dirty_ranges :: proc(ranges: ^[dynamic]Dirty_Range) {
	// Simple insertion sort. Range lists per kind stay small (one entry
	// per dirty element at most, typically O(hundreds) for a typical
	// frame) so the O(N^2) worst case is acceptable here; pulling in
	// the stdlib sort for one hot path is not worth the dependency.
	n := len(ranges)
	for i in 1 ..< n {
		key := ranges[i]
		j := i - 1
		for j >= 0 && ranges[j].first > key.first {
			ranges[j + 1] = ranges[j]
			j -= 1
		}
		ranges[j + 1] = key
	}
}

// ---------------------------------------------------------------------------
// Persistent upload ring.
//
// The previous vulkan_create_staging_buffer path allocated a fresh
// HOST_VISIBLE buffer per upload and immediately unmapped + destroyed it
// after the CPU-side copy. Per-frame scene uploads trigger ~10 such
// allocations, which thrashes the VMA sub-allocator and burns CPU time
// on vkMapMemory / vkUnmapMemory round-trips.
//
// The ring replaces that path with one persistently-mapped buffer of
// `slot_count * slot_size` bytes (defaults: MAX_FRAMES_IN_FLIGHT slots at
// 16 MiB each = 32 MiB on a 2-FIF setup). Each frame owns one slot; the
// cursor for that slot is reset at the top of the frame and is consumed
// by the GPU MAX_FRAMES_IN_FLIGHT frames later, so a slot is safe to
// overwrite when the index wraps.
//
// Allocations come out as a (buffer, offset, mapped_ptr) tuple the caller
// passes to vkCmdCopyBuffer without any per-call allocation. The fallback
// to transient staging is preserved for paths that genuinely need a
// dedicated buffer (e.g. uploads that must outlive the next frame).
// ---------------------------------------------------------------------------

VULKAN_UPLOAD_RING_DEFAULT_SLOT_BYTES :: vk.DeviceSize(16 * 1024 * 1024) // 16 MiB / frame

Vulkan_Upload_Ring :: struct {
	buffer:     vk.Buffer,
	allocation: vma.Allocation,
	mapped:     rawptr,
	total_size: vk.DeviceSize,
	slot_size:  vk.DeviceSize,
	slot_count: u32,
	cursors:    []vk.DeviceSize, // per-slot head cursor (bytes used)
}

@(private)
VULKAN_UPLOAD_RING: Vulkan_Upload_Ring

vulkan_upload_ring_init :: proc(slot_size: vk.DeviceSize, slot_count: u32) -> bool {
	if VULKAN_STATE.allocator == nil do return false
	slot_size_eff := slot_size
	slot_count_eff := slot_count
	if slot_size_eff == 0 do slot_size_eff = VULKAN_UPLOAD_RING_DEFAULT_SLOT_BYTES
	if slot_count_eff == 0 do slot_count_eff = MAX_FRAMES_IN_FLIGHT

	ring := &VULKAN_UPLOAD_RING
	ring.slot_size = slot_size_eff
	ring.slot_count = slot_count_eff
	ring.total_size = slot_size_eff * vk.DeviceSize(slot_count_eff)

	create_info := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = ring.total_size,
		usage       = {.TRANSFER_SRC},
		sharingMode = .EXCLUSIVE,
	}
	alloc_info := vma.AllocationCreateInfo {
		usage         = .CPU_ONLY,
		requiredFlags = {.HOST_VISIBLE, .HOST_COHERENT},
		flags         = {.MAPPED, .HOST_ACCESS_SEQUENTIAL_WRITE},
	}
	result := vma.CreateBuffer(
		VULKAN_STATE.allocator,
		create_info,
		alloc_info,
		&ring.buffer,
		&ring.allocation,
		nil,
	)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] upload ring create failed: %v", result)
		return false
	}
	// SEQUENTIAL_WRITE+MAPPED maps at create time; double-check the
	// pointer landed on a usable address.
	if ring.mapped == nil {
		map_result := vma.MapMemory(VULKAN_STATE.allocator, ring.allocation, &ring.mapped)
		if map_result != .SUCCESS || ring.mapped == nil {
			log.errorf("[BF_GPU/Vulkan] upload ring map failed: %v", map_result)
			vma.DestroyBuffer(VULKAN_STATE.allocator, ring.buffer, ring.allocation)
			ring^ = {}
			return false
		}
	}

	ring.cursors = make([]vk.DeviceSize, slot_count_eff)
	for i in 0 ..< slot_count_eff {
		ring.cursors[i] = 0
	}
	log.infof(
		"[BF_GPU/Vulkan] upload ring ready: %d slots x %d MiB = %d MiB",
		slot_count_eff,
		int(slot_size_eff / (1024 * 1024)),
		int(ring.total_size / (1024 * 1024)),
	)
	return true
}

vulkan_upload_ring_shutdown :: proc() {
	ring := &VULKAN_UPLOAD_RING
	if ring.allocation != nil {
		vma.DestroyBuffer(VULKAN_STATE.allocator, ring.buffer, ring.allocation)
	}
	delete(ring.cursors)
	ring^ = {}
}

// vulkan_upload_ring_reset_slot zeroes the cursor for `slot`, marking
// the slot's previous contents as safe to overwrite. The renderer calls
// this at the top of each frame with `slot = frame_index`. The
// invariant (frame N can safely overwrite slot[N mod slot_count]) holds
// because the GPU is at most MAX_FRAMES_IN_FLIGHT frames behind and
// `slot_count` matches MAX_FRAMES_IN_FLIGHT.
vulkan_upload_ring_reset_slot :: proc(slot: u32) {
	ring := &VULKAN_UPLOAD_RING
	if slot >= ring.slot_count do return
	ring.cursors[slot] = 0
}

// vulkan_upload_ring_alloc reserves `size` bytes at `alignment` within
// `slot` and returns the buffer, the device-side offset (use this as the
// srcOffset of vkCmdCopyBuffer), and the mapped CPU pointer to write
// through. ok=false means the slot is exhausted; callers fall back to
// vulkan_create_staging_buffer for that allocation.
vulkan_upload_ring_alloc :: proc(
	slot: u32,
	size: u64,
	alignment: u64,
) -> (
	buffer: vk.Buffer,
	offset: vk.DeviceSize,
	ptr: rawptr,
	ok: bool,
) {
	ring := &VULKAN_UPLOAD_RING
	if size == 0 do return {}, 0, nil, false
	if slot >= ring.slot_count do return {}, 0, nil, false
	if ring.buffer == {} || ring.mapped == nil do return {}, 0, nil, false

	cursor := &ring.cursors[slot]
	cursor_val := u64(cursor^)
	aligned := align_up_u64(cursor_val, alignment)
	if aligned + size > u64(ring.slot_size) do return {}, 0, nil, false

	cursor^ = vk.DeviceSize(aligned + size)
	slot_base := vk.DeviceSize(slot) * ring.slot_size
	return ring.buffer, slot_base + vk.DeviceSize(aligned), rawptr(uintptr(ring.mapped) + uintptr(slot_base) + uintptr(aligned)), true
}

// align_up_u64 rounds `value` up to the next multiple of `alignment`.
// alignment must be a power of two; matches the alignment requirements
// every Vulkan buffer / copy obeys on the target hardware.
@(private)
align_up_u64 :: proc(value, alignment: u64) -> u64 {
	if alignment <= 1 do return value
	return (value + alignment - 1) & ~(alignment - 1)
}

// vulkan_create_staging_buffer remains as the overflow path for
// allocations larger than the ring slot, or for callers that need a
// dedicated source buffer. The persistent upload ring is the fast path;
// this is the cold path.
vulkan_create_staging_buffer :: proc(data: rawptr, size: u64) -> (vk.Buffer, vma.Allocation, bool) {
	if size == 0 || data == nil do return {}, {}, false
	create_info := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = vk.DeviceSize(size),
		usage       = {.TRANSFER_SRC},
		sharingMode = .EXCLUSIVE,
	}
	alloc_info := vma.AllocationCreateInfo {
		usage         = .CPU_ONLY,
		requiredFlags = {.HOST_VISIBLE, .HOST_COHERENT},
	}
	buffer: vk.Buffer = {}
	allocation: vma.Allocation = {}
	result := vma.CreateBuffer(VULKAN_STATE.allocator, create_info, alloc_info, &buffer, &allocation, nil)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] staging buffer create failed: %v", result)
		return {}, {}, false
	}
	mapped: rawptr
	map_result := vma.MapMemory(VULKAN_STATE.allocator, allocation, &mapped)
	if map_result != .SUCCESS || mapped == nil {
		vma.DestroyBuffer(VULKAN_STATE.allocator, buffer, allocation)
		log.errorf("[BF_GPU/Vulkan] staging buffer map failed: %v", map_result)
		return {}, {}, false
	}
	mem.copy(mapped, data, int(size))
	vma.UnmapMemory(VULKAN_STATE.allocator, allocation)
	return buffer, allocation, true
}

// vulkan_create_buffer is a thin wrapper over vma.CreateBuffer that
// captures the device address into the returned Vulkan_Buffer. The
// validate step is shared with the GPU_Backend vtable.
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

// vulkan_destroy_buffer_now releases a Vulkan_Buffer's VMA allocation
// immediately. Used by the deferred-destruction queue (Vk_Resource.odin)
// and by shutdown paths where the caller has already verified the GPU
// is idle. Safe to call with a zeroed Vulkan_Buffer.
vulkan_destroy_buffer_now :: proc(buf: ^Vulkan_Buffer) {
	if buf == nil do return
	if buf.buffer != {} && VULKAN_STATE.allocator != nil {
		vma.DestroyBuffer(VULKAN_STATE.allocator, buf.buffer, buf.allocation)
	}
	buf^ = {}
}
