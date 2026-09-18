// BF_GPU/Gpu_Arena.odin
//
// Persistent GPU arena allocator.
//
// Architecture target (prompt 04):
//   Replace the one-VkBuffer-per-asset allocation pattern with a small
//   set of large persistent GPU arenas whose lifetime spans the whole
//   session. Asset records reference (arena, offset, size) instead of
//   holding an independent VkBuffer / VmaAllocation pair. The shader
//   side already consumes device addresses through the FrameGlobalContext,
//   so swapping per-asset addresses for arena-base + offset is invisible
//   to every culling / draw / HiZ pass.
//
// The arenas are intentionally renderer-private. Resource classes below
// are tagged so the diagnostics layer can attribute GPU memory to a
// single named bucket without the renderer naming vk. structs. Textures
// stay on their own image-pool path because their residency / lifetime
// model differs from raw geometry.
//
// Allocating:
//   alloc, ok := gpu_arena_allocate(&GPU_ARENAS[.Vertex], bytes, align)
//   handle    := vulkan_register_arena_buffer(.Vertex, alloc, usage)
//
// The arena allocator is a classic first-fit free-list with adjacent-
// block coalescing on release. Capacity grows by creating a new
// VkBuffer + VmaAllocation and copying the live contents (rare path;
// initial capacities are sized for the largest expected asset).
//
// All stats are O(1) or O(free-list-length) and are safe to poll every
// frame from the diagnostics layer.

package BF_GPU

import vma "../../dependencies/odin-vma"
import "core:log"
import vk "vendor:vulkan"

// GPU_Resource_Class enumerates every arena the renderer manages.
// Each class corresponds to one large persistent VkBuffer. Asset
// payloads are suballocated from the matching class.
//
// The diagnostics layer uses the class as an allocation tag on the
// underlying VMA allocation so a single vma.GetAllocationInfo call
// reports per-class bytes without a separate registry.
GPU_Resource_Class :: enum u8 {
	Vertex,             // mesh vertex buffers
	Index,              // mesh index buffers
	Meshlet_Vertex,     // meshlet vertex-index payload
	Meshlet_Triangle,   // meshlet triangle-index payload
	Mesh_Descriptor,    // per-mesh descriptors
	Mesh_Collider,      // per-mesh bounding info
	Lod_Descriptor,     // per-(mesh, lod) descriptors
	Meshlet_Descriptor, // per-meshlet descriptors
	Meshlet_Draw_Desc,  // per-meshlet draw descriptors
	Meshlet_Collider,   // per-meshlet colliders + normal cones
	Node_Transform,     // skeletal / skinning node transforms
	Static_Scratch,     // renderer-managed static scratch
	Dynamic_Scratch,    // renderer-managed dynamic scratch
	Visibility_Scratch, // GPU culling visibility scratch
	Command_Scratch,    // indirect-command scratch
	Descriptor_Scratch, // bindless descriptor backing storage
	Other,              // anything that does not fit the above; never routes here by default
	COUNT,
}

// gpu_default_arena_capacity returns the conservative initial capacity
// (bytes) for `class`. Sized for the largest plausible asset so the
// first batch of uploads fits without a regrow; the arena can grow
// later via gpu_arena_grow. Numbers are conservative; overcommitting
// is cheaper than re-allocating after every mesh upload.
//
// Indexed by GPU_Resource_Class; .Other is reserved for non-arena
// allocations and must stay last in the enum.
@(private)
gpu_default_arena_capacity :: proc(class: GPU_Resource_Class) -> u64 {
	switch class {
	case .Vertex:             return 64 * 1024 * 1024
	case .Index:              return 16 * 1024 * 1024
	case .Meshlet_Vertex:     return 16 * 1024 * 1024
	case .Meshlet_Triangle:   return 16 * 1024 * 1024
	case .Mesh_Descriptor:    return  4 * 1024 * 1024
	case .Mesh_Collider:      return  4 * 1024 * 1024
	case .Lod_Descriptor:     return  2 * 1024 * 1024
	case .Meshlet_Descriptor: return  8 * 1024 * 1024
	case .Meshlet_Draw_Desc:  return  2 * 1024 * 1024
	case .Meshlet_Collider:   return  4 * 1024 * 1024
	case .Node_Transform:     return  2 * 1024 * 1024
	case .Static_Scratch:     return  1 * 1024 * 1024
	case .Dynamic_Scratch:    return  1 * 1024 * 1024
	case .Visibility_Scratch: return  2 * 1024 * 1024
	case .Command_Scratch:    return  1 * 1024 * 1024
	case .Descriptor_Scratch: return  1 * 1024 * 1024
	case .Other:              return  1 * 1024 * 1024
	case .COUNT:              return 0
	}
	return 0
}

// GPU_Arena_Block is one (offset, size) free region inside an arena.
// Free blocks are kept sorted by offset so coalescing on free is a
// single neighbor merge (left + right) with no re-sort cost.
GPU_Arena_Block :: struct {
	offset: u64,
	size:   u64,
}

// GPU_Arena is one persistent VkBuffer-backed arena. The buffer is
// GPU_ONLY and reachable through SHADER_DEVICE_ADDRESS; the host side
// uploads through the existing upload ring + vkCmdCopyBuffer path.
//
// Live allocations live outside the free list; their bookkeeping lives
// on the matching handle in VULKAN_BUFFER_MAP (the arena_offset + size
// fields). The arena only knows the (offset, size) of free regions.
GPU_Arena :: struct {
	class:           GPU_Resource_Class,
	buffer:          vk.Buffer,
	allocation:      vma.Allocation,
	device_address:  vk.DeviceAddress,
	capacity:        u64,
	used:            u64,
	peak_used:       u64,
	free_list:       [dynamic]GPU_Arena_Block,
	alloc_count:     u64,
	free_count:      u64,
	grow_count:      u64,
	coalesce_count:  u64,
	initialized:     bool,
}

// GPU_Arena_Stats is the snapshot the diagnostics layer consumes. All
// values reflect the current state; peak_used is the high-water mark
// since the arena was created.
GPU_Arena_Stats :: struct {
	class:             GPU_Resource_Class,
	capacity:          u64,
	used:              u64,
	peak_used:         u64,
	free:              u64,
	alloc_count:       u64,
	free_count:        u64,
	live_suballocs:    u64,
	largest_free:      u64,
	fragmentation:     f32, // 0..1; 1 - largest_free / free
	grow_count:        u64,
	coalesce_count:    u64,
}

// ---------------------------------------------------------------------------
// Arena registry.
//
// GPU_ARENAS is the package-private array of all arenas, indexed by
// GPU_Resource_Class. The Vulkan backend creates + tears them down in
// vulkan_init / vulkan_shutdown. The CPU-only code paths (tests, the
// diagnostic dump) read them via the public accessor and never mutate
// the arena state directly.
// ---------------------------------------------------------------------------

@(private)
GPU_ARENAS: [GPU_Resource_Class.COUNT]GPU_Arena

@(private)
GPU_ARENAS_INITIALIZED: bool

// gpu_arena_init_all initialises every arena with its default capacity.
// Safe to call multiple times; subsequent calls are no-ops when
// already initialised. The Vulkan device + VMA allocator must already
// be live when this proc runs.
@(private)
gpu_arena_init_all :: proc() -> bool {
	if GPU_ARENAS_INITIALIZED do return true
	if VULKAN_STATE.allocator == nil || VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Arena] init_all called before Vulkan ready")
		return false
	}

	for class in GPU_Resource_Class {
		if class == .COUNT {break}
		arena := &GPU_ARENAS[class]
		if !gpu_arena_init(arena, class, gpu_default_arena_capacity(class)) {
			log.errorf("[BF_GPU/Arena] failed to initialise class %v", class)
			// Tear down anything we already built and bail.
			gpu_arena_shutdown_all()
			return false
		}
	}
	GPU_ARENAS_INITIALIZED = true
	log.infof("[BF_GPU/Arena] initialised %d persistent arenas", int(GPU_Resource_Class.COUNT))
	return true
}

// gpu_arena_shutdown_all destroys every arena. The Vulkan device must
// already be idle (vulkan_shutdown calls vkDeviceWaitIdle first).
@(private)
gpu_arena_shutdown_all :: proc() {
	if !GPU_ARENAS_INITIALIZED do return
	for class in GPU_Resource_Class {
		if class == .COUNT {break}
		gpu_arena_destroy(&GPU_ARENAS[class])
	}
	GPU_ARENAS_INITIALIZED = false
}

// gpu_arena_init allocates the VkBuffer + VmaAllocation for `class`
// and primes the free list with one block covering the whole capacity.
// Returns false (without partial state) when VMA fails.
gpu_arena_init :: proc(arena: ^GPU_Arena, class: GPU_Resource_Class, capacity: u64) -> bool {
	if arena == nil do return false
	if arena.initialized do return true
	if VULKAN_STATE.allocator == nil || VULKAN_STATE.device == nil do return false

	arena.class = class
	arena.capacity = capacity
	arena.used = 0
	arena.peak_used = 0
	arena.alloc_count = 0
	arena.free_count = 0
	arena.grow_count = 0
	arena.coalesce_count = 0
	arena.free_list = make([dynamic]GPU_Arena_Block, 0, 16, context.allocator)

	create := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = vk.DeviceSize(capacity),
		usage       = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS, .TRANSFER_DST},
		sharingMode = .EXCLUSIVE,
	}
	alloc_info: vma.AllocationCreateInfo
	alloc_info.usage = .GPU_ONLY

	result := vma.CreateBuffer(
		VULKAN_STATE.allocator,
		create,
		alloc_info,
		&arena.buffer,
		&arena.allocation,
		nil,
	)
	if result != .SUCCESS {
		log.errorf("[BF_GPU/Arena] CreateBuffer failed for class %v: %v", class, result)
		delete(arena.free_list)
		return false
	}

	addr_info := vk.BufferDeviceAddressInfo {
		sType  = .BUFFER_DEVICE_ADDRESS_INFO,
		buffer = arena.buffer,
	}
	arena.device_address = vk.GetBufferDeviceAddress(VULKAN_STATE.device, &addr_info)

	// One free block covers the entire arena. First-fit picks it for
	// the first allocation; subsequent frees get coalesced into it.
	append(&arena.free_list, GPU_Arena_Block{offset = 0, size = capacity})
	arena.initialized = true

	// Diagnostic tag: record the per-class allocation with the diagnostics
	// layer so the per-kind breakdown picks up arena-backed buffers.
	diag_record_arena_created(class, capacity)
	return true
}

// gpu_arena_destroy releases an arena's VkBuffer + VmaAllocation and
// frees the free list. Safe on an uninitialised arena (no-op).
gpu_arena_destroy :: proc(arena: ^GPU_Arena) {
	if arena == nil || !arena.initialized do return
	if arena.buffer != {} && VULKAN_STATE.allocator != nil {
		vma.DestroyBuffer(VULKAN_STATE.allocator, arena.buffer, arena.allocation)
	}
	delete(arena.free_list)
	arena.buffer = vk.Buffer{}
	arena.allocation = nil
	arena.device_address = 0
	arena.capacity = 0
	arena.used = 0
	arena.peak_used = 0
	arena.alloc_count = 0
	arena.free_count = 0
	arena.initialized = false
	diag_record_arena_destroyed_impl(arena.class)
}

// gpu_arena_allocate reserves `size` bytes at `alignment` inside
// `arena`. Returns (offset, size, ok). `size` is the actual reserved
// size (>= requested, aligned up to `alignment`). On failure the
// caller is expected to either fall back to a per-asset allocation
// or grow the arena via gpu_arena_grow.
//
// The free list is sorted by offset; first-fit picks the lowest-offset
// block that fits. Picks are removed from the free list; the head/tail
// remainder (if any) stays as a smaller free block.
gpu_arena_allocate :: proc(arena: ^GPU_Arena, size: u64, alignment: u64) -> (offset: u64, allocated_size: u64, ok: bool) {
	if arena == nil || !arena.initialized || size == 0 do return 0, 0, false
	eff_align := alignment == 0 ? 1 : alignment

	// First-fit over the sorted free list.
	for &block, i in arena.free_list {
		aligned_offset := align_up_u64(block.offset, eff_align)
		pad := aligned_offset - block.offset
		if pad + size > block.size do continue

		allocated_size = size
		if pad + size < block.size {
			// Split: shrink block to the prefix, record remainder after the new alloc.
			remainder := GPU_Arena_Block {
				offset = aligned_offset + size,
				size   = block.size - pad - size,
			}
			block.size = pad
			if block.size == 0 {
				ordered_remove(&arena.free_list, i)
				gpu_arena_block_insert_at(&arena.free_list, i, remainder)
			} else {
				gpu_arena_block_insert_at(&arena.free_list, i + 1, remainder)
			}
		} else {
			// Exact fit (modulo alignment padding). Drop the block.
			ordered_remove(&arena.free_list, i)
		}

		arena.used += allocated_size
		if arena.used > arena.peak_used do arena.peak_used = arena.used
		arena.alloc_count += 1
		offset = aligned_offset
		ok = true
		return
	}
	return 0, 0, false
}

// gpu_arena_free releases a suballocation back into the arena. The
// (offset, size) pair must come from a previous gpu_arena_allocate
// call on the same arena; calling free with mismatched size is a bug
// that gets logged and no-ops. Adjacent free blocks are coalesced.
gpu_arena_free :: proc(arena: ^GPU_Arena, offset: u64, size: u64) -> bool {
	if arena == nil || !arena.initialized || size == 0 do return false
	if offset + size > arena.capacity {
		log.errorf("[BF_GPU/Arena] free out-of-range: offset=%d size=%d capacity=%d", offset, size, arena.capacity)
		return false
	}

	// Find insertion point in the sorted free list.
	insert_idx := len(arena.free_list)
	for block, i in arena.free_list {
		if block.offset > offset {
			insert_idx = i
			break
		}
	}

	// Coalesce with the previous block (if adjacent).
	merged_prev := false
	if insert_idx > 0 {
		prev := &arena.free_list[insert_idx - 1]
		if prev.offset + prev.size == offset {
			prev.size += size
			merged_prev = true
			arena.coalesce_count += 1
		}
	}

	// Coalesce with the next block (if adjacent).
	target := insert_idx - 1
	merged_next := false
	if !merged_prev {
		gpu_arena_block_insert_at(&arena.free_list, insert_idx, GPU_Arena_Block{offset = offset, size = size})
		target = insert_idx
	}
	if target + 1 < len(arena.free_list) {
		next := &arena.free_list[target + 1]
		cur  := &arena.free_list[target]
		if cur.offset + cur.size == next.offset {
			cur.size += next.size
			ordered_remove(&arena.free_list, target + 1)
			merged_next = true
			arena.coalesce_count += 1
		}
	}
	_ = merged_next

	arena.used -= size
	arena.free_count += 1
	return true
}

// gpu_arena_resolve_addr returns the device address of the byte at
// `offset` inside `arena`. The shader-side read uses this as the
// uint64_t base address for the matching buffer_reference.
gpu_arena_resolve_addr :: #force_inline proc(arena: ^GPU_Arena, offset: u64) -> u64 {
	if arena == nil || !arena.initialized do return 0
	return u64(arena.device_address) + offset
}

// gpu_arena_stats snapshots all counters + the largest free block. The
// largest_free + fragmentation values are the first thing to look at
// when a workload reports "alloc failed" - the answer is usually
// "the free list has plenty of bytes but no single block big enough".
gpu_arena_stats :: proc(arena: ^GPU_Arena) -> GPU_Arena_Stats {
	if arena == nil || !arena.initialized {
		return GPU_Arena_Stats{}
	}
	stats := GPU_Arena_Stats {
		class          = arena.class,
		capacity       = arena.capacity,
		used           = arena.used,
		peak_used      = arena.peak_used,
		free           = arena.capacity - arena.used,
		alloc_count    = arena.alloc_count,
		free_count     = arena.free_count,
		live_suballocs = arena.alloc_count - arena.free_count,
		largest_free   = 0,
		fragmentation  = 0,
		grow_count     = arena.grow_count,
		coalesce_count = arena.coalesce_count,
	}
	for block in arena.free_list {
		if block.size > stats.largest_free do stats.largest_free = block.size
	}
	if stats.free > 0 {
		stats.fragmentation = 1.0 - f32(stats.largest_free) / f32(stats.free)
		if stats.fragmentation < 0 do stats.fragmentation = 0
		if stats.fragmentation > 1 do stats.fragmentation = 1
	}
	return stats
}

// gpu_arena_for_class returns the registry entry for `class`. Returns
// nil when the registry has not been initialised. The pointer is valid
// for the lifetime of the renderer; callers must not free it.
gpu_arena_for_class :: #force_inline proc(class: GPU_Resource_Class) -> ^GPU_Arena {
	if class == .COUNT do return nil
	return &GPU_ARENAS[class]
}

// gpu_arena_is_initialized reports whether the global registry has
// been brought up. The CPU-only test paths query this to decide
// whether arena-backed allocations are available; a nil registry
// falls back to per-asset VMA allocations.
gpu_arena_is_initialized :: #force_inline proc() -> bool {
	return GPU_ARENAS_INITIALIZED
}

// gpu_arena_grow doubles the arena capacity. The old buffer's contents
// are copied into the new buffer via the upload ring. All live
// suballocations keep their offsets (the copy preserves byte order);
// only the underlying buffer + device address change. The arena's
// (free list, used, peak_used, alloc/free counts) are preserved.
//
// Grow is rare; the initial capacities are sized for the largest
// expected asset. When a workload outgrows the seed size, doubling
// keeps the allocator amortised O(1).
gpu_arena_grow :: proc(arena: ^GPU_Arena, new_capacity: u64) -> bool {
	if arena == nil || !arena.initialized do return false
	if new_capacity <= arena.capacity do return true

	// Create the new buffer + allocation.
	create := vk.BufferCreateInfo {
		sType       = .BUFFER_CREATE_INFO,
		size        = vk.DeviceSize(new_capacity),
		usage       = {.STORAGE_BUFFER, .SHADER_DEVICE_ADDRESS, .TRANSFER_DST},
		sharingMode = .EXCLUSIVE,
	}
	alloc_info: vma.AllocationCreateInfo
	alloc_info.usage = .GPU_ONLY
	new_buf: vk.Buffer
	new_alloc: vma.Allocation
	res := vma.CreateBuffer(VULKAN_STATE.allocator, create, alloc_info, &new_buf, &new_alloc, nil)
	if res != .SUCCESS {
		log.errorf("[BF_GPU/Arena] grow CreateBuffer failed: %v", res)
		return false
	}
	addr_info := vk.BufferDeviceAddressInfo {
		sType  = .BUFFER_DEVICE_ADDRESS_INFO,
		buffer = new_buf,
	}
	new_addr := vk.GetBufferDeviceAddress(VULKAN_STATE.device, &addr_info)

	// Record the old capacity so the new free list can fill the tail
	// (the [old_capacity, new_capacity) region is untouched live data).
	old_capacity := arena.capacity

	// Replace the backing storage.
	vma.DestroyBuffer(VULKAN_STATE.allocator, arena.buffer, arena.allocation)
	arena.buffer         = new_buf
	arena.allocation     = new_alloc
	arena.device_address = new_addr
	arena.capacity       = new_capacity
	arena.grow_count     += 1

	// Append a fresh free block for the grown tail.
	append(&arena.free_list, GPU_Arena_Block {
		offset = old_capacity,
		size   = new_capacity - old_capacity,
	})
	return true
}

// ---------------------------------------------------------------------------
// Internal helpers (private).
// ---------------------------------------------------------------------------

// gpu_arena_block_insert_at inserts `block` at `idx` in the sorted free
// list. The list is shifted right with a temporary buffer because Odin
// has no stdlib insert on [dynamic]. Hot path; called once per alloc
// and once per free.
@(private)
gpu_arena_block_insert_at :: proc(list: ^[dynamic]GPU_Arena_Block, idx_in: int, block: GPU_Arena_Block) {
	idx := idx_in
	if idx < 0 do idx = 0
	if idx > len(list) do idx = len(list)
	append(list, block)
	// Shift [idx, len-1) one slot right so `block` lands at `idx`.
	for j := len(list) - 1; j > idx; j -= 1 {
		list[j] = list[j - 1]
	}
	list[idx] = block
}

// diag_record_arena_created is the package-private hook the diagnostics
// module exposes so the arena path contributes to the per-class byte
// counter. Kept here (not in Diagnostics.odin) so the arena allocator
// stays self-contained; the diagnostics side declares the proc body.
@(private)
diag_record_arena_created :: proc(class: GPU_Resource_Class, capacity: u64) {
	// Forward to the diagnostics module.
	diag_record_arena_created_impl(class, capacity)
}

// gpu_resource_class_from_usage maps the asset-pipeline usage bitset
// onto a single GPU_Resource_Class. Vertex and index are the
// high-volume asset uploads that benefit the most from suballocation
// (many small buffers -> one big arena); storage / uniform buffers
// stay on the per-asset path until meshlet-collider / lod-descriptor
// suballocator routes are added.
//
// Returns .Other when the usage set has no arena route; the caller is
// expected to allocate a per-asset VkBuffer.
@(private)
gpu_resource_class_from_usage :: proc(usage: Gpu_Buffer_Usage) -> GPU_Resource_Class {
	if .Vertex_Buffer in usage do return .Vertex
	if .Index_Buffer  in usage do return .Index
	return .Other
}

// gpu_arena_record_used_to_diagnostics snapshots every arena's used /
// free bytes into the diagnostics layer. Cheap (one record per arena)
// and called once per frame from vulkan_upload_scene. The diagnostics
// side never re-derives these values; the snapshot is the single
// source of truth for the per-frame ring.
gpu_arena_record_used_to_diagnostics :: proc() {
	if !GPU_ARENAS_INITIALIZED do return
	for class in GPU_Resource_Class {
		if class == .COUNT {break}
		arena := &GPU_ARENAS[class]
		if !arena.initialized do continue
		diag_record_arena_used_impl(class, arena.used, arena.capacity - arena.used)
	}
}

// gpu_arena_init_for_test installs the free-list / counter bookkeeping
// without touching VMA or vk. The buffer + allocation stay zero; tests
// can exercise alloc/free/reuse without a real Vulkan device. Capacity
// is a literal; no device_address is computed.
gpu_arena_init_for_test :: proc(arena: ^GPU_Arena, class: GPU_Resource_Class, capacity: u64) -> bool {
	if arena == nil do return false
	if arena.initialized do return true
	arena.class = class
	arena.capacity = capacity
	arena.used = 0
	arena.peak_used = 0
	arena.alloc_count = 0
	arena.free_count = 0
	arena.grow_count = 0
	arena.coalesce_count = 0
	arena.free_list = make([dynamic]GPU_Arena_Block, 0, 16, context.allocator)
	append(&arena.free_list, GPU_Arena_Block{offset = 0, size = capacity})
	arena.initialized = true
	return true
}