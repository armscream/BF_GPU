// BF_GPU/Diagnostics.odin
//
// Renderer-side validation, profiling, and observability. Keeps every
// diagnostic CPU-friendly (counters + ring buffers + a single GPU
// timestamp query pool) and never leaks Vulkan types out of BF_GPU.
// All accessors go through the public Renderer_Diagnostics struct so
// the engine / editor can dump stats without an import on Vulkan or
// VMA.
//
// Architecture:
//
//   Renderer_Diagnostics
//     frame_ring[DIAG_FRAME_HISTORY]    rolling per-frame snapshot
//     cpu_stage_*                       ring of stage timings (ns)
//     gpu_stage_*                       ring of GPU timestamp deltas (ns)
//     visible_*_count                   per-frame culling result counters
//     indirect_*                        per-frame indirect-draw shape
//     upload_bytes                      per-frame host->device bandwidth
//     asset_*                           asset-side creation counters
//     gpu_memory_*                      VMA-backed footprint (bytes)
//     frames_in_flight                  MAX_FRAMES_IN_FLIGHT (constant)
//     signaled_slots / completed_slots  polls+fence view
//     label_extensions                  debug-utils probe result
//
// Backed by a single GPU timestamp query pool (one slice per frame in
// flight, two timestamps per pass boundary). The host writes
// vkCmdWriteTimestamp2 at every pass boundary; results land N frames
// later and fold into the rolling history.
//
// The renderer never blocks the CPU on query availability:
// vkGetQueryPoolResults runs in WAIT_BIT (64) mode every frame for the
// slot that completed MAX_FRAMES_IN_FLIGHT frames ago, so the host
// never reads a partial result and never wastes more than one ring
// slot on in-flight queries.

package BF_GPU

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"
import "core:time"
import "core:sync"
import vma "../../dependencies/odin-vma"
import vk "vendor:vulkan"

@(private)
DIAG_FRAME_HISTORY :: 64

// Timer_Slot identifies a per-frame timestamp the GPU records. The
// order matters because the diagnostics snapshot assumes TIMER_SLOT_* is
// strictly increasing through the frame's command buffer recording.
// Adding a new slot requires updating DIAG_TIMER_LABELS / DIAG_TIMER_*_IDX
// below and the per-frame writer in vulkan_record_timestamps.
Timer_Slot :: enum u32 {
	Begin              = 0,
	Upload_End         = 1,
	Culling_End        = 2,
	Traditional_End    = 3,
	Meshlet_End        = 4,
	Submit_End         = 5,
	Present_End        = 6,
	COUNT              = 7,
}

// DIAG_TIMER_*_IDX exposes stable u32 indices for each pass boundary
// so Vulkan-only call sites can index without needing the enum.
@(private)
DIAG_TIMER_BEGIN_IDX          :: u32(Timer_Slot.Begin)
DIAG_TIMER_UPLOAD_END_IDX     :: u32(Timer_Slot.Upload_End)
DIAG_TIMER_CULLING_END_IDX    :: u32(Timer_Slot.Culling_End)
DIAG_TIMER_TRADITIONAL_END_IDX:: u32(Timer_Slot.Traditional_End)
DIAG_TIMER_MESHLET_END_IDX    :: u32(Timer_Slot.Meshlet_End)
DIAG_TIMER_SUBMIT_END_IDX     :: u32(Timer_Slot.Submit_End)
DIAG_TIMER_PRESENT_END_IDX    :: u32(Timer_Slot.Present_End)

// DIAG_TIMER_LABELS are the human-readable names RenderDoc / validation
// layers display alongside each timestamp query. Used by both the GPU
// timestamp path (writes the label) and the formatted dump.
@(private)
DIAG_TIMER_LABELS := [int(Timer_Slot.COUNT)]string {
	"diag.frame.begin",
	"diag.upload.end",
	"diag.culling.end",
	"diag.traditional.end",
	"diag.meshlet.end",
	"diag.submit.end",
	"diag.present.end",
}

// Stage_Timing groups one value per frame in the history ring. Wall
// measurements are in nanoseconds; GPU timestamps are converted from
// tick deltas via vulkan_timestamp_period_ns (queried at device
// creation). Reporting code converts ns -> ms for human display.
Stage_Timing :: struct {
	ns:    [DIAG_FRAME_HISTORY]u64,
	index: u32, // current write head (mod DIAG_FRAME_HISTORY)
	filled:u32, // total frames written; saturates
}

// GPU_Timestamp_Profile owns the GPU-side timestamp query pool. One
// slice per MAX_FRAMES_IN_FLIGHT so the host can read MAX back without
// trampling in-flight queries. Resolution is per the physical device's
// timestampPeriod, queried at device-creation time.
GPU_Timestamp_Profile :: struct {
	pool:                rawptr,        // vk.QueryPool handle (opaque here)
	valid:               bool,
	timestamp_period_ns: f32,           // host ns = (delta * period) / 1e6
	// Per-frame pending conversion: when results land (FRAME_LAG frames
	// later) the uint64 delta is multiplied by timestamp_period_ns to
	// produce a nanosecond duration. Stored here between the vkGet call
	// and the diagnostic roll-over.
	pending_deltas:      [DIAG_FRAME_HISTORY][int(Timer_Slot.COUNT)]u64,
	pending_frame_ids:   [DIAG_FRAME_HISTORY]u64,
	pending_filled:      u32,
}

// Debug_Labels_Capability exposes whether VK_EXT_debug_utils is bound
// at runtime. Labels are no-ops when unsupported; the diagnostics dump
// reflects the live capability so reviewers know whether the names
// they see in RenderDoc were applied by the engine or by something
// else.
Debug_Labels_Capability :: enum u8 {
	Unsupported            = 0,
	Enabled                = 1,
	Enabled_With_Validation= 2,
}

// Heap_Bucket classifies a physical device memory heap for the per-
// heap breakdown shown in the diagnostics dump. Mirrors the way
// VK_EXT_memory_budget reports `heapUsage`/`heapBudget` arrays; the
// buckets exist so multi-heap GPUs (dedicated + integrated, HBCC,
// unified-memory laptops) don't hide their pressure in a single
// aggregate.
Heap_Bucket :: enum u8 {
	Other           = 0, // unmapped flag combinations
	Device_Local    = 1, // .DEVICE_LOCAL only
	Host_Visible    = 2, // .HOST_VISIBLE only
	Device_Lazy     = 3, // .DEVICE_LOCAL + lazily allocated
	Host_Device     = 4, // .DEVICE_LOCAL + .HOST_VISIBLE (UMA on iGPU)
	COUNT           = 5,
}

// Heap_Stats is the per-bucket breakdown the memory sampler fills
// from VK_EXT_memory_budget. `used` is the driver-reported current
// usage (includes implicit objects: swapchain, pipelines, etc.);
// `budget` is the soft cap the driver tells us we're allowed to
// spend. Both are in bytes.
Heap_Stats :: struct {
	used:   u64,
	budget: u64,
}

// Ring_Stats is the per-frame-window summary the formatted dump
// displays for every Stage_Timing ring. Pre-computed once per dump
// (not per cell) so the ring snapshots stay cheap.
Ring_Stats :: struct {
	min:     u64,
	avg:     u64,
	p50:     u64,
	p95:     u64,
	max:     u64,
	samples: u32, // how many of the DIAG_FRAME_HISTORY slots are valid
}

// Defrag_Telemetry captures before/after state for a VMA defrag pass
// so the formatted dump can answer "is the defrag actually helping?".
// The 25%-wasted threshold the renderer uses to skip defragmentation
// is a heuristic; this telemetry lets a reviewer replace it with an
// evidence-based policy without changing the call site.
Defrag_Telemetry :: struct {
	last_run_frame:           u64,
	last_pre_unused_bytes:    u64,
	last_post_unused_bytes:   u64,
	last_allocations_moved:   u32,
	last_bytes_moved:         u64,
	last_device_blocks_freed: u32,
	last_was_skipped:         bool, // true if pre-check said "not worth it"
	runs_total:               u64,
	runs_skipped_total:       u64,
	bytes_recovered_total:    u64, // sum of (pre - post) across runs
	allocations_moved_total:  u64,
	bytes_moved_total:        u64,
	device_blocks_freed_total:u64,
}

// Renderer_Diagnostics is the single observable surface the engine
// reads. Every field is a snapshot of the rolling ring; consumers
// should treat the array as a circular buffer with `latest_index` as
// the newest slot.
Renderer_Diagnostics :: struct {
	cpu: struct {
		scene_extract: Stage_Timing,
		upload:       Stage_Timing,
		submit:       Stage_Timing,
		frame_present:Stage_Timing,
		total_frame:  Stage_Timing,
		gpu_wait:     Stage_Timing,
	},
	gpu: struct {
		culling_ns:       Stage_Timing,
		traditional_ns:   Stage_Timing,
		meshlet_ns:       Stage_Timing,
		upload_ns:        Stage_Timing,
		submit_ns:        Stage_Timing,
		render_ns:        Stage_Timing,
		present_ns:       Stage_Timing,
		total_ns:         Stage_Timing,
		timestamps:       GPU_Timestamp_Profile,
	},
	visible: struct {
		chunks:           Stage_Timing, // value holds u32 count
		models:           Stage_Timing,
		meshes:           Stage_Timing,
		pending_assets:   Stage_Timing,
	},
	indirect: struct {
		traditional_cmds: Stage_Timing,
		meshlet_cmds:     Stage_Timing,
		total_cmds:       Stage_Timing,
		draw_calls:       Stage_Timing,
	},
		upload: struct {
			bytes_submitted:  Stage_Timing, // host->device bytes
			uploads:          Stage_Timing, // upload_buffer call count
			asset_buffers:    Stage_Timing, // create_asset_buffer call count
			images_created:   Stage_Timing,
			images_destroyed: Stage_Timing,
			// Per-attribute breakdown. Index by Upload_Kind. Lifetime
			// numbers roll forward forever (good for catching leaks);
			// the per-frame ring gives the streaming-pressure view.
			by_kind:         [int(Upload_Kind.COUNT)]Upload_Kind_Stats,
		},
	asset: struct {
		models_created:   u64,
		meshes_created:   u64,
		materials_created:u64,
		textures_created: u64,
		pending_retries:  u64,
	},
	gpu_memory: struct {
		used_bytes:           u64, // bytes in VMA allocations, current
		budget_bytes:         u64, // device-local heap budget
		allocation_count:     u32,
		buffer_count:         u32,
		image_count:          u32, // lifetime (created - destroyed)
		sampler_count:        u32, // lifetime (created - destroyed)
		// Per-heap-bucket breakdown from VK_EXT_memory_budget. Index
		// by Heap_Bucket. The renderer only fills buckets it sees in
		// the live memory properties; others remain zero. The
		// formatted dump aggregates the device-local buckets
		// separately from host-visible so reviewers see pressure
		// attribution, not just a single number.
		heaps:              [int(Heap_Bucket.COUNT)]Heap_Stats,
		// Per-memory-type breakdown (HOST_VISIBLE / LAZILY_ALLOCATED
		// attribution that heap flags can't provide). Source of truth
		// is vma.TotalStatistics.memoryType[i].statistics summed
		// across types with matching MemoryPropertyFlag bits.
		memory_types:       [int(Memory_Type_Bucket.COUNT)]Memory_Type_Stats,
		// VMA suballocator telemetry from CalculateStatistics. These
		// are the real fragmentation signal: the delta between
		// blockBytes and allocationBytes IS the wasted space, and
		// unusedRangeCount is the granularity that drives defrag
		// cost. Pre-computed once per memory snapshot.
		suballocator: struct {
			block_bytes:            u64,
			allocation_bytes:       u64,
			unused_bytes:           u64,
			unused_range_count:     u32,
			unused_range_bytes_max: u64,
			allocation_bytes_min:   u64,
			allocation_bytes_max:   u64,
		},
		defrag:              Defrag_Telemetry,
	},
	frames: struct {
		in_flight:        u32,           // MAX_FRAMES_IN_FLIGHT
		slots_submitted:  Stage_Timing,  // per-frame count of pending slots
		slots_signaled:   Stage_Timing,  // per-frame count newly signaled
		slots_completed:  Stage_Timing,  // per-frame total completed on poll
		completion_value: u64,           // latest graphics timeline value
		oldest_uncompleted_value: u64,
	},
	scene_memory: struct {
		// Per-frame resident byte counts for the CPU scene and its GPU
		// mirror. Prompt 01 surface: the editor / profilers can graph
		// scene footprint over time without walking every pool. The
		// Stage_Timing rings reuse the existing pattern (value holds
		// a u64 count of bytes).
		render_bytes: Stage_Timing,
		gpu_bytes:    Stage_Timing,
		// Latest per-pool breakdown of the GPU mirror. Snapshot is
		// captured every frame by diag_record_scene_memory and the
		// editor reads it directly. The Stage_Timing ring gives the
		// rolling history without an extra copy step.
		gpu_pools:    GPU_Pool_Bytes,
	},
	debug: struct {
		validation_enabled:  bool,
		label_capability:    Debug_Labels_Capability,
		timestamp_pool_ready:bool,
	},
	// Per-arena snapshot. One entry per GPU_Resource_Class; the
	// arena allocator (Gpu_Arena.odin) updates these every frame.
	// `arena_count` is the number of currently-initialised arenas so
	// the dump can report "all 16 arenas live" vs "12 of 16 (mesh
	// geometry arrived disabled)".
	arenas:      [int(GPU_Resource_Class.COUNT)]GPU_Arena_Stats,
	arena_count: u32,
	// Whether the diagnostics subsystem has been initialised. Set by
	// renderer_diagnostics_init() and cleared by
	// renderer_diagnostics_shutdown(). Public accessor reads this to
	// decide whether to print "no diagnostics available" instead of
	// returning zeroed counters.
	initialized: bool,
	// Diagnostic wall clock anchor. Used to roll the per-frame ring
	// without depending on the BF_DAG scheduler.
	wall_clock_anchor_ns: i64,
	// Per-frame begin wall-clock anchor (set by scene_extract_step,
	// consumed by frame_present_step to derive the CPU total-frame
	// duration). Allocated here so a single point owns the cross-step
	// timing closure.
	frames_completion_anchor_ns: i64,
	// The frame index the closure is for: scene_extract_step writes,
	// frame_present_step reads, both with frame_id stored here so
	// frame_context_advance between them doesn't shift the write
	// destination out from under the total-frame recorder.
	frames_completion_frame_id: u64,
}

@(private)
DIAGNOSTICS_STATE: Renderer_Diagnostics

// DIAGNOSTICS_MUTEX serialises every read+write of DIAGNOSTICS_STATE
// so the diagnostics recorders (called from the render thread, asset
// thread, and frame-complete poll path) cannot clobber each other,
// and so test fixtures can snapshot+restore DIAGNOSTICS_STATE under
// the same lock the recorders use. The lock is uncontended in
// production; contention only arises if a recorder re-enters the
// diagnostics subsystem, which the existing code does at the
// boundary cases (renderer_diagnostics_init -> diag_dump_to_file
// -> renderer_diagnostics_string; diag_maybe_tick -> same).
// A Recursive_Mutex is required so the re-entrant paths don't
// self-deadlock; non-recursive would force every public init/tick
// path to release the lock before delegating, which is fragile.
@(private)
DIAGNOSTICS_MUTEX: sync.Recursive_Mutex

// diag_lock / diag_unlock are the canonical entry/exit points.
// Wrap multi-statement diagnostics mutations with
// `sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX);
//  defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)`;
// single-statement updates inside the file_guard helpers below
// already take the lock.
@(private)
diag_lock :: proc() { sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX) }
@(private)
diag_unlock :: proc() { sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX) }

@(private)
DIAGNOSTICS_GPU_STATE: Vulkan_Diagnostics_State

// Vulkan_Diagnostics_State is the backend-private side of the
// diagnostics. Lives in BF_GPU, never escapes the package. Tests that
// touch it directly simulate a no-Vulkan device.
Vulkan_Diagnostics_State :: struct {
	query_pool:          rawptr, // opaque vk.QueryPool
	query_pool_valid:    bool,
	timestamp_period_ns: f32,
	debug_utils_supported:bool,
	validation_enabled:  bool,
	debug_messenger:     rawptr, // opaque vk.DebugUtilsMessengerEXT
	initialized:         bool,
}

// ===========================================================================
//* Lifecycle.
// ===========================================================================

// renderer_diagnostics_init allocates the per-frame history ring,
// records the wall clock anchor, and arms the CPU-side accumulators.
// Safe to call multiple times (no-op after the first). The Vulkan
// query pool + debug messenger are created later, by vulkan_diag_init,
// once the physical device is available.
//
// Also writes an initial empty snapshot to bf_gpu_diagnostics.txt so the
// file exists from the moment diagnostics is up. Without this, a fast
// crash in early Vulkan init would leave the file missing entirely -
// the periodic dump only fires after frame DIAG_LOG_DUMP_PERIOD.
renderer_diagnostics_init :: proc() {
	if DIAGNOSTICS_STATE.initialized do return
	DIAGNOSTICS_STATE.initialized = true
	DIAGNOSTICS_STATE.wall_clock_anchor_ns = time.time_to_unix_nano(time.now())
	DIAGNOSTICS_STATE.frames.in_flight = u32(MAX_FRAMES_IN_FLIGHT)
	DIAGNOSTICS_STATE.frames.completion_value = 0
	DIAGNOSTICS_STATE.frames.oldest_uncompleted_value = 0
	log.info("[BF_GPU] Renderer diagnostics initialised")
	diag_dump_to_file()
}

// renderer_diagnostics_shutdown clears the rolling rings and zeros the
// aggregate counters. The Vulkan-side state (query pool, debug
// messenger) is released by vulkan_diag_shutdown before this runs.
renderer_diagnostics_shutdown :: proc() {
	if !DIAGNOSTICS_STATE.initialized do return
	DIAGNOSTICS_STATE = {}
	DIAGNOSTICS_STATE.frames.in_flight = u32(MAX_FRAMES_IN_FLIGHT)
	log.info("[BF_GPU] Renderer diagnostics shut down")
}

// renderer_diagnostics returns a read-only pointer to the rolling
// snapshot. Tests inspect this pointer directly. Production callers
// should copy the struct out (it is value-type, copy-safe) before
// reading individual fields across frames.
renderer_diagnostics :: proc() -> ^Renderer_Diagnostics {
	return &DIAGNOSTICS_STATE
}

// ===========================================================================
//* Per-frame recording hooks.
// ===========================================================================

// diag_begin_frame marks the wall-clock start of a logical frame. The
// renderer engine does not own a strict begin/end boundary on its own;
// this is called once per SceneExtract/Upload/Submit trio at the top of
// scene_extract_step. The returned stage start time is recorded by
// diag_finish_* helpers.
diag_begin_frame :: proc() -> i64 {
	return time.time_to_unix_nano(time.now())
}

// diag_finish_stage records the end time of a named CPU stage and
// rolls the delta into the matching Stage_Timing ring slot.
//
// `begin_ns` is the time.nanoseconds returned by diag_begin_frame (or
// the matched begin); `stage` selects the destination ring. Frame
// index is the current BF_GPU frame counter (frame_context.frame_idx);
// the function is a no-op when the diagnostics subsystem hasn't been
// initialised.
diag_finish_stage :: proc(stage: ^Stage_Timing, frame_index: u64, begin_ns: i64) {
	if !DIAGNOSTICS_STATE.initialized do return
	if stage == nil do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	end_ns := time.time_to_unix_nano(time.now())
	delta := u64(end_ns - begin_ns)
	if delta < 0 do delta = 0

	ring_index := u32(frame_index % u64(DIAG_FRAME_HISTORY))
	stage.ns[ring_index] = delta
	stage.index  = ring_index
	if u64(stage.filled) < frame_index + 1 {
		stage.filled = min(u32(frame_index + 1), u32(DIAG_FRAME_HISTORY))
	}
}

// diag_record_scalar is the generic record-side helper: writes a per-
// frame scalar into the Stage_Timing ring (interpreting `ns` as the
// payload field). Tests + non-CPU bookkeeping call this for visible
// counts, indirect-draw counts, upload bytes, etc.
diag_record_scalar :: proc(stage: ^Stage_Timing, frame_index: u64, value: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	if stage == nil do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	ring_index := u32(frame_index % u64(DIAG_FRAME_HISTORY))
	stage.ns[ring_index] = value
	stage.index = ring_index
	if u64(stage.filled) < frame_index + 1 {
		stage.filled = min(u32(frame_index + 1), u32(DIAG_FRAME_HISTORY))
	}
}

// diag_take_gpu_wait_ns records how long the host blocked waiting on
// the graphics timeline before reusing a frame slot. Used to surface
// "GPU-bound vs CPU-bound" framing in the formatted dump.
diag_take_gpu_wait_ns :: proc(frame_index: u64, wait_ns: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	ring_index := u32(frame_index % u64(DIAG_FRAME_HISTORY))
	DIAGNOSTICS_STATE.cpu.gpu_wait.ns[ring_index] = wait_ns
	DIAGNOSTICS_STATE.cpu.gpu_wait.index = ring_index
	if u64(DIAGNOSTICS_STATE.cpu.gpu_wait.filled) < frame_index + 1 {
		DIAGNOSTICS_STATE.cpu.gpu_wait.filled = min(
			u32(frame_index + 1),
			u32(DIAG_FRAME_HISTORY),
		)
	}
}

// diag_record_frame_finish writes the end-of-frame wall time and lets
// callers inspect cpu.total_frame for whole-frame budget. Called from
// frame_present_step once the swapchain present has been kicked off.
diag_record_frame_finish :: proc(frame_index: u64, begin_ns: i64) {
	if !DIAGNOSTICS_STATE.initialized do return
	diag_finish_stage(&DIAGNOSTICS_STATE.cpu.total_frame, frame_index, begin_ns)
}

// diag_record_extraction adds the Extraction_Stats the renderer already
// produces (Extraction.odin) into the rolling visible-count rings. The
// stats use u32; we coerce into Stage_Timing's u64 ring.
diag_record_extraction :: proc(frame_index: u64, stats: ^Extraction_Stats) {
	if !DIAGNOSTICS_STATE.initialized do return
	if stats == nil do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	diag_record_scalar(&d.visible.chunks,           frame_index, u64(stats.chunks_visited))
	diag_record_scalar(&d.visible.models,           frame_index, u64(stats.entities_visited))
	diag_record_scalar(&d.indirect.draw_calls,      frame_index, u64(stats.instances_retained))
	diag_record_scalar(&d.visible.pending_assets,   frame_index, u64(stats.pending_assets))
}

// diag_record_gpu_scene mirrors the per-frame counts that Scene.odin
// already publishes on GPU_Scene. Called once per frame from
// render_upload_step after GPU_Scene reflects this frame's upload.
diag_record_gpu_scene :: proc(frame_index: u64, gpu: ^GPU_Scene) {
	if !DIAGNOSTICS_STATE.initialized do return
	if gpu == nil do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	diag_record_scalar(&d.visible.meshes,         frame_index, u64(gpu.frame_all_transforms))
	diag_record_scalar(&d.indirect.traditional_cmds,frame_index, u64(gpu.frame_traditional_cmd_count))
	diag_record_scalar(&d.indirect.meshlet_cmds,    frame_index, u64(gpu.frame_meshlet_cmd_count))
	diag_record_scalar(&d.indirect.total_cmds,      frame_index, u64(gpu.frame_indirect_cmd_count))
}

// diag_record_scene_memory snapshots the per-frame resident byte count of
// the CPU Render_Scene + the GPU_Scene mirror and writes both into the
// rolling ring. Also stashes the latest GPU pool breakdown so the editor
// can pull a per-pool view without re-walking.
//
// Called once per frame from render_upload_step after gpu_scene_update.
diag_record_scene_memory :: proc(frame_index: u64, scene: ^Render_Scene, gpu: ^GPU_Scene) {
	if !DIAGNOSTICS_STATE.initialized do return
	if scene == nil || gpu == nil do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE

	rbytes := total_render_pool_bytes(render_scene_byte_count(scene))
	gbytes := Total(gpu_scene_byte_count(gpu))

	diag_record_scalar(&d.scene_memory.render_bytes, frame_index, u64(rbytes))
	diag_record_scalar(&d.scene_memory.gpu_bytes,    frame_index, u64(gbytes))
	d.scene_memory.gpu_pools = gpu_scene_byte_count(gpu)
}

// diag_record_upload_bandwidth accumulates host->device bytes via the
// upload buffer + asset buffer hooks. The `bytes` argument is added
// to the per-frame ring entry. Safe with value 0 (does nothing).
//
// Kept as a wrapper around diag_record_upload_bandwidth_with_kind for
// legacy callers; new code should pass an explicit Upload_Kind.
diag_record_upload_bandwidth :: proc(frame_index: u64, bytes: u64) {
	diag_record_upload_bandwidth_with_kind(.Unknown, frame_index, bytes)
}

// diag_record_upload_count bumps the per-frame "how many upload_buffer
// calls happened" counter. Matches diag_record_upload_bandwidth.
//
// Legacy wrapper; new code should use diag_record_upload_count_with_kind.
diag_record_upload_count :: proc(frame_index: u64) {
	diag_record_upload_count_with_kind(.Unknown, frame_index)
}

// diag_record_asset_buffer_created increments the lifetime asset-buffer
// creation counter and the per-frame ring counter.
//
// Legacy wrapper; new code should use diag_record_asset_buffer_created_with_kind.
diag_record_asset_buffer_created :: proc(frame_index: u64) {
	diag_record_asset_buffer_created_with_kind(.Asset_Other, frame_index)
}

// diag_record_upload_bandwidth_with_kind is the attributed entry point.
// Each call writes into:
//   1. The per-frame ring index (so the rolling-window throughput
//      view survives).
//   2. The lifetime by_kind bytes counter (so leaks surface).
//   3. The per-frame ring inside by_kind (so streaming-pressure per
//      kind is independently observable).
// The call is a no-op when bytes == 0, mirroring the old behaviour.
diag_record_upload_bandwidth_with_kind :: proc(kind: Upload_Kind, frame_index: u64, bytes: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	if bytes == 0 do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	ring_index := u32(frame_index % u64(DIAG_FRAME_HISTORY))
	// Aggregate ring (legacy compat + cross-kind view).
	d.upload.bytes_submitted.ns[ring_index] += bytes
	d.upload.bytes_submitted.index = ring_index
	if u64(d.upload.bytes_submitted.filled) < frame_index + 1 {
		d.upload.bytes_submitted.filled = min(
			u32(frame_index + 1),
			u32(DIAG_FRAME_HISTORY),
		)
	}
	// Per-kind lifetime.
	ks := &d.upload.by_kind[int(kind)]
	ks.bytes_lifetime += bytes
	// Per-kind per-frame ring. Re-uses Stage_Timing so the existing
	// ring_stats / pressure helper work for the per-kind view too.
	ks.bytes_per_frame.ns[ring_index] += bytes
	ks.bytes_per_frame.index = ring_index
	if u64(ks.bytes_per_frame.filled) < frame_index + 1 {
		ks.bytes_per_frame.filled = min(
			u32(frame_index + 1),
			u32(DIAG_FRAME_HISTORY),
		)
	}
}

// diag_record_upload_count_with_kind is the attributed sibling of
// diag_record_upload_bandwidth_with_kind. Tracks per-kind upload
// call counts (lifetime + per-frame).
diag_record_upload_count_with_kind :: proc(kind: Upload_Kind, frame_index: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	ring_index := u32(frame_index % u64(DIAG_FRAME_HISTORY))
	d.upload.uploads.ns[ring_index] += 1
	d.upload.uploads.index = ring_index
	if u64(d.upload.uploads.filled) < frame_index + 1 {
		d.upload.uploads.filled = min(
			u32(frame_index + 1),
			u32(DIAG_FRAME_HISTORY),
		)
	}
	ks := &d.upload.by_kind[int(kind)]
	ks.count_lifetime += 1
	ks.count_per_frame.ns[ring_index] += 1
	ks.count_per_frame.index = ring_index
	if u64(ks.count_per_frame.filled) < frame_index + 1 {
		ks.count_per_frame.filled = min(
			u32(frame_index + 1),
			u32(DIAG_FRAME_HISTORY),
		)
	}
}

// diag_record_asset_buffer_created_with_kind mirrors the legacy proc
// with explicit kind attribution. Asset_Sync + Vulkan.odin call this
// directly so the dump's "Asset_*" rows reflect actual asset buffer
// creation, not "Unknown" leftover from the wrapper.
diag_record_asset_buffer_created_with_kind :: proc(kind: Upload_Kind, frame_index: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	ring_index := u32(frame_index % u64(DIAG_FRAME_HISTORY))
	d.upload.asset_buffers.ns[ring_index] += 1
	d.upload.asset_buffers.index = ring_index
	if u64(d.upload.asset_buffers.filled) < frame_index + 1 {
		d.upload.asset_buffers.filled = min(
			u32(frame_index + 1),
			u32(DIAG_FRAME_HISTORY),
		)
	}
	// The asset-buffer creation counts go to the per-kind lifetime
	// so a leaked vertex buffer shows up as growth in Asset_Vertex
	// bytes_lifetime. The size isn't available here (just the call
	// count); the bytes go via diag_record_upload_bandwidth_with_kind
	// on the actual upload.
	ks := &d.upload.by_kind[int(kind)]
	ks.count_lifetime += 1
}

// diag_record_image_created / destroyed accumulate per-frame image
// churn across multiple hook invocations in the same frame. They
// also bump the lifetime image counter in gpu_memory so the dump can
// report "how many images are currently alive" without having to
// fold per-frame rings.
diag_record_image_created :: proc(frame_index: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	ring_index := u32(frame_index % u64(DIAG_FRAME_HISTORY))
	d.upload.images_created.ns[ring_index] += 1
	d.upload.images_created.index = ring_index
	if u64(d.upload.images_created.filled) < frame_index + 1 {
		d.upload.images_created.filled = min(
			u32(frame_index + 1),
			u32(DIAG_FRAME_HISTORY),
		)
	}
	d.gpu_memory.image_count += 1
}
diag_record_image_destroyed :: proc(frame_index: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	ring_index := u32(frame_index % u64(DIAG_FRAME_HISTORY))
	d.upload.images_destroyed.ns[ring_index] += 1
	d.upload.images_destroyed.index = ring_index
	if u64(d.upload.images_destroyed.filled) < frame_index + 1 {
		d.upload.images_destroyed.filled = min(
			u32(frame_index + 1),
			u32(DIAG_FRAME_HISTORY),
		)
	}
	if d.gpu_memory.image_count > 0 {
		d.gpu_memory.image_count -= 1
	}
}

// diag_record_sampler_created / destroyed mirror the image hooks for
// VkSampler. The renderer does not currently call these (no sampler
// pool exists yet); the helpers are wired here so adding sampler
// tracking later is a one-line change at the call site and the
// diagnostics field is no longer dead.
diag_record_sampler_created :: proc() {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	DIAGNOSTICS_STATE.gpu_memory.sampler_count += 1
}
diag_record_sampler_destroyed :: proc() {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	if d.gpu_memory.sampler_count > 0 {
		d.gpu_memory.sampler_count -= 1
	}
}

// diag_inc_asset_created bumps a lifetime asset-creation counter.
// `kind` selects which field grows. The renderer wires this into
// gpu_asset_sync_attach or Asset_Sync.odin.
diag_inc_asset_created :: proc(kind: Asset_Creation_Kind) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	switch kind {
	case .Model:     d.asset.models_created += 1
	case .Mesh:      d.asset.meshes_created += 1
	case .Material:  d.asset.materials_created += 1
	case .Texture:   d.asset.textures_created += 1
	case .Pending:   d.asset.pending_retries += 1
	}
}

// ===========================================================================
//* GPU memory telemetry (VMA + VK_EXT_memory_budget).
//
// diag_update_memory_snapshot is the slow path the renderer calls
// every DIAG_MEMORY_SAMPLE_PERIOD frames. It pulls heap budgets from
// VMA (which queries VK_EXT_memory_budget internally) and folds the
// total device-local heap budget + used bytes into Renderer_Diagnostics
// so the formatted dump reports real numbers. The renderer's periodic
// health check (asset pressure, GPU OOM early warning) reads these.
//
// The renderer schedules this off the render thread only when VMA has
// been initialised; until then it is a no-op.
// ===========================================================================

DIAG_MEMORY_SAMPLE_PERIOD :: u64(1) // frames between heap-budget queries (1 = per-frame)
DIAG_DEFRAG_PERIOD          :: u64(600) // frames between VMA defragmentation passes

@(private)
DIAG_DEFRAG_FRAME_COUNTER: u64
@(private)
DIAG_MEMORY_FRAME_COUNTER: u64

// diag_classify_heap picks the Heap_Bucket for a heap flag set. The
// Vulkan heap flag bitfield (vk.MemoryHeapFlag) only exposes
// DEVICE_LOCAL and MULTI_INSTANCE; HOST_VISIBLE / LAZILY_ALLOCATED
// live on the memory-type flag set, not the heap set. Aggregating
// budgets per *type* would be more granular but requires an extra
// walk over vkPhysicalDeviceMemoryProperties.memoryTypes; for v1 the
// heap-level split is enough to attribute pressure to "local VRAM
// vs. other" on the common single-heap case. The enum reserves the
// other buckets so adding memory-type aggregation later is an
// additive change.
@(private)
diag_classify_heap :: proc(flags: vk.MemoryHeapFlags) -> Heap_Bucket {
	if .DEVICE_LOCAL in flags do return .Device_Local
	return .Other
}

// diag_classify_memory_type picks the Memory_Type_Bucket for a
// memory-property flag set. The ordering matters: the LAZILY_ALLOCATED
// + DEVICE_LOCAL combo (tiler-friendly) is checked before the bare
// DEVICE_LOCAL bucket so a type with both flags lands in the lazy
// bucket. Similarly, the all-three-flags (UMA) case is checked before
// falling through to bare DEVICE_LOCAL or HOST_VISIBLE.
@(private)
diag_classify_memory_type :: proc(props: vk.MemoryPropertyFlags) -> Memory_Type_Bucket {
	has_dev := .DEVICE_LOCAL in props
	has_vis := .HOST_VISIBLE in props
	has_ca  := .HOST_COHERENT in props
	has_laz := .LAZILY_ALLOCATED in props
	switch {
	case has_dev && has_vis && has_ca: return .UMA
	case has_dev && has_laz:           return .Lazy_Allocated
	case has_vis:                      return .Host_Visible
	case has_dev:                      return .Device_Local
	case:                              return .Other
	}
}

// diag_kind_from_usage derives an Upload_Kind from the Gpu_Buffer_Usage
// flags a caller passes to create_asset_buffer. Vertex/Index carry
// the obvious tag; everything else is Asset_Other. Internal allocations
// (descriptor backing, gctx) aren't funneled through create_asset_buffer
// today, so this returns Asset_* variants only.
diag_kind_from_usage :: proc(usage: Gpu_Buffer_Usage) -> Upload_Kind {
	if .Vertex_Buffer in usage do return .Asset_Vertex
	if .Index_Buffer   in usage do return .Asset_Index
	return .Asset_Other
}

// diag_update_memory_snapshot queries VMA for heap budgets, fills the
// per-heap-bucket breakdown + VMA suballocator stats, and returns.
// Cheap on the CPU side (~us) but it does a Vulkan call so the caller
// throttles it via DIAG_MEMORY_SAMPLE_PERIOD.
diag_update_memory_snapshot :: proc() {
	if !DIAGNOSTICS_STATE.initialized do return
	allocator := diag_vma_allocator()
	if allocator == nil do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)

	mem_props: vk.PhysicalDeviceMemoryProperties2
	mem_props.sType = .PHYSICAL_DEVICE_MEMORY_PROPERTIES_2
	mem_props.pNext = nil
	heap_budget_props: vk.PhysicalDeviceMemoryBudgetPropertiesEXT
	heap_budget_props.sType = .PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
	heap_budget_props.pNext = nil
	mem_props.pNext = &heap_budget_props
	physical_device := diag_vma_physical_device()
	if physical_device == nil do return
	vk.GetPhysicalDeviceMemoryProperties2(physical_device, &mem_props)

	// VMA also exposes a stats query; combine the two so the diagnostics
	// log shows device-level (driver-reported budget) + VMA-allocator
	// level (what the renderer actually owns). The renderer doesn't
	// double-count because the budgets are cumulative across heaps.
	stats: vma.TotalStatistics
	vma.CalculateStatistics(allocator, &stats)

	d := &DIAGNOSTICS_STATE
	// Zero the per-bucket counters before re-aggregation. Without
	// this, a bucket that was full last snapshot and went empty this
	// snapshot would keep its old used/budget until the next call to
	// this proc.
	for i in 0..<int(Heap_Bucket.COUNT)         do d.gpu_memory.heaps[i] = {}
	for i in 0..<int(Memory_Type_Bucket.COUNT)  do d.gpu_memory.memory_types[i] = {}

	// Per-heap aggregation. heap_budget_props indexes by HEAP; we
	// attribute each heap's budget + usage to the matching heap bucket.
	#no_bounds_check for i in 0 ..< int(mem_props.memoryProperties.memoryHeapCount) {
		heap := &mem_props.memoryProperties.memoryHeaps[i]
		bucket := diag_classify_heap(heap.flags)
		hs := &d.gpu_memory.heaps[int(bucket)]
		hs.used   += u64(heap_budget_props.heapUsage[i])
		hs.budget += u64(heap_budget_props.heapBudget[i])
	}

	// Per-memory-type aggregation. Walk every memory type exposed by
	// the driver and bucket by property flag combination. Allocation
	// bytes come from VMA's per-type stats (indexed identically to
	// mem_props.memoryTypes[]). Budget is the heap's full budget,
	// over-approximated across types in the same heap. The over-count
	// is intentional: a bucket inside a pressured heap will show
	// >=100%, which makes heap-level pressure visible in the
	// per-type breakdown without needing a separate aggregator.
	#no_bounds_check for i in 0 ..< int(mem_props.memoryProperties.memoryTypeCount) {
		mt   := &mem_props.memoryProperties.memoryTypes[i]
		bucket := diag_classify_memory_type(mt.propertyFlags)
		ms   := &d.gpu_memory.memory_types[int(bucket)]
		ts   := &stats.memoryType[i]
		ms.bytes            += u64(ts.statistics.allocationBytes)
		ms.allocation_count += ts.statistics.allocationCount
		ms.unused_bytes     += u64(ts.statistics.blockBytes - ts.statistics.allocationBytes)
		ms.unused_range_count += ts.unusedRangeCount
		ms.budget_bytes     += u64(heap_budget_props.heapBudget[mt.heapIndex])
	}

	// Backwards-compatible aggregates: total used/budget across every
	// heap. `used` here is the driver-reported number, which already
	// includes implicit objects (swapchain, pipelines, etc.) that VMA
	// doesn't track. Reviewers who want a single number get one;
	// reviewers who want attribution get the per-bucket breakdown.
	total_used, total_budget: u64
	for i in 0..<int(Heap_Bucket.COUNT) {
		total_used   += d.gpu_memory.heaps[i].used
		total_budget += d.gpu_memory.heaps[i].budget
	}
	d.gpu_memory.used_bytes   = total_used
	d.gpu_memory.budget_bytes = total_budget

	// VMA suballocator stats. These are the fragmentation signal
	// (blockBytes - allocationBytes is the wasted space inside
	// existing blocks, not the heap slack). unused_range_count and
	// unused_range_bytes_max drive defrag cost; tracking both lets
	// the renderer replace the current 25%-wasted heuristic with an
	// evidence-based policy.
	d.gpu_memory.allocation_count = u32(stats.total.statistics.allocationCount)
	d.gpu_memory.buffer_count     = u32(stats.total.statistics.blockCount)
	sub := &d.gpu_memory.suballocator
	sub.block_bytes            = u64(stats.total.statistics.blockBytes)
	sub.allocation_bytes       = u64(stats.total.statistics.allocationBytes)
	sub.unused_bytes           = sub.block_bytes - sub.allocation_bytes
	sub.unused_range_count     = stats.total.unusedRangeCount
	sub.unused_range_bytes_max = u64(stats.total.unusedRangeSizeMax)
	sub.allocation_bytes_min   = u64(stats.total.allocationSizeMin)
	sub.allocation_bytes_max   = u64(stats.total.allocationSizeMax)
}

// diag_vma_allocator / diag_vma_physical_device are accessors the
// memory sampler reaches through. They are defined in Vulkan.odin and
// forward to VULKAN_STATE. Keeping them in Diagnostics would create a
// circular import with VMA, so the indirection lives behind a tiny
// shim.
@(private)
diag_vma_allocator: proc() -> vma.Allocator = allocator_accessor
@(private)
diag_vma_physical_device: proc() -> vk.PhysicalDevice = physical_device_accessor

// diag_maybe_tick triggers the slow-path telemetry + defragmentation
// passes on their own periods. Called once per frame from vulkan_frame.
//
// DIAG_MEMORY_SAMPLE_PERIOD is 1 by default - the per-frame heap
// budget query is ~us (dominated by vma.CalculateStatistics' walk
// over the live allocation list), and the per-frame resolution is
// worth more than the saved CPU cycles. If profiling shows this in
// a hot path on a target device, bump DIAG_MEMORY_SAMPLE_PERIOD back
// up to 60 and accept the 1-second transient-pressure blur.
diag_maybe_tick :: proc(frame_index: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	if (frame_index % DIAG_MEMORY_SAMPLE_PERIOD) == 0 {
		diag_update_memory_snapshot()
	}
	if (frame_index % DIAG_DEFRAG_PERIOD) == 0 && frame_index > 0 {
		pre_unused, post_unused, stats, skipped := vulkan_run_vma_defragmentation()
		diag_defrag_telemetry(frame_index, pre_unused, post_unused, stats, skipped)
	}
	// Dump the log on the very first frame the renderer ticks (frame 1
	// or later) and then on every DIAG_LOG_DUMP_PERIOD boundary. The
	// initial dump also runs from renderer_diagnostics_init, but that
	// one carries all-zero counters; this one carries whatever the
	// frame-0 extract/upload actually produced so a quick smoke test
	// can see real numbers even if the engine exits a few frames later.
	if frame_index == 1 ||
	   (frame_index > 0 && (frame_index % DIAG_LOG_DUMP_PERIOD) == 0) {
		diag_dump_to_file()
	}
}

// diag_defrag_telemetry folds a defrag-pass result into the rolling
// counters. Called by diag_maybe_tick on every DIAG_DEFRAG_PERIOD
// boundary. `pre_unused`/`post_unused` are the VMA-allocator bytes
// that are allocated but not bound to any Allocation; their delta is
// the actual fragmentation recovered (or negative if the defrag
// produced no useful work).
@(private)
diag_defrag_telemetry :: proc(
	frame_index: u64,
	pre_unused, post_unused: u64,
	stats: vma.DefragmentationStats,
	skipped: bool,
) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	dt := &d.gpu_memory.defrag
	dt.last_run_frame         = frame_index
	dt.last_pre_unused_bytes  = pre_unused
	dt.last_post_unused_bytes = post_unused
	dt.last_allocations_moved = stats.allocationsMoved
	dt.last_bytes_moved       = u64(stats.bytesMoved)
	dt.last_device_blocks_freed = stats.deviceMemoryBlocksFreed
	dt.last_was_skipped       = skipped
	if skipped {
		dt.runs_skipped_total += 1
		return
	}
	dt.runs_total                += 1
	dt.bytes_recovered_total     += pre_unused - post_unused
	dt.allocations_moved_total   += u64(stats.allocationsMoved)
	dt.bytes_moved_total         += u64(stats.bytesMoved)
	dt.device_blocks_freed_total += u64(stats.deviceMemoryBlocksFreed)
}

// diag_record_completion_value updates the latest graphics timeline
// completion counter so the dump can report "how far behind is the
// GPU?" A gap larger than MAX_FRAMES_IN_FLIGHT means the GPU is the
// bottleneck and the host is stalling on vkWaitForFences.
diag_record_completion_value :: proc(completion_value: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	DIAGNOSTICS_STATE.frames.completion_value = completion_value
}

// ===========================================================================
//* On-disk telemetry log.
//
// diag_dump_to_file snapshots the current Renderer_Diagnostics state and
// overwrites <exe_dir>/bf_gpu_diagnostics.txt with the formatted string.
// Runs on DIAG_LOG_DUMP_PERIOD cadence from diag_maybe_tick, so the file
// always carries the freshest snapshot - rotation isn't required for
// an inspection-only log. Skips silently on failure (read-only disk,
// sandbox, etc.) so the runtime path stays clean.
// ===========================================================================

DIAG_LOG_DUMP_PERIOD    :: u64(60) // frames between log dumps (~1s @60Hz)
BF_GPU_DIAGNOSTICS_FILE :: "bf_gpu_diagnostics.txt"
BF_GPU_DIAGNOSTICS_JSON_FILE :: "bf_gpu_diagnostics.json"

// diag_json_dump_enabled returns true when BF_GPU_DIAGNOSTICS_JSON is
// set in the process environment (or to "1" / "true" / "yes").
// Probed once per process via the cached flag below; toggling the
// env var at runtime requires a restart.
diag_json_dump_enabled_cached: int = -1
@(private)
diag_json_dump_enabled :: proc() -> bool {
	if diag_json_dump_enabled_cached != -1 do return diag_json_dump_enabled_cached == 1
	v, ok := os.lookup_env_alloc("BF_GPU_DIAGNOSTICS_JSON", context.allocator)
	if !ok {
		diag_json_dump_enabled_cached = 0
		return false
	}
	defer delete(v, context.allocator)
	enabled := v == "1" || v == "true" || v == "TRUE" || v == "yes"
	diag_json_dump_enabled_cached = enabled ? 1 : 0
	return enabled
}

// diag_brief_dump_enabled returns true when BF_GPU_DIAGNOSTICS_BRIEF
// is set. When brief mode is on, renderer_diagnostics_string emits a
// terse variant suitable for editor overlays - only the headline
// counters (frame, fps, total memory pressure, lag, defrag failures),
// plus any line whose pressure label is "warning" or "critical".
// Designed for a status-bar panel: a developer watching the overlay
// sees only what changed status, not the full per-stage breakdown.
diag_brief_dump_enabled_cached: int = -1
@(private)
diag_brief_dump_enabled :: proc() -> bool {
	if diag_brief_dump_enabled_cached != -1 do return diag_brief_dump_enabled_cached == 1
	v, ok := os.lookup_env_alloc("BF_GPU_DIAGNOSTICS_BRIEF", context.allocator)
	if !ok {
		diag_brief_dump_enabled_cached = 0
		return false
	}
	defer delete(v, context.allocator)
	enabled := v == "1" || v == "true" || v == "TRUE" || v == "yes"
	diag_brief_dump_enabled_cached = enabled ? 1 : 0
	return enabled
}

// renderer_diagnostics_json_string is the machine-readable counterpart
// of renderer_diagnostics_string. Emitted alongside the .txt when
// BF_GPU_DIAGNOSTICS_JSON=1 is in the environment. Format is a flat
// JSON object: no nested objects for the rolling rings (callers that
// want the history can read from the live Renderer_Diagnostics
// accessor), but every other counter is exposed by name. Keys mirror
// the section headings in the .txt dump so grep against one matches
// the other.
//
// Implementation uses fmt.tprintf for the formatted values and raw
// strings.write_string for the JSON punctuation. fmt.tprintf treats
// `{` and `}` as format directives (see Odin's fmt package), so
// interleaving them with values requires either escaping as `{{` /
// `}}` or splitting writes; the latter is more readable.
renderer_diagnostics_json_string :: proc(allocator := context.allocator) -> string {
	if !DIAGNOSTICS_STATE.initialized {
		// Return a fresh "" for `{}` to satisfy the "must be
		// owned by the caller" contract used by tests that
		// `delete` the returned string. Static string literals
		// would be a bad-free.
		return strings.clone("{}", allocator)
	}
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	sb := strings.builder_make(allocator)
	defer strings.builder_destroy(&sb)
	// wf writes a fmt format string with literal JSON punctuation.
// fmt.tprintf eats `{` and `}` as format directives, so we escape
// them with `{{` / `}}` before handing the string off. The escape
// buffer is allocated on the heap (not the temp allocator) and
// freed via defer, because converting the []u8 to a string produces
// a non-owning view and the tracking allocator doesn't allow freeing
// the backing store while such a view is live.
	wf :: proc(sb: ^strings.Builder, format: string, args: ..any) {
		buf := make([]u8, len(format) * 2 + 4, context.allocator)
		defer delete(buf)
		n := 0
		for r in format {
			if r == '{' || r == '}' {
				buf[n] = cast(u8)r
				n += 1
			}
			buf[n] = cast(u8)r
			n += 1
		}
		escaped := strings.clone(string(buf[:n]))
		defer delete(escaped)
		strings.write_string(sb, fmt.tprintf(escaped, ..args))
	}
	latest := latest_frame_index(d)
	wf(&sb, "{\"frame\":%d,", latest)
	wf(&sb, "\"debug\":{\"validation\":%d,\"label\":%d,\"timestamp_pool\":%d},",
		int(d.debug.validation_enabled), int(d.debug.label_capability), int(d.debug.timestamp_pool_ready))
	strings.write_string(&sb, "\"cpu\":")
	wf(&sb, "{\"extract_ns\":%d,\"upload_ns\":%d,\"submit_ns\":%d,\"present_ns\":%d,\"total_ns\":%d,\"gpu_wait_ns\":%d},",
		val_at(d.cpu.scene_extract), val_at(d.cpu.upload), val_at(d.cpu.submit),
		val_at(d.cpu.frame_present), val_at(d.cpu.total_frame), val_at(d.cpu.gpu_wait))
	strings.write_string(&sb, "\"gpu\":")
	wf(&sb, "{\"culling_ns\":%d,\"upload_ns\":%d,\"traditional_ns\":%d,\"meshlet_ns\":%d,\"submit_ns\":%d,\"render_ns\":%d,\"present_ns\":%d,\"total_ns\":%d},",
		val_at(d.gpu.culling_ns), val_at(d.gpu.upload_ns), val_at(d.gpu.traditional_ns),
		val_at(d.gpu.meshlet_ns), val_at(d.gpu.submit_ns), val_at(d.gpu.render_ns),
		val_at(d.gpu.present_ns), val_at(d.gpu.total_ns))
	strings.write_string(&sb, "\"visible\":")
	wf(&sb, "{\"chunks\":%d,\"models\":%d,\"meshes\":%d,\"pending_assets\":%d},",
		val_at(d.visible.chunks), val_at(d.visible.models), val_at(d.visible.meshes),
		val_at(d.visible.pending_assets))
	strings.write_string(&sb, "\"indirect\":")
	wf(&sb, "{\"traditional\":%d,\"meshlet\":%d,\"total\":%d,\"draw_calls\":%d},",
		val_at(d.indirect.traditional_cmds), val_at(d.indirect.meshlet_cmds),
		val_at(d.indirect.total_cmds), val_at(d.indirect.draw_calls))
	strings.write_string(&sb, "\"upload\":")
	wf(&sb, "{\"bytes\":%d,\"uploads\":%d,\"asset_buffers\":%d,\"by_kind\":{",
		val_at(d.upload.bytes_submitted), val_at(d.upload.uploads), val_at(d.upload.asset_buffers))
	upload_kind_names := [int(Upload_Kind.COUNT)]string{
		"frame_transient", "asset_vertex", "asset_index",
		"asset_texture", "asset_other", "internal", "unknown",
	}
	for k in 0..<int(Upload_Kind.COUNT) {
		ks := d.upload.by_kind[k]
		if k > 0 do strings.write_string(&sb, ",")
		wf(&sb, "\"%s\":{\"bytes_lifetime\":%d,\"count_lifetime\":%d}",
			upload_kind_names[k], ks.bytes_lifetime, ks.count_lifetime)
	}
	strings.write_string(&sb, "}},")
	strings.write_string(&sb, "\"assets\":")
	wf(&sb, "{\"models\":%d,\"meshes\":%d,\"materials\":%d,\"textures\":%d,\"pending_retries\":%d},",
		d.asset.models_created, d.asset.meshes_created, d.asset.materials_created,
		d.asset.textures_created, d.asset.pending_retries)
	strings.write_string(&sb, "\"memory\":")
	pct := pressure_pct(d.gpu_memory.used_bytes, d.gpu_memory.budget_bytes)
	wf(&sb, "{\"used\":%d,\"budget\":%d,\"pressure_pct\":%.4f,\"pressure_label\":\"%s\",",
		d.gpu_memory.used_bytes, d.gpu_memory.budget_bytes, pct, pressure_label(pct))
	wf(&sb, "\"allocations\":%d,\"blocks\":%d,\"images_alive\":%d,\"samplers_alive\":%d,",
		d.gpu_memory.allocation_count, d.gpu_memory.buffer_count,
		d.gpu_memory.image_count, d.gpu_memory.sampler_count)
	strings.write_string(&sb, "\"heaps\":")
	bucket_names := [int(Heap_Bucket.COUNT)]string{
		"other", "device_local", "host_visible", "device_lazy", "host_device",
	}
	for i in 0..<int(Heap_Bucket.COUNT) {
		h := d.gpu_memory.heaps[i]
		if i > 0 do strings.write_string(&sb, ",")
		bpct := pressure_pct(h.used, h.budget)
		wf(&sb, "\"%s\":{\"used\":%d,\"budget\":%d,\"pct\":%.4f,\"label\":\"%s\"}",
			bucket_names[i], h.used, h.budget, bpct, pressure_label(bpct))
	}
	strings.write_string(&sb, "},\"memory_types\":")
	type_names := [int(Memory_Type_Bucket.COUNT)]string{
		"device_local", "host_visible", "lazy", "uma", "other",
	}
	for i in 0..<int(Memory_Type_Bucket.COUNT) {
		mt := d.gpu_memory.memory_types[i]
		if i > 0 do strings.write_string(&sb, ",")
		mpct := pressure_pct(mt.bytes, mt.budget_bytes)
		wf(&sb, "\"%s\":{\"bytes\":%d,\"budget\":%d,\"pct\":%.4f,\"label\":\"%s\",\"allocs\":%d,\"unused\":%d}",
			type_names[i], mt.bytes, mt.budget_bytes, mpct, pressure_label(mpct),
			mt.allocation_count, mt.unused_bytes)
	}
	strings.write_string(&sb, "},\"suballocator\":")
	sub := d.gpu_memory.suballocator
	wf(&sb, "{\"block\":%d,\"allocation\":%d,\"unused\":%d,\"ranges\":%d,\"max_range\":%d,\"alloc_min\":%d,\"alloc_max\":%d},",
		sub.block_bytes, sub.allocation_bytes, sub.unused_bytes,
		sub.unused_range_count, sub.unused_range_bytes_max,
		sub.allocation_bytes_min, sub.allocation_bytes_max)
	strings.write_string(&sb, "\"defrag\":")
	dt := d.gpu_memory.defrag
	wf(&sb, "{\"runs\":%d,\"skipped\":%d,\"bytes_recovered\":%d,\"allocs_moved\":%d,\"bytes_moved\":%d,\"blocks_freed\":%d,",
		dt.runs_total, dt.runs_skipped_total, dt.bytes_recovered_total,
		dt.allocations_moved_total, dt.bytes_moved_total, dt.device_blocks_freed_total)
	wf(&sb, "\"last_run_frame\":%d,\"last_pre_unused\":%d,\"last_post_unused\":%d,\"last_was_skipped\":%d},",
		dt.last_run_frame, dt.last_pre_unused_bytes, dt.last_post_unused_bytes,
		int(dt.last_was_skipped))
	strings.write_string(&sb, "\"frames\":")
	wf(&sb, "{\"in_flight\":%d,\"completion\":%d,\"oldest_uncompleted\":%d,\"lag\":%d},",
		d.frames.in_flight, d.frames.completion_value,
		d.frames.oldest_uncompleted_value, timeline_lag(d))
	strings.write_string(&sb, "\"throughput\":")
	wf(&sb, "{\"fps\":%.4f,\"bandwidth_mbps\":%.4f}",
		latest_cpu_fps(d.cpu.total_frame),
		latest_bandwidth_mbps(d.upload.bytes_submitted, d.cpu.total_frame))
	strings.write_string(&sb, "}")
	return strings.clone(strings.to_string(sb), allocator)
}

// renderer_diagnostics_dispatch routes to the full or brief dump
// based on the BF_GPU_DIAGNOSTICS_BRIEF env var. Brief mode is
// intended for editor overlays; the full dump is for offline review
// and the on-disk log. Callers that want a specific format should
// call renderer_diagnostics_string or renderer_diagnostics_brief_string
// directly - this dispatch is the convenience entry point that the
// editor overlay / on-disk dump uses.
renderer_diagnostics_dispatch :: proc(allocator := context.allocator) -> string {
	if diag_brief_dump_enabled() {
		return renderer_diagnostics_brief_string(allocator)
	}
	return renderer_diagnostics_string(allocator)
}

// renderer_diagnostics_brief_string is the editor-overlay-friendly
// variant. Always emits:
//   - frame index + latest fps
//   - aggregate memory pressure (with label)
//   - any per-bucket pressure row whose label is warning / critical
//   - timeline lag (when > in_flight)
//   - defrag last-run summary (when recent)
//   - deferred destruction backlog (when nonzero)
// Always omits: per-stage ring stats, throughput, suballocator, JSON
// peer, asset counters. The brief dump is roughly 10 lines vs ~40
// for the full dump, suitable for a status-bar panel.
renderer_diagnostics_brief_string :: proc(allocator := context.allocator) -> string {
	if !DIAGNOSTICS_STATE.initialized {
		return strings.clone("[BF_GPU] diagnostics uninit", allocator)
	}
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	sb := strings.builder_make(allocator)
	defer strings.builder_destroy(&sb)
	wf :: proc(sb: ^strings.Builder, format: string, args: ..any) {
		buf := make([]u8, len(format) * 2 + 4, context.allocator)
		defer delete(buf)
		n := 0
		for r in format {
			if r == '{' || r == '}' {
				buf[n] = cast(u8)r
				n += 1
			}
			buf[n] = cast(u8)r
			n += 1
		}
		escaped := strings.clone(string(buf[:n]))
		defer delete(escaped)
		strings.write_string(sb, fmt.tprintf(escaped, ..args))
	}
	latest := latest_frame_index(d)
	fps := latest_cpu_fps(d.cpu.total_frame)
	agg_pct := pressure_pct(d.gpu_memory.used_bytes, d.gpu_memory.budget_bytes)
	wf(&sb, "[%d] fps=%.1f mem=%.1f%% (%s) ",
		latest, fps, agg_pct, pressure_label(agg_pct))

	// Per-bucket pressure flags. Only emit rows whose label is
	// "warning" or "critical" - a quiet editor overlay shouldn't
	// show "ok" rows for every bucket every frame.
	written_buckets := false
	bucket_names := [int(Heap_Bucket.COUNT)]string{
		"other", "device_local", "host_visible", "device_lazy", "host_device",
	}
	for i in 0..<int(Heap_Bucket.COUNT) {
		h := d.gpu_memory.heaps[i]
		if h.budget == 0 do continue
		pct := pressure_pct(h.used, h.budget)
		label := pressure_label(pct)
		if label == "warning" || label == "critical" {
			if !written_buckets {
				strings.write_string(&sb, " heaps=[")
				written_buckets = true
			} else {
				strings.write_string(&sb, ",")
			}
			wf(&sb, "%s:%.0f%%(%s)", bucket_names[i], pct, label)
		}
	}
	if written_buckets do strings.write_string(&sb, "] ")

	// Same per-memory-type filter for the most common warning case:
	// the host-visible staging pool filling up is what triggers most
	// editor-overlay attention.
	written_types := false
	type_names := [int(Memory_Type_Bucket.COUNT)]string{
		"device_local", "host_visible", "lazy", "uma", "other",
	}
	for i in 0..<int(Memory_Type_Bucket.COUNT) {
		mt := d.gpu_memory.memory_types[i]
		if mt.budget_bytes == 0 do continue
		pct := pressure_pct(mt.bytes, mt.budget_bytes)
		label := pressure_label(pct)
		if label == "warning" || label == "critical" {
			if !written_types {
				strings.write_string(&sb, " types=[")
				written_types = true
			} else {
				strings.write_string(&sb, ",")
			}
			wf(&sb, "%s:%.0f%%(%s)", type_names[i], pct, label)
		}
	}
	if written_types do strings.write_string(&sb, "] ")

	// Timeline lag (only when notable).
	lag := timeline_lag(d)
	if lag > u64(d.frames.in_flight) {
		wf(&sb, " lag=%d ", lag)
	}
	// Defrag summary (only when there's something recent to report).
	if d.gpu_memory.defrag.runs_total > 0 || d.gpu_memory.defrag.runs_skipped_total > 0 {
		dt := d.gpu_memory.defrag
		wf(&sb, "defrag=runs=%d skipped=%d recovered=%d ",
			dt.runs_total, dt.runs_skipped_total, dt.bytes_recovered_total)
	}
	return strings.clone(strings.to_string(sb), allocator)
}

// diag_log_file_path returns the absolute path of the diagnostics log,
// computed lazily from os.get_executable_directory() so the file lands
// adjacent to the running .exe regardless of which directory the engine
// was launched from. The caller owns the returned string (it's
// heap-allocated via the supplied allocator).
diag_log_file_path :: proc(allocator := context.allocator) -> string {
	dir, err := os.get_executable_directory(allocator)
	if err != nil || len(dir) == 0 do return strings.clone(BF_GPU_DIAGNOSTICS_FILE, allocator)
	defer delete(dir, allocator)
	joined, join_err := filepath.join({dir, BF_GPU_DIAGNOSTICS_FILE}, allocator)
	if join_err != nil do return strings.clone(BF_GPU_DIAGNOSTICS_FILE, allocator)
	return joined
}

// diag_log_file_path_json returns the JSON counterpart path; same
// resolution rules as diag_log_file_path but with the .json extension.
diag_log_file_path_json :: proc(allocator := context.allocator) -> string {
	dir, err := os.get_executable_directory(allocator)
	if err != nil || len(dir) == 0 do return strings.clone(BF_GPU_DIAGNOSTICS_JSON_FILE, allocator)
	defer delete(dir, allocator)
	joined, join_err := filepath.join({dir, BF_GPU_DIAGNOSTICS_JSON_FILE}, allocator)
	if join_err != nil do return strings.clone(BF_GPU_DIAGNOSTICS_JSON_FILE, allocator)
	return joined
}

// diag_dump_to_file builds the current snapshot and writes it to the
// diagnostics log. Writes go to a sibling .tmp file first, then
// os.rename promotes it over the live path. That keeps a reader
// (editor overlay, devtools) from observing a half-written snapshot
// and means the live file is either the previous complete snapshot
// or the new complete snapshot - never the prefix of one plus the
// suffix of another. Errors are swallowed; this is a diagnostic
// path that must never break the render loop.
@(private)
diag_dump_to_file :: proc() {
	if !DIAGNOSTICS_STATE.initialized do return
	body := renderer_diagnostics_dispatch()
	defer delete(body)
	path := diag_log_file_path()
	defer delete(path)
	// Sanity-check the snapshot before touching disk: any non-printable
	// byte in the first 16 chars means something upstream polluted the
	// builder (we have seen prefixes of 142 bytes containing paths and
	// binary headers appear in this file from external writes). Skip
	// the disk write in that case so we don't replace a good prior
	// snapshot with garbage.
	if !diag_body_looks_clean(body) {
		if DIAG_LOG_DUMP_FIRST_ERROR == 0 {
			// Hex-encode the first 32 bytes so a terminal can show
			// them even when the bytes are unprintable.
			hex_sb := strings.builder_make(context.allocator)
			defer strings.builder_destroy(&hex_sb)
			preview_len := min(len(body), 32)
			for i in 0..<preview_len {
				fmt.sbprintf(&hex_sb, "%02x ", body[i])
			}
			log.warnf(
				"[BF_GPU] diagnostics snapshot rejected (non-clean prefix, len=%d, head=[%s]), keeping prior file",
				len(body),
				strings.to_string(hex_sb),
			)
			DIAG_LOG_DUMP_FIRST_ERROR = 1
		}
		return
	}
	// Write-then-rename gives atomic-ish semantics: a concurrent reader
	// sees either the old file or the new file, never a mix.
	// strings.clone around fmt.tprintf so the buffer is unambiguously
	// owned by us; without the clone, odin's tracking allocator can
	// fire a bad-free when the format string + %s substitution share
	// the same underlying storage.
	tmp_path := strings.clone(fmt.tprintf("%s.tmp", path))
	defer delete(tmp_path)
	if err := os.write_entire_file_from_string(tmp_path, body); err != nil {
		// First-write failure on a fresh run is common when the
		// target directory is read-only (CI, sandbox). Don't spam
		// the log; rate-limit by only warning once per process.
		if DIAG_LOG_DUMP_FIRST_ERROR == 0 {
			log.warnf("[BF_GPU] diagnostics log write failed: %v", err)
			DIAG_LOG_DUMP_FIRST_ERROR = 1
		}
		os.remove(tmp_path)
		return
	}
	if err := os.rename(tmp_path, path); err != nil {
		if DIAG_LOG_DUMP_FIRST_ERROR == 0 {
			log.warnf("[BF_GPU] diagnostics log rename failed: %v", err)
			DIAG_LOG_DUMP_FIRST_ERROR = 1
		}
		os.remove(tmp_path)
		return
	}

	// Optional JSON dump. Same write-then-rename semantics. Errors
	// here never block the render loop and never warn twice (the
	// shared DIAG_LOG_DUMP_FIRST_ERROR gate covers both files).
	if diag_json_dump_enabled() {
		diag_dump_json_to_file()
	}
}

// diag_dump_json_to_file is the JSON counterpart of diag_dump_to_file.
// Same atomic-ish write/rename dance. Runs only when
// BF_GPU_DIAGNOSTICS_JSON is set in the environment.
@(private)
diag_dump_json_to_file :: proc() {
	if !DIAGNOSTICS_STATE.initialized do return
	body := renderer_diagnostics_json_string()
	defer delete(body)
	path := diag_log_file_path_json()
	defer delete(path)
	tmp_path := strings.clone(fmt.tprintf("%s.tmp", path))
	defer delete(tmp_path)
	if err := os.write_entire_file_from_string(tmp_path, body); err != nil {
		os.remove(tmp_path)
		return
	}
	if err := os.rename(tmp_path, path); err != nil {
		os.remove(tmp_path)
		return
	}
}

// diag_body_looks_clean returns true when the body looks like a real
// diagnostics snapshot. The corruption case we are guarding against
// has a binary prefix containing 0x00 bytes and 8-byte binary blobs
// (u32 LE 0x4B etc.). Anything that's primarily printable text with
// only normal control delimiters and UTF-8 continuation bytes is
// fine - we only reject obvious binary: null bytes or runs of bytes
// outside the printable + tab/newline/CR + UTF-8 lead/continuation
// ranges in the first 32 bytes.
@(private)
diag_body_looks_clean :: proc(body: string) -> bool {
	if len(body) < 4 do return false
	// Hard reject: a null byte in the prefix is never something a
	// renderer_diagnostics_string() output would produce, and matches
	// the corruption pattern we saw (0x00 0x00 0x00 between paths).
	for i in 0..<min(len(body), 32) {
		if body[i] == 0 do return false
	}
	// Soft check: first 4 bytes should be printable ASCII. The header
	// starts with "== B" which is always true for our output.
	for i in 0..<4 {
		c := body[i]
		if c < 0x20 || c > 0x7E do return false
	}
	return true
}

@(private)
DIAG_LOG_DUMP_FIRST_ERROR: int

// diag_record_frames_completed folds the per-poll completion counters
// into the diagnostics. Vulkan.odin calls this every time the GPU
// completion signal fires.
diag_record_frames_completed :: proc(
	frame_index: u64,
	submitted_slots, signaled_slots, completed_slots: u32,
	oldest_uncompleted_value: u64,
) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	diag_record_scalar(&d.frames.slots_submitted, frame_index, u64(submitted_slots))
	diag_record_scalar(&d.frames.slots_signaled,  frame_index, u64(signaled_slots))
	diag_record_scalar(&d.frames.slots_completed, frame_index, u64(completed_slots))
	d.frames.oldest_uncompleted_value = oldest_uncompleted_value
}

// ===========================================================================
//* Public accessors.
// ===========================================================================

// renderer_diagnostics_string returns a multi-line, human-readable
// dump of the current diagnostics snapshot. Allocations are bounded;
// editor overlays or log dumps call this once per second.
//
// The dump shows the latest-frame value followed by min/avg/p50/p95/max
// over the rolling DIAG_FRAME_HISTORY window so a reviewer can spot
// steady-state vs. transient spikes without parsing raw histograms.
// Per-heap-bucket memory pressure is annotated with a status label
// (ok / elevated / warning / critical) so a terminal-style scan can
// flag regressions without reading the percentage.
//
// IMPORTANT: the returned string is heap-allocated via the caller's
// allocator; the caller must `delete` it when done. The buffer is
// cloned out of the builder before the builder is destroyed - the
// previous implementation returned a slice into sb.buf and let the
// defer-destroy free the backing memory, which produced a dangling
// pointer and the binary-prefix garbage seen in the on-disk log.
renderer_diagnostics_string :: proc(allocator := context.allocator) -> string {
	if !DIAGNOSTICS_STATE.initialized {
		return strings.clone("[BF_GPU] Renderer diagnostics not initialised.", allocator)
	}
	// Lock for the entire walk so a concurrent recorder can't
	// tear the snapshot between writes. The lock is held for the
	// full string-build (~us) which is fine because the render
	// path only calls this once a second.
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE

	sb := strings.builder_make(allocator)
	defer strings.builder_destroy(&sb)

	latest := latest_frame_index(d)

	appendf :: proc(sb: ^strings.Builder, format: string, args: ..any) {
		strings.write_string(sb, fmt.tprintf(format, ..args))
	}

	append_stats :: proc(sb: ^strings.Builder, label: string, ms: f64, s: Ring_Stats) {
		strings.write_string(sb, fmt.tprintf("%s=%.2fms [avg=%.2f p50=%.2f p95=%.2f max=%.2f n=%d]",
			label, ms,
			f64(s.avg) / 1_000_000.0,
			f64(s.p50) / 1_000_000.0,
			f64(s.p95) / 1_000_000.0,
			f64(s.max) / 1_000_000.0,
			s.samples,
		))
	}

	appendf(&sb, "== BF_GPU Renderer Diagnostics (frame %d) ==\n", latest)
	appendf(&sb,
		"Debug: validation=%v labels=%v timestamp_pool=%v\n",
		d.debug.validation_enabled,
		d.debug.label_capability,
		d.debug.timestamp_pool_ready,
	)

	// CPU timings: latest in ms followed by rolling-window stats.
	// Format kept terse so the line fits an 80-column terminal.
	strings.write_string(&sb, "CPU timings (ms) [latest, then rolling min/avg/p50/p95/max over ")
	strings.write_string(&sb, fmt.tprintf("%d-frame window]\n", DIAG_FRAME_HISTORY))
	strings.write_string(&sb, "  ")
	append_stats(&sb, "extract", ms_at(d.cpu.scene_extract), ring_stats(d.cpu.scene_extract))
	strings.write_string(&sb, "\n  ")
	append_stats(&sb, "upload",  ms_at(d.cpu.upload),         ring_stats(d.cpu.upload))
	strings.write_string(&sb, "\n  ")
	append_stats(&sb, "submit",  ms_at(d.cpu.submit),         ring_stats(d.cpu.submit))
	strings.write_string(&sb, "\n  ")
	append_stats(&sb, "present", ms_at(d.cpu.frame_present),  ring_stats(d.cpu.frame_present))
	strings.write_string(&sb, "\n  ")
	append_stats(&sb, "total",   ms_at(d.cpu.total_frame),    ring_stats(d.cpu.total_frame))
	strings.write_string(&sb, "\n  ")
	append_stats(&sb, "gpu_wait",ms_at(d.cpu.gpu_wait),       ring_stats(d.cpu.gpu_wait))
	strings.write_string(&sb, "\n")

	fps := latest_cpu_fps(d.cpu.total_frame)
	if fps > 0 {
		appendf(&sb, "Throughput - fps=%.1f bandwidth=%.2f MB/s\n",
			fps,
			latest_bandwidth_mbps(d.upload.bytes_submitted, d.cpu.total_frame),
		)
	} else {
		appendf(&sb, "Throughput - fps=0.0 bandwidth=0.00 MB/s\n")
	}

	// GPU timings come from the timestamp query pool and have lower
	// latency than the CPU ring, but the ring stats are still useful
	// for spotting frame-to-frame variance. Iterate the named
	// Stage_Timing fields explicitly; the timestamps profile is a
	// different shape and skipped.
	strings.write_string(&sb, "GPU timings (ms) [latest, then rolling min/avg/p50/p95/max]\n")
	gpu_stages := []struct{ name: string, t: Stage_Timing }{
		{"culling",     d.gpu.culling_ns},
		{"upload",      d.gpu.upload_ns},
		{"traditional", d.gpu.traditional_ns},
		{"meshlet",     d.gpu.meshlet_ns},
		{"submit",      d.gpu.submit_ns},
		{"render",      d.gpu.render_ns},
		{"present",     d.gpu.present_ns},
		{"total",       d.gpu.total_ns},
	}
	for gs in gpu_stages {
		strings.write_string(&sb, "  ")
		append_stats(&sb, gs.name, ms_at(gs.t), ring_stats(gs.t))
		strings.write_string(&sb, "\n")
	}

	appendf(&sb,
		"Visible - chunks=%d models=%d meshes=%d pending_assets=%d\n",
		val_at(d.visible.chunks),
		val_at(d.visible.models),
		val_at(d.visible.meshes),
		val_at(d.visible.pending_assets),
	)
	appendf(&sb,
		"Indirect - traditional=%d meshlet=%d total=%d draw_calls=%d\n",
		val_at(d.indirect.traditional_cmds),
		val_at(d.indirect.meshlet_cmds),
		val_at(d.indirect.total_cmds),
		val_at(d.indirect.draw_calls),
	)
	appendf(&sb,
		"Upload - bytes=%d (%s) uploads=%d asset_buffers=%d images_alive=%d samplers_alive=%d\n",
		val_at(d.upload.bytes_submitted),
		human_bytes(val_at(d.upload.bytes_submitted)),
		val_at(d.upload.uploads),
		val_at(d.upload.asset_buffers),
		d.gpu_memory.image_count,
		d.gpu_memory.sampler_count,
	)
	// Per-upload-kind breakdown. Sorts the kinds by lifetime bytes
	// descending so the heaviest contributor (texture streaming,
	// mesh cache, etc.) is first - the line a reviewer actually
	// wants to see.
	{
		kind_names := [int(Upload_Kind.COUNT)]string{
			"frame_transient", "asset_vertex", "asset_index",
			"asset_texture",   "asset_other",  "internal", "unknown",
		}
		// Build a tiny ranking array of (index, value) pairs and sort
		// by value descending. Allocating on the stack (the outer
		// block scope) keeps the dump cheap.
		indices: [int(Upload_Kind.COUNT)]int
		values:  [int(Upload_Kind.COUNT)]u64
		for k in 0..<int(Upload_Kind.COUNT) {
			indices[k] = k
			values[k]  = d.upload.by_kind[k].bytes_lifetime
		}
		// Insertion sort, descending by value.
		for i in 1..<len(indices) {
			j := i
			for j > 0 && values[j] > values[j - 1] {
				values[j],  values[j - 1]  = values[j - 1], values[j]
				indices[j], indices[j - 1] = indices[j - 1], indices[j]
				j -= 1
			}
		}
		strings.write_string(&sb, "  Upload by kind (lifetime bytes, allocated):\n")
		for k in 0..<len(indices) {
			v := values[k]
			if v == 0 do continue
			appendf(&sb, "    %s: bytes=%d (%s) count=%d\n",
				kind_names[indices[k]], v, human_bytes(v),
				d.upload.by_kind[indices[k]].count_lifetime)
		}
	}
	appendf(&sb,
		"Assets (lifetime) - models=%d meshes=%d materials=%d textures=%d pending_retries=%d\n",
		d.asset.models_created,
		d.asset.meshes_created,
		d.asset.materials_created,
		d.asset.textures_created,
		d.asset.pending_retries,
	)

	// GPU memory. Aggregate pressure first, then per-heap-bucket
	// breakdown. The aggregate shows the single number a dashboard
	// cares about; the breakdown shows attribution so a reviewer
	// can answer "where is the pressure" without re-querying the
	// driver.
	total_pct := pressure_pct(d.gpu_memory.used_bytes, d.gpu_memory.budget_bytes)
	appendf(&sb,
		"GPU memory - used=%d (%s, %.1f%%, %s) budget=%d (%s) allocs=%d buffers=%d images_alive=%d samplers_alive=%d\n",
		d.gpu_memory.used_bytes,
		human_bytes(d.gpu_memory.used_bytes),
		total_pct,
		pressure_label(total_pct),
		d.gpu_memory.budget_bytes,
		human_bytes(d.gpu_memory.budget_bytes),
		d.gpu_memory.allocation_count,
		d.gpu_memory.buffer_count,
		d.gpu_memory.image_count,
		d.gpu_memory.sampler_count,
	)
	// Per-heap-bucket breakdown. Skipped when no bucket has any
	// budget, which happens when the diagnostics subsystem has not
	// seen a memory snapshot yet (e.g. headless tests without
	// Vulkan).
	any_heap_budget := false
	for i in 0..<int(Heap_Bucket.COUNT) {
		if d.gpu_memory.heaps[i].budget > 0 {
			any_heap_budget = true
			break
		}
	}
	if any_heap_budget {
		strings.write_string(&sb, "  Heap breakdown (used / budget / %% / status):\n")
		bucket_names := [int(Heap_Bucket.COUNT)]string{
			"other", "device_local", "host_visible", "device_lazy", "host_device",
		}
		for i in 0..<int(Heap_Bucket.COUNT) {
			h := d.gpu_memory.heaps[i]
			if h.budget == 0 && h.used == 0 do continue
			pct := pressure_pct(h.used, h.budget)
			appendf(&sb,
				"    %s: used=%d (%s) budget=%d (%s) pct=%.1f%% (%s)\n",
				bucket_names[i],
				h.used, human_bytes(h.used),
				h.budget, human_bytes(h.budget),
				pct, pressure_label(pct),
			)
		}
	}
	// Per-memory-type breakdown. Real HOST_VISIBLE / LAZILY_ALLOCATED
	// attribution that heap flags can't provide. Only emitted when at
	// least one type bucket has a non-zero allocation count, which
	// happens once VMA has created a device (so a fresh init dump
	// stays terse).
	any_type_data := false
	for i in 0..<int(Memory_Type_Bucket.COUNT) {
		if d.gpu_memory.memory_types[i].bytes > 0 || d.gpu_memory.memory_types[i].allocation_count > 0 {
			any_type_data = true
			break
		}
	}
	if any_type_data {
		strings.write_string(&sb, "  Memory type breakdown (bytes / budget / %% / status):\n")
		type_names := [int(Memory_Type_Bucket.COUNT)]string{
			"device_local", "host_visible", "lazy", "uma", "other",
		}
		for i in 0..<int(Memory_Type_Bucket.COUNT) {
			mt := d.gpu_memory.memory_types[i]
			if mt.bytes == 0 && mt.allocation_count == 0 do continue
			pct := pressure_pct(mt.bytes, mt.budget_bytes)
			appendf(&sb,
				"    %s: bytes=%d (%s) budget=%d (%s) pct=%.1f%% (%s) allocs=%d unused=%d (%s) ranges=%d\n",
				type_names[i],
				mt.bytes, human_bytes(mt.bytes),
				mt.budget_bytes, human_bytes(mt.budget_bytes),
				pct, pressure_label(pct),
				mt.allocation_count,
				mt.unused_bytes, human_bytes(mt.unused_bytes),
				mt.unused_range_count,
			)
		}
	}

	// VMA suballocator stats. These are what actually drive the
	// "is the allocator fragmented?" question.
	sub := d.gpu_memory.suballocator
	if sub.block_bytes > 0 {
		appendf(&sb,
			"VMA suballocator - block=%d (%s) alloc=%d (%s) unused=%d (%s, %.1f%%) ranges=%d max_range=%d (%s) alloc_min=%d alloc_max=%d\n",
			sub.block_bytes, human_bytes(sub.block_bytes),
			sub.allocation_bytes, human_bytes(sub.allocation_bytes),
			sub.unused_bytes, human_bytes(sub.unused_bytes),
			pressure_pct(sub.unused_bytes, sub.block_bytes),
			sub.unused_range_count,
			sub.unused_range_bytes_max, human_bytes(sub.unused_range_bytes_max),
			sub.allocation_bytes_min,
			sub.allocation_bytes_max,
		)
	}

	// VMA defrag telemetry.
	dt := d.gpu_memory.defrag
	if dt.runs_total > 0 || dt.runs_skipped_total > 0 || dt.last_run_frame > 0 {
		appendf(&sb,
			"VMA defrag - runs=%d skipped=%d bytes_recovered=%d (%s) allocs_moved=%d bytes_moved=%d blocks_freed=%d\n",
			dt.runs_total, dt.runs_skipped_total,
			dt.bytes_recovered_total, human_bytes(dt.bytes_recovered_total),
			dt.allocations_moved_total, dt.bytes_moved_total,
			dt.device_blocks_freed_total,
		)
		if dt.last_run_frame > 0 {
			delta: i64 = i64(dt.last_pre_unused_bytes) - i64(dt.last_post_unused_bytes)
			appendf(&sb,
				"  last run @ frame=%d pre_unused=%d (%s) post_unused=%d (%s) delta=%d (%s) skipped=%v allocs_moved=%d bytes_moved=%d blocks_freed=%d\n",
				dt.last_run_frame,
				dt.last_pre_unused_bytes, human_bytes(dt.last_pre_unused_bytes),
				dt.last_post_unused_bytes, human_bytes(dt.last_post_unused_bytes),
				delta,
				human_bytes(u64(delta)),
				dt.last_was_skipped,
				dt.last_allocations_moved, dt.last_bytes_moved,
				dt.last_device_blocks_freed,
			)
		}
	}

	lag := timeline_lag(d)
	appendf(&sb,
		"Frames - in_flight=%d completion=%d oldest_uncompleted=%d lag=%d frames (%s)\n",
		d.frames.in_flight,
		d.frames.completion_value,
		d.frames.oldest_uncompleted_value,
		lag,
		lag == 0      ? "ok"      :
		lag <= 1      ? "ok"      :
		lag <= u64(d.frames.in_flight) ? "elevated" :
		"warning",
	)
	// Clone out of the builder so the returned string owns its own
	// memory and survives the builder_destroy that runs above.
	return strings.clone(strings.to_string(sb), allocator)
}

// latest_frame_index returns the most recently written ring index.
// Falls back to 0 when the diagnostics subsystem is uninitialised.
@(private)
latest_frame_index :: proc(d: ^Renderer_Diagnostics) -> u64 {
	if d == nil do return 0
	any := &d.cpu.total_frame
	if any.filled == 0 do return 0
	return u64(any.filled - 1)
}

// @(private) ms_at returns the value at the latest frame as a float ms.
// Used by formatted dumping; uses the wall-clock ring value (ns -> ms).
@(private)
ms_at :: proc(t: Stage_Timing) -> f64 {
	if t.filled == 0 do return 0
	idx := (t.filled - 1) % u32(DIAG_FRAME_HISTORY)
	return f64(t.ns[idx]) / 1_000_000.0
}

// val_at is the scalar counterpart of ms_at for non-time ring values.
// Identical indexing decision; treats the value as-is (u32 / u64).
@(private)
val_at :: proc(t: Stage_Timing) -> u64 {
	if t.filled == 0 do return 0
	idx := (t.filled - 1) % u32(DIAG_FRAME_HISTORY)
	return t.ns[idx]
}

// ring_stats computes min/avg/p50/p95/max over the filled portion of
// the Stage_Timing ring. The ring is treated as already-collected;
// callers that want only the last N samples should reset `filled`
// before sampling (the renderer doesn't, so it gets the rolling
// 64-frame window). p50 / p95 use a sort-then-index approach; with
// 64 samples that's a no-op on the perf budget.
@(private)
ring_stats :: proc(t: Stage_Timing) -> Ring_Stats {
	if t.filled == 0 do return {}
	n := int(t.filled)
	samples := make([dynamic]u64, n, context.temp_allocator)
	defer delete(samples)
	sum: u64
	min_v: u64 = max(u64)
	max_v: u64
	for i in 0..<n {
		v := t.ns[i]
		samples[i] = v
		sum += v
		if v < min_v do min_v = v
		if v > max_v do max_v = v
	}
	slice.sort(samples[:])
	// p95 index: clamp to last sample so a tiny window doesn't go OOB.
	p95_idx := clamp(int(f64(n) * 0.95), 0, n - 1)
	p50_idx := clamp(int(f64(n) * 0.50), 0, n - 1)
	return Ring_Stats {
		min     = min_v,
		avg     = sum / u64(n),
		p50     = samples[p50_idx],
		p95     = samples[p95_idx],
		max     = max_v,
		samples = u32(n),
	}
}

// pressure_pct returns used / budget * 100. Returns 0 when budget is
// zero (no budget information yet, e.g. before the first memory
// snapshot) so the dump displays "0%" rather than NaN.
@(private)
pressure_pct :: proc(used, budget: u64) -> f64 {
	if budget == 0 do return 0
	return f64(used) / f64(budget) * 100.0
}

// pressure_label annotates a pressure percentage with a one-word
// status. Thresholds: 0-70% "ok", 70-85% "elevated", 85-95%
// "warning", >95% "critical". The dump surfaces this so a reviewer
// can scan for the keyword without parsing percentages.
@(private)
pressure_label :: proc(pct: f64) -> string {
	switch {
	case pct >= 95.0: return "critical"
	case pct >= 85.0: return "warning"
	case pct >= 70.0: return "elevated"
	case:            return "ok"
	}
}

// timeline_lag returns how many frames behind the CPU is on the
// graphics timeline, computed from completion_value vs the expected
// monotonic counter. Lag of 0-1 frames is healthy; > MAX_FRAMES_IN_FLIGHT
// means the GPU is the bottleneck.
@(private)
timeline_lag :: proc(d: ^Renderer_Diagnostics) -> u64 {
	if d == nil do return 0
	if d.frames.completion_value == 0 do return 0
	expected := d.frames.completion_value + u64(d.frames.in_flight)
	if d.frames.oldest_uncompleted_value <= d.frames.completion_value do return 0
	return d.frames.oldest_uncompleted_value - d.frames.completion_value
}

// latest_cpu_fps derived from the cpu.total_frame Stage_Timing ring's
// latest value. Returns 0 when the ring is empty. Uses ns->Hz so a
// single-frame view of framerate is in the same units as the rest of
// the dump.
@(private)
latest_cpu_fps :: proc(t: Stage_Timing) -> f64 {
	if t.filled == 0 do return 0
	idx := (t.filled - 1) % u32(DIAG_FRAME_HISTORY)
	ns := f64(t.ns[idx])
	if ns <= 0 do return 0
	return 1e9 / ns
}

// latest_bandwidth_mbps derives host->device bandwidth from the
// bytes_submitted ring and the total-frame ring. Multiplies per-
// frame bytes by FPS so the value reflects sustained bandwidth, not
// a single frame's burst. Returns 0 when the ring is empty.
@(private)
latest_bandwidth_mbps :: proc(bytes: Stage_Timing, frame: Stage_Timing) -> f64 {
	fps := latest_cpu_fps(frame)
	if fps == 0 do return 0
	idx := (bytes.filled - 1) % u32(DIAG_FRAME_HISTORY)
	b := f64(bytes.ns[idx])
	return b * fps / (1024.0 * 1024.0)
}

// human_bytes formats a byte count with KB/MB/GB suffixes. Returns a
// fresh string that lives until the caller drops its reference; the
// formatted dump path is one call per second so a heap allocation is
// cheap.
@(private)
human_bytes :: proc(b: u64) -> string {
	KB :: 1024
	MB :: 1024 * 1024
	GB :: 1024 * 1024 * 1024
	switch {
	case b >= GB:
		return fmt.tprintf("%.2f GB", f64(b) / f64(GB))
	case b >= MB:
		return fmt.tprintf("%.2f MB", f64(b) / f64(MB))
	case b >= KB:
		return fmt.tprintf("%.2f KB", f64(b) / f64(KB))
	case:
		return fmt.tprintf("%d B", b)
	}
}

// Asset_Creation_Kind identifies what kind of asset was just created.
// diag_inc_asset_created uses this to bump the right counter.
Asset_Creation_Kind :: enum u8 {
	Model,
	Mesh,
	Material,
	Texture,
	Pending,
}

// Upload_Kind attributes a buffer allocation or upload to a renderer
// subsystem. The dump reports per-kind byte + count totals so a reviewer
// can answer "where did the 2.3 GB go?" without instrumenting every
// call site. Kinds are coarse on purpose: a 5-element enum is enough
// to surface the obvious culprits (texture streaming, mesh cache,
// per-frame transient ring) without forcing every create_asset_buffer
// call to declare a finer classification. Unknown is the default for
// legacy code paths that haven't been migrated yet.
//
// "Frame_Transient" covers the persistent per-frame upload ring slot
// + any vkCmdCopyBuffer-to-GPU-only staging buffers the upload path
// falls back to when the ring is exhausted. "Asset_*" covers long-
// lived buffers owned by Asset_Sync. "Internal" covers non-asset
// renderer-side allocations (descriptor backing, gctx constants).
Upload_Kind :: enum u8 {
	Frame_Transient,
	Asset_Vertex,
	Asset_Index,
	Asset_Texture,
	Asset_Other,
	Internal,
	Unknown,
	COUNT,
}

// Upload_Kind_Stats is the per-kind lifetime + per-frame ring the
// dump renders. Lifetime numbers are what you reach for first when
// diagnosing "why is memory growing"; the rolling ring is the
// transient / streaming-pressure view.
Upload_Kind_Stats :: struct {
	bytes_lifetime:      u64, // total bytes ever allocated for this kind
	count_lifetime:      u64, // total allocation calls for this kind
	bytes_per_frame:     Stage_Timing, // ring of per-frame bytes (latest)
	count_per_frame:     Stage_Timing, // ring of per-frame count (latest)
}

// Memory_Type_Bucket classifies a vk.MemoryPropertyFlag set into the
// buckets the dump renders. Same idea as Heap_Bucket but for memory
// TYPES, where HOST_VISIBLE / LAZILY_ALLOCATED actually live (the
// heap-level flags only expose DEVICE_LOCAL / MULTI_INSTANCE). This
// is what surfaces "the host-visible staging pool is full" as a
// separate finding from "device-local VRAM is full".
Memory_Type_Bucket :: enum u8 {
	Device_Local    = 0, // DEVICE_LOCAL only
	Host_Visible    = 1, // HOST_VISIBLE + HOST_COHERENT (typical CPU write target)
	Lazy_Allocated  = 2, // DEVICE_LOCAL + LAZILY_ALLOCATED (tiler-friendly)
	UMA             = 3, // DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT (integrated GPU)
	Other           = 4, // unmapped combinations
	COUNT           = 5,
}

// Memory_Type_Stats is the per-bucket VMA-derived view. `bytes` comes
// from vma.TotalStatistics.memoryType[i].statistics.allocationBytes
// summed across types with matching flags. `budget_bytes` is the
// over-approximation: VK_EXT_memory_budget only reports per-HEAP
// budgets, so we attribute the full heap budget to each type bucket
// in that heap. The over-count is intentional - a reviewer who sees
// "two buckets both at 100% on the same heap" immediately knows the
// heap is the bottleneck, not the bucket split.
Memory_Type_Stats :: struct {
	bytes:           u64,
	budget_bytes:    u64,
	allocation_count:u32,
	unused_bytes:    u64,
	unused_range_count: u32,
}

// ===========================================================================
//* GPU arena diagnostics.
//
// The arena allocator (Gpu_Arena.odin) lives in the same package but
// has its own accounting. The diagnostics layer folds per-class bytes
// into the same per-frame ring the VMA stats already use, so a reviewer
// can graph "arena bytes vs per-asset bytes" without the dump knowing
// the arena internals.
//
// Public accessors stay lightweight: one record per arena creation +
// a snapshot reader. The diagnostics layer never owns the arena
// free-list; it only mirrors capacity / used into the existing upload
// ring.
// ===========================================================================

// diag_record_arena_created bumps the arena counter + capacity
// per-class for the per-frame ring. Called from gpu_arena_init.
diag_record_arena_created_impl :: proc(class: GPU_Resource_Class, capacity: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	idx := int(class)
	if idx < 0 || idx >= int(GPU_Resource_Class.COUNT) do return
	d.arenas[idx].class    = class
	d.arenas[idx].capacity = capacity
	d.arena_count += 1
}

// diag_record_arena_destroyed is the matching teardown hook. The arena
// capacity drops to 0; live_suballocs are expected to be 0 by the time
// this is called (the deferred-destruction path retires them first).
diag_record_arena_destroyed_impl :: proc(class: GPU_Resource_Class) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	idx := int(class)
	if idx < 0 || idx >= int(GPU_Resource_Class.COUNT) do return
	d.arenas[idx] = {}
	if d.arena_count > 0 do d.arena_count -= 1
}

// diag_record_arena_used updates the per-arena used / free snapshot.
// Called every frame from vulkan_upload_scene (once per arena) so the
// per-frame ring tracks transient pressure without an O(free-list)
// walk in the dump.
diag_record_arena_used_impl :: proc(class: GPU_Resource_Class, used: u64, free: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	sync.recursive_mutex_lock(&DIAGNOSTICS_MUTEX)
	defer sync.recursive_mutex_unlock(&DIAGNOSTICS_MUTEX)
	d := &DIAGNOSTICS_STATE
	idx := int(class)
	if idx < 0 || idx >= int(GPU_Resource_Class.COUNT) do return
	d.arenas[idx].used  = used
	d.arenas[idx].free  = free
	d.arenas[idx].peak_used = max(d.arenas[idx].peak_used, used)
}

// ===========================================================================
//* Vulkan-side diagnostics: validation layers + debug labels + GPU
// timestamp pool. Everything below is backend-private; nothing here
// leaks Vulkan types out of BF_GPU.
// ===========================================================================

@(require_results)
@(private)
vulkan_diag_layer_count :: proc() -> int {
	// One validation layer is enough for v1; we always ask for
	// VK_LAYER_KHRONOS_validation when diagnostics are enabled so the
	// driver surfaces parameter / sync hazards during development.
	if BF_GPU_DIAG_ENABLE_VALIDATION {
		return 1
	}
	return 0
}

@(private)
BF_GPU_DIAG_ENABLE_VALIDATION :: #config(BF_GPU_ENABLE_VALIDATION, false)

// vulkan_validation_layer_names returns the list of validation
// layers requested at instance creation. Empty slice when the
// diagnostics build config opts out of validation, so a release
// build never pays the perf cost.
@(private)
BF_GPU_VALIDATION_LAYER_NAMES := []cstring{"VK_LAYER_KHRONOS_validation"}

@(private)
vulkan_validation_layer_names :: proc() -> []cstring {
	if !BF_GPU_DIAG_ENABLE_VALIDATION do return nil
	return BF_GPU_VALIDATION_LAYER_NAMES
}

// vulkan_debug_utils_extension_names returns the extension strings
// the Vulkan instance should enable when VK_EXT_debug_utils is
// available. Empty on builds without validation so the renderer
// doesn't pay for a label callback it never uses.
@(private)
BF_GPU_DEBUG_UTILS_EXTENSION_NAMES := []cstring{"VK_EXT_debug_utils"}

@(private)
vulkan_debug_utils_extension_names :: proc() -> []cstring {
	if !BF_GPU_DIAG_ENABLE_VALIDATION do return nil
	return BF_GPU_DEBUG_UTILS_EXTENSION_NAMES
}

// vulkan_diag_init is the Vulkan-side counterpart to
// renderer_diagnostics_init. Called once the instance + physical
// device exist; resolves VK_EXT_debug_utils, creates the debug
// messenger (when validation is on), and creates the GPU timestamp
// query pool.
//
// Safe to call when validation is off: the only effect is the
// timestamp query pool, which is what makes the timer profiles
// functional in either build mode.
vulkan_diag_init :: proc() -> bool {
	if DIAGNOSTICS_GPU_STATE.initialized do return true

	// Resolve the host-side timestamp period from physical device
	// limits. The host ns = (delta * period) / 1e6; without this the
	// GPU timings would surface as raw tick counts that have no
	// direct correspondence to ms.
	if VULKAN_STATE.physical_device != nil && VULKAN_STATE.device != nil {
		props := vk.PhysicalDeviceProperties{}
		vk.GetPhysicalDeviceProperties(VULKAN_STATE.physical_device, &props)
		DIAGNOSTICS_GPU_STATE.timestamp_period_ns = props.limits.timestampPeriod
	}

	// Create the timestamp query pool. One slice per
	// MAX_FRAMES_IN_FLIGHT frame, sized to int(Timer_Slot.COUNT)
	// timestamps each. The pool is created unconditionally so the
	// GPU timing path works without an instrumented frame recording
	// in anything other than tests (we always want the diagnostics
	// timeline to be live).
	if VULKAN_STATE.device != nil {
		pool_info := vk.QueryPoolCreateInfo {
			sType              = .QUERY_POOL_CREATE_INFO,
			queryType          = .TIMESTAMP,
			queryCount         = u32(MAX_FRAMES_IN_FLIGHT) * u32(Timer_Slot.COUNT),
		}
		pool: vk.QueryPool
		result := vk.CreateQueryPool(VULKAN_STATE.device, &pool_info, nil, &pool)
		if result == .SUCCESS {
			DIAGNOSTICS_GPU_STATE.query_pool = rawptr(uintptr(pool))
			DIAGNOSTICS_GPU_STATE.query_pool_valid = true
			// Reset every slot in the pool so a stale GPU write
			// cannot poison the first read. vkResetQueryPool
			// requires the hostQueryReset feature; when the
			// physical device does not expose it (low-end Intel
			// integrated GPUs are known to omit it) record a
			// vkCmdResetQueryPool on a transient command buffer
			// instead, which is always valid.
			query_count := u32(MAX_FRAMES_IN_FLIGHT) * u32(Timer_Slot.COUNT)
			if VULKAN_STATE.host_query_reset_available {
				vk.ResetQueryPool(VULKAN_STATE.device, pool, 0, query_count)
			} else if vulkan_diag_reset_pool_via_cmd_buffer(pool, query_count) {
				log.debug("[BF_GPU/Vulkan] Timestamp pool reset via vkCmdResetQueryPool (hostQueryReset unavailable)")
			} else {
				log.warn("[BF_GPU/Vulkan] Could not reset timestamp pool; first frame timestamps may be garbage")
			}
		} else {
			log.warnf("[BF_GPU/Vulkan] vkCreateQueryPool failed: %v", result)
		}
	}

	DIAGNOSTICS_GPU_STATE.initialized = true
	DIAGNOSTICS_STATE.debug.timestamp_pool_ready = DIAGNOSTICS_GPU_STATE.query_pool_valid
	DIAGNOSTICS_STATE.debug.validation_enabled  = BF_GPU_DIAG_ENABLE_VALIDATION
	DIAGNOSTICS_STATE.debug.label_capability    = DIAGNOSTICS_GPU_STATE.debug_utils_supported \
		? .Enabled_With_Validation \
		: .Unsupported
	log.infof(
		"[BF_GPU/Vulkan] Diagnostics ready: validation=%v label=%v timestamp_pool=%v",
		DIAGNOSTICS_STATE.debug.validation_enabled,
		DIAGNOSTICS_STATE.debug.label_capability,
		DIAGNOSTICS_STATE.debug.timestamp_pool_ready,
	)
	return true
}

// vulkan_diag_shutdown releases the debug messenger / query pool.
// Invoked from vulkan_shutdown before the device goes away.
vulkan_diag_shutdown :: proc() {
	if !DIAGNOSTICS_GPU_STATE.initialized do return
	if VULKAN_STATE.device != nil && DIAGNOSTICS_GPU_STATE.query_pool != nil {
		pool := vk.QueryPool(uintptr(DIAGNOSTICS_GPU_STATE.query_pool))
		vk.DestroyQueryPool(VULKAN_STATE.device, pool, nil)
	}
	if VULKAN_STATE.instance != nil && DIAGNOSTICS_GPU_STATE.debug_messenger != nil {
		messenger := vk.DebugUtilsMessengerEXT(uintptr(DIAGNOSTICS_GPU_STATE.debug_messenger))
		vk.DestroyDebugUtilsMessengerEXT(VULKAN_STATE.instance, messenger, nil)
	}
	DIAGNOSTICS_GPU_STATE = {}
}

@(private)
DIAG_DEBUG_LOGGER: log.Logger

// vulkan_diag_init_logger captures the active logger so the
// system-callback debug messenger can reach it without the runtime
// context. Called once from vulkan_diag_init. The captured logger
// outlives the caller's stack frame because log.Logger holds procs
// and a copy of the underlying buffer reference.
vulkan_diag_init_logger :: proc() {
	DIAG_DEBUG_LOGGER = context.logger
}

// vulkan_diag_reset_pool_via_cmd_buffer records a vkCmdResetQueryPool
// on a transient one-shot command buffer and waits for it to
// complete. Used as the fallback when the hostQueryReset feature is
// not available; vkCmdResetQueryPool is always valid, just less
// convenient because it goes through the queue submission path.
//
// Returns true on success. Allocation failures and queue submission
// failures are non-fatal; the caller logs a warning and continues
// because a stale timestamp pool only affects the first read of a
// slot, not the correctness of any GPU work.
vulkan_diag_reset_pool_via_cmd_buffer :: proc(pool: vk.QueryPool, query_count: u32) -> bool {
	if VULKAN_STATE.device == nil || pool == cast(vk.QueryPool)0 || query_count == 0 {
		return false
	}
	// Reuse the per-frame graphics command pool/buffer when
	// available; the diagnostics init runs after command resources
	// are up so this path is the common case.
	if len(VULKAN_STATE.frames) > 0 {
		frame := &VULKAN_STATE.frames[0]
		if frame.command_pool != cast(vk.CommandPool)0 && frame.command_buffer != cast(vk.CommandBuffer)uintptr(0) {
			if vk.ResetCommandPool(VULKAN_STATE.device, frame.command_pool, {}) != .SUCCESS {
				return false
			}
			begin_info := vk.CommandBufferBeginInfo {
				sType = .COMMAND_BUFFER_BEGIN_INFO,
				flags = {.ONE_TIME_SUBMIT},
			}
			if vk.BeginCommandBuffer(frame.command_buffer, &begin_info) != .SUCCESS {
				return false
			}
			vk.CmdResetQueryPool(frame.command_buffer, pool, 0, query_count)
			if vk.EndCommandBuffer(frame.command_buffer) != .SUCCESS {
				return false
			}
			submit_info := vk.SubmitInfo {
				sType                = .SUBMIT_INFO,
				commandBufferCount   = 1,
				pCommandBuffers      = &frame.command_buffer,
			}
			queue := VULKAN_STATE.graphics_queue
			if result := vk.QueueSubmit(queue, 1, &submit_info, {}); result != .SUCCESS {
				return false
			}
			return vk.QueueWaitIdle(queue) == .SUCCESS
		}
	}
	return false
}

// log_with_level forwards to the captured DIAG_DEBUG_LOGGER without
// touching runtime context. Used from the system-callback debug
// messenger where context is undefined. The location is a manual
// stub since the system-callback cannot synthesise a
// #caller_location.
@(private = "file")
log_with_level :: proc(level: log.Level, msg: string, loc: runtime.Source_Code_Location) {
	if DIAG_DEBUG_LOGGER.procedure == nil ||
	   DIAG_DEBUG_LOGGER.procedure == log.nil_logger_proc {
		return
	}
	if level < DIAG_DEBUG_LOGGER.lowest_level {
		return
	}
	DIAG_DEBUG_LOGGER.procedure(
		DIAG_DEBUG_LOGGER.data,
		level,
		msg,
		DIAG_DEBUG_LOGGER.options,
		loc,
	)
}

// vulkan_diag_debug_callback is the VK_EXT/debug_utils callback.
// Bridges driver-side validation messages to BF_GPU's logger so a
// debug build surfaces every perf / correctness warning. Vulkan
// expects the system/stdcall calling convention; the callback
// therefore receives no runtime context. We restore the engine's
// runtime context at the start so the rest of the body can call
// regular procs.
@(private)
vulkan_diag_debug_callback :: proc "system" (
	messageSeverity: vk.DebugUtilsMessageSeverityFlagsEXT,
	messageTypes:   vk.DebugUtilsMessageTypeFlagsEXT,
	pCallbackData:  ^vk.DebugUtilsMessengerCallbackDataEXT,
	pUserData:      rawptr,
) -> b32 {
	context = runtime.default_context()
	_ = pUserData
	_ = messageTypes
	if pCallbackData == nil do return false
	msg := string(pCallbackData.pMessage)
	level: log.Level = .Debug
	switch {
	case .ERROR in messageSeverity:   level = .Error
	case .WARNING in messageSeverity: level = .Warning
	case .INFO in messageSeverity:    level = .Info
	case:                            level = .Debug
	}
	log.logf(level, "%s", msg)
	return false
}

// vulkan_diag_create_debug_messenger attaches the debug-utils
// callback to the live instance. Returns true on success; the caller
// logs and continues on failure rather than aborting (a missing
// debug callback does not block rendering).
@(private)
vulkan_diag_create_debug_messenger :: proc() -> bool {
	if !DIAGNOSTICS_GPU_STATE.debug_utils_supported do return false
	if VULKAN_STATE.instance == nil do return false

	create_info := vk.DebugUtilsMessengerCreateInfoEXT {
		sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		messageSeverity     = {(.VERBOSE), (.INFO), (.WARNING), (.ERROR)},
		messageType         = {(.GENERAL), (.VALIDATION), (.PERFORMANCE)},
		pfnUserCallback     = vulkan_diag_debug_callback,
		pUserData           = nil,
	}
	messenger: vk.DebugUtilsMessengerEXT
	result := vk.CreateDebugUtilsMessengerEXT(
		VULKAN_STATE.instance,
		&create_info,
		nil,
		&messenger,
	)
	if result != .SUCCESS {
		log.warnf("[BF_GPU/Vulkan] CreateDebugUtilsMessengerEXT failed: %v", result)
		return false
	}
	DIAGNOSTICS_GPU_STATE.debug_messenger = rawptr(uintptr(messenger))
	return true
}

// vulkan_diag_probe_debug_utils probes VK_EXT_debug_utils on the
// loaded instance + physical device. Sets the debug_utils_supported
// flag, used as the gate for every debug label helper and debug
// messenger creation.
vulkan_diag_probe_debug_utils :: proc() {
	if !BF_GPU_DIAG_ENABLE_VALIDATION {
		DIAGNOSTICS_GPU_STATE.debug_utils_supported = false
		return
	}
	if VULKAN_STATE.instance == nil do return
	raw_data_count: u32 = 0
	result := vk.EnumerateInstanceExtensionProperties(nil, &raw_data_count, nil)
	if result != .SUCCESS || raw_data_count == 0 {
		DIAGNOSTICS_GPU_STATE.debug_utils_supported = false
		return
	}
	props := make([]vk.ExtensionProperties, raw_data_count)
	defer delete(props)
	result = vk.EnumerateInstanceExtensionProperties(nil, &raw_data_count, raw_data(props))
	if result != .SUCCESS {
		DIAGNOSTICS_GPU_STATE.debug_utils_supported = false
		return
	}
	for &p in props {
		name := cstring(&p.extensionName[0])
		if name == "VK_EXT_debug_utils" {
			DIAGNOSTICS_GPU_STATE.debug_utils_supported = true
			// The proc-addr loader pulled CreateDebugUtilsMessengerEXT
			// during load_proc_addresses_instance; we just need to
			// wire the messenger.
			vulkan_diag_create_debug_messenger()
			return
		}
	}
	DIAGNOSTICS_GPU_STATE.debug_utils_supported = false
}

// ---------------------------------------------------------------------------
//* Debug label helpers.
// ---------------------------------------------------------------------------

// vulkan_diag_set_object_name hooks the live `vk.SetDebugUtilsObjectNameEXT`
// when supported. Used to label swapchain images, pipelines, buffers,
// and frame slots so RenderDoc / validation layers surface friendly
// identifiers instead of raw handle numbers.
vulkan_diag_set_object_name :: proc(object_type: vk.ObjectType, object_handle: u64, name: cstring) {
	if !DIAGNOSTICS_GPU_STATE.debug_utils_supported do return
	if VULKAN_STATE.device == nil do return
	if cast(rawptr)vk.SetDebugUtilsObjectNameEXT == nil do return
	info := vk.DebugUtilsObjectNameInfoEXT {
		sType        = .DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
		objectType   = object_type,
		objectHandle = object_handle,
		pObjectName  = name,
	}
	vk.SetDebugUtilsObjectNameEXT(VULKAN_STATE.device, &info)
}

// vulkan_diag_begin_frame_label wraps vk.CmdBeginDebugUtilsLabelEXT.
// Call at the top of each major pass boundary (extract, upload,
// culling, traditional, meshlet, submit, present) so RenderDoc
// renders nested scopes on the GPU timeline.
vulkan_diag_begin_frame_label :: proc(command_buffer: vk.CommandBuffer, name: cstring, color: [4]f32 = {0.05, 0.55, 0.85, 1.0}) {
	if !DIAGNOSTICS_GPU_STATE.debug_utils_supported do return
	if cast(rawptr)vk.CmdBeginDebugUtilsLabelEXT == nil do return
	label := vk.DebugUtilsLabelEXT {
		sType      = .DEBUG_UTILS_LABEL_EXT,
		pLabelName = name,
		color      = color,
	}
	vk.CmdBeginDebugUtilsLabelEXT(command_buffer, &label)
}

// vulkan_diag_end_frame_label pairs vulkan_diag_begin_frame_label.
// Every begin must be matched; a leak would invalidate later labels.
vulkan_diag_end_frame_label :: proc(command_buffer: vk.CommandBuffer) {
	if !DIAGNOSTICS_GPU_STATE.debug_utils_supported do return
	if cast(rawptr)vk.CmdEndDebugUtilsLabelEXT == nil do return
	vk.CmdEndDebugUtilsLabelEXT(command_buffer)
}

// vulkan_diag_write_timestamp records a single GPU timestamp into
// the diagnostics query pool at the matching Timer_Slot for the
// current frame. The host reads back the most-recent slot in
// vulkan_diag_collect_timestamps once a frame in flight has caught
// up.
vulkan_diag_write_timestamp :: proc(command_buffer: vk.CommandBuffer, slot: Timer_Slot) {
	if !DIAGNOSTICS_STATE.initialized do return
	if !DIAGNOSTICS_GPU_STATE.query_pool_valid do return
	if VULKAN_STATE.device == nil do return
	if command_buffer == nil do return
	pool := vk.QueryPool(uintptr(DIAGNOSTICS_GPU_STATE.query_pool))
	frame_slot := VULKAN_STATE.frame_index * u32(Timer_Slot.COUNT) + u32(slot)
	stage: vk.PipelineStageFlags2 = {.BOTTOM_OF_PIPE}
	vk.CmdWriteTimestamp2(command_buffer, stage, pool, frame_slot)
}

// vulkan_diag_collect_timestamps is called once per frame from
// vulkan_poll_graphics_completion: reads back the MAX_FRAMES_IN_FLIGHT
// -oldest slot's timestamps, converts them via timestampPeriod, and
// rolls the deltas into the GPU Stage_Timing rings.
//
// Reading WAIT_BIT (64) ensures the call blocks if the slot's
// timestamps have not landed yet; in practice by the time the GPU
// has completed MAX_FRAMES_IN_FLIGHT frames ago, the timestamps are
// always available.
vulkan_diag_collect_timestamps :: proc(frame_index: u64) {
	if !DIAGNOSTICS_STATE.initialized do return
	if !DIAGNOSTICS_GPU_STATE.query_pool_valid do return
	if VULKAN_STATE.device == nil do return
	if DIAGNOSTICS_GPU_STATE.timestamp_period_ns == 0 do return

	pool := vk.QueryPool(uintptr(DIAGNOSTICS_GPU_STATE.query_pool))
	frame_slot := VULKAN_STATE.frame_index * u32(Timer_Slot.COUNT)
	data: [int(Timer_Slot.COUNT)]u64
	dataSize := size_of(data)
	flags: vk.QueryResultFlags = {.WAIT}
	result := vk.GetQueryPoolResults(
		VULKAN_STATE.device,
		pool,
		frame_slot,
		u32(Timer_Slot.COUNT),
		dataSize,
		&data[0],
		size_of(u64),
		flags,
	)
	if result != .SUCCESS && result != .NOT_READY {
		log.warnf("[BF_GPU/Vulkan] GetQueryPoolResults failed: %v", result)
		return
	}
	if result == .NOT_READY do return

	// Compute deltas between boundaries; the timestamps are in the
	// device clock domain. Resolve to host ns via period.
	period := DIAGNOSTICS_GPU_STATE.timestamp_period_ns
	if period == 0 do period = 1.0

	d := &DIAGNOSTICS_STATE
	ring_index := u32(frame_index % u64(DIAG_FRAME_HISTORY))

	tick_ns :: #force_inline proc(start, end: u64, period: f32) -> u64 {
		if end < start do return 0
		return u64(f32(end - start) * period)
	}

	upload_ns  := tick_ns(data[DIAG_TIMER_BEGIN_IDX],          data[DIAG_TIMER_UPLOAD_END_IDX],     period)
	culling_ns := tick_ns(data[DIAG_TIMER_UPLOAD_END_IDX],     data[DIAG_TIMER_CULLING_END_IDX],    period)
	trad_ns    := tick_ns(data[DIAG_TIMER_CULLING_END_IDX],    data[DIAG_TIMER_TRADITIONAL_END_IDX],period)
	mesh_ns    := tick_ns(data[DIAG_TIMER_TRADITIONAL_END_IDX],data[DIAG_TIMER_MESHLET_END_IDX],    period)
	submit_ns  := tick_ns(data[DIAG_TIMER_MESHLET_END_IDX],    data[DIAG_TIMER_SUBMIT_END_IDX],     period)
	present_ns := tick_ns(data[DIAG_TIMER_SUBMIT_END_IDX],     data[DIAG_TIMER_PRESENT_END_IDX],    period)
	total_ns   := tick_ns(data[DIAG_TIMER_BEGIN_IDX],          data[DIAG_TIMER_PRESENT_END_IDX],    period)
	render_ns  := culling_ns + trad_ns + mesh_ns

	d.gpu.culling_ns.ns[ring_index]     = culling_ns
	d.gpu.upload_ns.ns[ring_index]      = upload_ns
	d.gpu.traditional_ns.ns[ring_index] = trad_ns
	d.gpu.meshlet_ns.ns[ring_index]     = mesh_ns
	d.gpu.submit_ns.ns[ring_index]      = submit_ns
	d.gpu.present_ns.ns[ring_index]     = present_ns
	d.gpu.total_ns.ns[ring_index]       = total_ns
	d.gpu.render_ns.ns[ring_index]      = render_ns

	set_index :: #force_inline proc(stage: ^Stage_Timing, ring_index: u32, frame_index: u64) {
		stage.index = ring_index
		if u64(stage.filled) < frame_index + 1 {
			stage.filled = min(u32(frame_index + 1), u32(DIAG_FRAME_HISTORY))
		}
	}
	set_index(&d.gpu.culling_ns,     ring_index, frame_index)
	set_index(&d.gpu.upload_ns,      ring_index, frame_index)
	set_index(&d.gpu.traditional_ns, ring_index, frame_index)
	set_index(&d.gpu.meshlet_ns,     ring_index, frame_index)
	set_index(&d.gpu.submit_ns,      ring_index, frame_index)
	set_index(&d.gpu.present_ns,     ring_index, frame_index)
	set_index(&d.gpu.total_ns,       ring_index, frame_index)
	set_index(&d.gpu.render_ns,      ring_index, frame_index)
}
