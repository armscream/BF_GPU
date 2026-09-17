// BF_GPU/Memory.odin
//
// Host-side memory accounting for the renderer-facing scene storage.
//
// These counters measure what the renderer holds in CPU / mirror GPU buffers
// (the per-frame, per-instance state). They do NOT measure VMA-backed
// geometry, texture, material, or asset residency — those are reported by
// the GPU memory diagnostics in prompt 16.
//
// Every helper is O(1) (it sums per-pool sizes; no walk). The numbers are
// designed to be polled every frame so the editor / profilers can graph
// resident scene memory without paying an O(N) cost per poll.

package BF_GPU

import ECS "../BF_ECS"

// GPU_Pool_Bytes is a single-frame snapshot of the GPU_Scene mirror's
// resident byte count. All values include the Odin dynamic-array header
// overhead (the per-pool `len * stride` cost the user actually pays).
GPU_Pool_Bytes :: struct {
	transform_pool:  uint, // Gpu_Transform_Component
	model_pool:      uint, // Gpu_Model_Component
	culling_pool:    uint, // GPU_Culling_Instance
	camera_pool:     uint, // Gpu_Camera
	material_pool:   uint, // Gpu_Material
	material_lookup: uint, // u32 per (model, mesh)
	pipeline_lookup: uint, // u32 per (model, mesh)
	model_alloc:     uint, // Gpu_Model_Allocation
	mesh_alloc:      uint, // Gpu_Mesh_Allocation
	draw_descriptors:uint, // Gpu_Mesh_Draw_Descriptor
	model_addresses: uint, // Gpu_Model_Addresses
	indirect_commands:uint,
	static_chunks:   uint,
	chunk_instances: uint,
	sparse_maps:     uint,
}

// Total returns the sum of every field on the snapshot.
Total :: #force_inline proc(b: GPU_Pool_Bytes) -> uint {
	return b.transform_pool + b.model_pool + b.culling_pool +
	       b.camera_pool + b.material_pool +
	       b.material_lookup + b.pipeline_lookup +
	       b.model_alloc + b.mesh_alloc + b.draw_descriptors +
	       b.model_addresses + b.indirect_commands +
	       b.static_chunks + b.chunk_instances + b.sparse_maps;
}

// gpu_scene_byte_count returns the per-pool byte breakdown for a GPU_Scene.
// All counts include only the actively-populated prefix of each pool
// (`len(data) * size_of(T)`), not the reserved capacity. Capacity-aware
// counts are exposed separately when the editor needs to graph headroom.
gpu_scene_byte_count :: proc(gpu: ^GPU_Scene) -> GPU_Pool_Bytes {
	b: GPU_Pool_Bytes
	if !gpu_scene_is_valid(gpu) do return b
	b.transform_pool   = uint(len(gpu.transforms.data))   * uint(size_of(Gpu_Transform_Component))
	b.model_pool       = uint(len(gpu.models.data))       * uint(size_of(Gpu_Model_Component))
	b.culling_pool     = uint(len(gpu.culling))           * uint(size_of(GPU_Culling_Instance))
	b.camera_pool      = uint(len(gpu.cameras.data))      * uint(size_of(Gpu_Camera))
	b.material_pool    = uint(len(gpu.materials.data))    * uint(size_of(Gpu_Material))
	b.material_lookup  = uint(len(gpu.material_lookup.data)) * uint(size_of(u32))
	b.pipeline_lookup  = uint(len(gpu.pipeline_lookup.data)) * uint(size_of(u32))
	b.model_alloc      = uint(len(gpu.model_allocations.data)) * uint(size_of(Gpu_Model_Allocation))
	b.mesh_alloc       = uint(len(gpu.mesh_allocations.data))  * uint(size_of(Gpu_Mesh_Allocation))
	b.draw_descriptors = uint(len(gpu.draw_descriptors.data))  * uint(size_of(Gpu_Mesh_Draw_Descriptor))
	b.model_addresses  = uint(len(gpu.model_addresses.data))   * uint(size_of(Gpu_Model_Addresses))
	b.indirect_commands= uint(len(gpu.indirect_commands))      * uint(size_of(GPU_Indirect_Command))
	b.static_chunks    = uint(len(gpu.static_chunks))     * uint(size_of(Gpu_Static_Chunk))
	b.chunk_instances  = uint(len(gpu.chunk_instances))   * uint(size_of(u32))
	b.sparse_maps = uint(len(gpu.sparse.transforms.entity_to_dense)) * uint(size_of(u32))
		b.sparse_maps += uint(len(gpu.sparse.transforms.dense_to_entity)) * uint(size_of(u32))
		b.sparse_maps += uint(len(gpu.sparse.models.entity_to_dense))     * uint(size_of(u32))
		b.sparse_maps += uint(len(gpu.sparse.models.dense_to_entity))     * uint(size_of(u32))
		b.sparse_maps += uint(len(gpu.sparse.cameras.entity_to_dense))    * uint(size_of(u32))
		b.sparse_maps += uint(len(gpu.sparse.cameras.dense_to_entity))    * uint(size_of(u32))
	return b
}

// Render_Pool_Bytes is the same kind of snapshot for the CPU-side
// Render_Scene.
Render_Pool_Bytes :: struct {
	instances:      uint, // Render_Instance
	transforms:     uint, // Render_Transform
	spatial:        uint, // Render_Spatial_Metadata
	lights:         uint, // Render_Light
	cameras:        uint, // Render_Camera
	particles:      uint, // Render_Particle
	chunks:         uint, // Render_Chunk
	chunk_instances:uint, // u32 slot indices
	change_sets:    uint, // added + updated + removed buffers
	entity_map:     uint, // entity_to_instance
	free_slots:     uint, // free slot list
}

// Total returns the sum of every field on the snapshot.
@(require_results)
total_render_pool_bytes :: #force_inline proc(b: Render_Pool_Bytes) -> uint {
	return b.instances + b.transforms + b.spatial +
	       b.lights + b.cameras + b.particles + b.chunks +
	       b.chunk_instances + b.change_sets + b.entity_map + b.free_slots;
}

// render_scene_byte_count returns the per-pool byte breakdown for a
// Render_Scene. The change-set and map sizes are amortized by counting the
// underlying dynamic-array backing store, which is what actually costs RAM.
render_scene_byte_count :: proc(scene: ^Render_Scene) -> Render_Pool_Bytes {
	b: Render_Pool_Bytes
	if !render_scene_is_valid(scene) do return b
	b.instances        = uint(len(scene.instances))    * uint(size_of(Render_Instance))
	b.transforms       = uint(len(scene.transforms))   * uint(size_of(Render_Transform))
	b.spatial          = uint(len(scene.spatial))      * uint(size_of(Render_Spatial_Metadata))
	b.lights           = uint(len(scene.lights))       * uint(size_of(Render_Light))
	b.cameras          = uint(len(scene.cameras))      * uint(size_of(Render_Camera))
	b.particles        = uint(len(scene.particles))    * uint(size_of(Render_Particle))
	b.chunks           = uint(len(scene.chunks))       * uint(size_of(Render_Chunk))
	b.chunk_instances  = uint(len(scene.chunk_instances)) * uint(size_of(u32))
	b.change_sets      = (uint(len(scene.added))   + uint(len(scene.updated)) +
	                     uint(len(scene.removed))) * uint(size_of(u32))
	// The entity map cost is approximated as (entries * (Entity + id)) which
	// matches Odin's map layout closely enough for graphing purposes.
	b.entity_map       = uint(len(scene.entity_to_instance)) * uint(size_of(ECS.Entity) + size_of(Render_Instance_ID))
	b.free_slots       = uint(len(scene.free_slots))   * uint(size_of(u32))
	return b
}

// gpu_scene_per_instance_bytes returns the byte cost of the slot-parallel
// GPU pools combined (`transform_pool + model_pool + culling_pool`).
// Useful for sizing budgets: a scene with `n` live instances pays roughly
// `n * gpu_scene_per_instance_bytes` bytes for the mirror storage, plus
// overhead for the frame-global pools.
gpu_scene_per_instance_bytes :: #force_inline proc() -> uint {
	return uint(size_of(Gpu_Transform_Component)) +
	       uint(size_of(Gpu_Model_Component)) +
	       uint(size_of(GPU_Culling_Instance));
}

// render_scene_per_instance_bytes returns the byte cost of the
// slot-parallel Render_Scene arrays combined. Same shape as the GPU side.
render_scene_per_instance_bytes :: #force_inline proc() -> uint {
	return uint(size_of(Render_Instance)) +
	       uint(size_of(Render_Transform)) +
	       uint(size_of(Render_Spatial_Metadata));
}
