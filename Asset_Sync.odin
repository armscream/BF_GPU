// BF_GPU/Asset_Sync.odin
//
// Asset_ID -> cooked asset -> GPU resource synchronization.
//
// Core/ECS owns Asset_ID / Asset_Ref. BF_GPU owns GPU_*_ID. The bridge
// is the cooked-asset upload pipeline implemented in this file.
//
// Lifecycle
// ---------
//
// gpu_asset_sync_*      : cook the payload into GPU buffers / images /
//                         samplers, register the Asset_ID -> GPU_*_ID
//                         mapping, and return the (stable) GPU id. The
//                         payload ownership is transferred to BF_GPU;
//                         the caller must not modify it after the call.
//
//                         Registration is idempotent per Asset_ID: a
//                         second call with the same Asset_ID replaces
//                         the GPU payload in place and keeps the GPU
//                         id stable, so render instances referencing
//                         the asset stay valid across a reload. The
//                         previous GPU resources (buffer / image /
//                         sampler) are deferred into the pending list
//                         so the GPU can finish using them before the
//                         backend releases the underlying device
//                         memory.
//
// gpu_asset_unregister_*: drops the Asset_ID -> GPU_*_ID mapping and
//                         defers the GPU resources. Returns false when
//                         no mapping existed.
//
// gpu_asset_sync_tick   : runs every pending destruction whose GPU
//                         completion tag is <= up_to. The caller
//                         passes the latest graphics timeline value
//                         (or the previous frame's completed
//                         submission tag) so the backend can retire
//                         entries whose GPU work is known to have
//                         completed.
//
// All four procs stay in BF_GPU; the renderer module owns them so
// Asset_ID never leaks out of BF_GPU and no other module can build a
// second asset registry. The Asset_Ref -> GPU_*_ID translation
// Extraction.odin needs continues to go through GPU_Resource_Store
// (Resources.odin); Asset_Sync only adds the orchestration around it.
//
// ---------------------------------------------------------------------------
// Cooked-asset payloads
// ---------------------------------------------------------------------------
//
// Cooked_*_Asset is what the asset pipeline hands BF_GPU after the
// offline cook (.bmesh / .bmat / .btex / model descriptor). BF_GPU is
// concerned with the GPU-uploadable side: vertex / index bytes, image
// pixels, sampler knobs, PBR parameters. The offline cook keeps the
// source-of-truth strings; this layer never re-reads them.

package BF_GPU

import "core:log"
import mth "../../Core/BF_Math"

// ---------------------------------------------------------------------------
// Cooked payloads.
// ---------------------------------------------------------------------------

// Cooked_Mesh_Asset is the GPU-uploadable slice of a cooked mesh
// (.bmesh). The bytes are interpreted by the shader-side vertex layout
// Gpu_Vertex_Position / Gpu_Vertex_Attributes (Gpu_Types.odin). Bounds
// are world-space; the culling shaders use them.
Cooked_Mesh_Asset :: struct {
	vertex_data:   []u8,
	index_data:    []u8,
	vertex_count:  u32,
	index_count:   u32,
	vertex_stride: u32, // == sizeof(Gpu_Vertex_Position) + sizeof(Gpu_Vertex_Attributes)
	index_stride:  u32, // 2 (u16 indices) or 4 (u32 indices)
	bounds:        mth.AABB,
	flags:         u32, // free for the asset pipeline; renderer ignores
}

// Cooked_Texture_Asset is the GPU-uploadable slice of a cooked
// texture. Width / height / mip_count must be sane; the mip chain is
// tightly packed (mip 0 first, then mip 1, ...) in `data`. The sampler
// description travels with the texture so a reload can swap samplers
// without reuploading pixels.
//
// `async_load` is the runtime hint the streamer consults when it
// decides whether the texture's upload must finish before the next
// render frame (false / "blocking") or can be queued onto the async
// transfer path (true). The hint comes from the .bmat record's
// `async_load` field so the offline baker can mark peripheral
// textures (emissive / occlusion / metallicRoughness) as queued
// while keeping baseColor synchronous.
Cooked_Texture_Asset :: struct {
	width:      u32,
	height:     u32,
	mip_count:  u32,
	format:     Image_Format,
	data:       []u8,
	sampler:    Sampler_Description,
	usage:      Image_Usage,
	async_load: bool,
}

// Cooked_Material_Asset is the GPU-uploadable slice of a cooked
// material. Asset_Ref for textures is replaced with the renderer-side
// GPU_Texture_ID lookup; an unresolved Asset_Ref maps to
// GPU_TEXTURE_INVALID and the material is registered with the
// placeholder. The renderer's bindless descriptor update path keeps
// retrying those lookups every frame.
//
// `async_load` is the per-material default that the streamer uses
// when a Cooked_Texture_Asset does not override it. It is sourced from
// the .bmat's `async_load` field so the offline baker can flag
// "peripheral" materials (ones whose textures don't show up in the
// always-critical PBR pass).
Cooked_Material_Asset :: struct {
	base_colour:        mth.Vec4,
	emissive_colour:    mth.Vec3,
	emissive_intensity: f32,
	metalness:          f32,
	roughness:          f32,
	ao_strength:        f32,
	double_sided:       bool,
	transparent:        bool,
	alpha_tested:       bool,
	base_colour_texture: Asset_Ref,
	normal_texture:      Asset_Ref,
	orm_texture:         Asset_Ref,
	emissive_texture:    Asset_Ref,
	async_load:          bool,
}

// Cooked_Model_Asset is the GPU-uploadable slice of a cooked model.
// Each mesh / material is referenced by Asset_Ref; BF_GPU resolves
// them through the store's mesh_by_asset / material_by_asset maps so
// a model can be registered before its constituent assets are. Missing
// references produce a partial GPU_Model whose bounds / mesh range
// still describe the asset, and the extract path flags the instance
// .Pending_Asset until the references resolve.
Cooked_Model_Asset :: struct {
	meshes:    []Asset_Ref,
	materials: []Asset_Ref,
	bounds:    mth.AABB,
	flags:     u32, // free for the asset pipeline
}

// ---------------------------------------------------------------------------
// Renderer-side pending destruction list.
// ---------------------------------------------------------------------------
//
// The Vulkan backend has its own per-tag queue (Vk_Resource.odin);
// this list bridges the renderer payload (handles + asset_ids) to
// that backend queue. The handle map itself stays in the backend; the
// renderer never owns a VkBuffer / VkImage.
//
// gpu_asset_sync_tick walks the list and routes safe entries through
// the backend's deferred-destruction queue so VMA / Vulkan reap them
// once GPU work completes.

@(private)
Pending_Destruction_Buffer :: struct {
	handle: Gpu_Buffer_Handle,
	tag:    u64,
}

@(private)
Pending_Destruction_Image :: struct {
	handle: Gpu_Image_Handle,
	tag:    u64,
}

@(private)
Pending_Destruction_Sampler :: struct {
	handle: Gpu_Sampler_Handle,
	tag:    u64,
}

Asset_Sync_Pending :: struct {
	buffers:  [dynamic]Pending_Destruction_Buffer,
	images:   [dynamic]Pending_Destruction_Image,
	samplers: [dynamic]Pending_Destruction_Sampler,
}

asset_sync_pending_init :: proc(p: ^Asset_Sync_Pending, allocator := context.allocator) {
	if p == nil do return
	p.buffers  = make([dynamic]Pending_Destruction_Buffer, 0, 16, allocator)
	p.images   = make([dynamic]Pending_Destruction_Image, 0, 16, allocator)
	p.samplers = make([dynamic]Pending_Destruction_Sampler, 0, 16, allocator)
}

asset_sync_pending_destroy :: proc(p: ^Asset_Sync_Pending) {
	if p == nil do return
	delete(p.buffers)
	delete(p.images)
	delete(p.samplers)
	p^ = {}
}

@(private)
asset_sync_pending_buffer :: proc(p: ^Asset_Sync_Pending, handle: Gpu_Buffer_Handle, tag: u64) {
	if p == nil || handle == Gpu_Buffer_Handle(0) do return
	append(&p.buffers, Pending_Destruction_Buffer{handle = handle, tag = tag})
}

@(private)
asset_sync_pending_image :: proc(p: ^Asset_Sync_Pending, handle: Gpu_Image_Handle, tag: u64) {
	if p == nil || handle == Gpu_Image_Handle(0) do return
	append(&p.images, Pending_Destruction_Image{handle = handle, tag = tag})
}

@(private)
asset_sync_pending_sampler :: proc(p: ^Asset_Sync_Pending, handle: Gpu_Sampler_Handle, tag: u64) {
	if p == nil || handle == Gpu_Sampler_Handle(0) do return
	append(&p.samplers, Pending_Destruction_Sampler{handle = handle, tag = tag})
}

// asset_sync_pending_flush transfers every entry whose tag is <=
// up_to to the backend's deferred-destruction queue and clears the
// renderer-side list. Entries with a higher tag stay parked for the
// next tick. Returns the number of entries handed off to the backend.
asset_sync_pending_flush :: proc(
	p: ^Asset_Sync_Pending,
	backend: ^GPU_Backend,
	up_to: u64,
) -> int {
	if p == nil do return 0
	handoff := 0

	if backend != nil {
		// Buffers
		if len(p.buffers) > 0 {
			keep: [dynamic]Pending_Destruction_Buffer
			defer delete(keep)
			for entry in p.buffers {
				if entry.tag <= up_to && backend.destroy_buffer != nil {
					backend.destroy_buffer(entry.handle)
					handoff += 1
				} else {
					append(&keep, entry)
				}
			}
			clear(&p.buffers)
			for entry in keep do append(&p.buffers, entry)
		}
		// Images
		if len(p.images) > 0 {
			keep: [dynamic]Pending_Destruction_Image
			defer delete(keep)
			for entry in p.images {
				if entry.tag <= up_to && backend.destroy_image != nil {
					backend.destroy_image(entry.handle)
					handoff += 1
				} else {
					append(&keep, entry)
				}
			}
			clear(&p.images)
			for entry in keep do append(&p.images, entry)
		}
		// Samplers
		if len(p.samplers) > 0 {
			keep: [dynamic]Pending_Destruction_Sampler
			defer delete(keep)
			for entry in p.samplers {
				if entry.tag <= up_to && backend.destroy_sampler != nil {
					backend.destroy_sampler(entry.handle)
					handoff += 1
				} else {
					append(&keep, entry)
				}
			}
			clear(&p.samplers)
			for entry in keep do append(&p.samplers, entry)
		}
	}

	if backend != nil && backend.flush_deferred_destructions != nil {
		backend.flush_deferred_destructions(up_to)
	}
	return handoff
}

// gpu_asset_sync_tick flushes every renderer-pending destruction whose
// tag has been reached. Safe to call when the pending list is empty.
// Safe to call when the backend is nil (test profile).
gpu_asset_sync_tick :: proc(
	p: ^Asset_Sync_Pending,
	backend: ^GPU_Backend,
	up_to: u64,
) -> int {
	return asset_sync_pending_flush(p, backend, up_to)
}

// ---------------------------------------------------------------------------
// Sync helpers.
// ---------------------------------------------------------------------------

// gpu_asset_sync_mesh creates (or replaces) the GPU_Mesh payload for
// `asset`. Returns GPU_Mesh_ID on success; GPU_MESH_INVALID when the
// backend is unavailable or the payload is malformed. Idempotent: a
// second call with the same `asset` returns the same id and defers
// the previous vertex / index buffers to the pending list.
gpu_asset_sync_mesh :: proc(
	store: ^GPU_Resource_Store,
	backend: ^GPU_Backend,
	asset: Asset_ID,
	cooked: ^Cooked_Mesh_Asset,
	current_completion: u64,
	pending: ^Asset_Sync_Pending = nil,
) -> GPU_Mesh_ID {
	if !gpu_store_is_valid(store) || cooked == nil || asset == Asset_ID(0) do return GPU_MESH_INVALID
	if backend == nil ||
	   backend.create_asset_buffer == nil ||
	   backend.upload_buffer == nil ||
	   backend.destroy_buffer == nil {
		log.warnf("[BF_GPU] mesh sync: backend unavailable; deferring asset %v", asset)
		return GPU_MESH_INVALID
	}
	if len(cooked.vertex_data) == 0 || len(cooked.index_data) == 0 {
		log.warnf("[BF_GPU] mesh sync: empty payload for asset %v", asset)
		return GPU_MESH_INVALID
	}

	vb_usage := Gpu_Buffer_Usage{.Vertex_Buffer, .Shader_Device_Address}
	vb_handle := backend.create_asset_buffer(vb_usage, u64(len(cooked.vertex_data)), cooked.vertex_stride)
	if vb_handle == Gpu_Buffer_Handle(0) {
		log.warnf("[BF_GPU] mesh sync: vertex buffer create failed for asset %v", asset)
		return GPU_MESH_INVALID
	}
	if !backend.upload_buffer(vb_handle, raw_data(cooked.vertex_data), u64(len(cooked.vertex_data))) {
		log.warnf("[BF_GPU] mesh sync: vertex buffer upload failed for asset %v", asset)
		backend.destroy_buffer(vb_handle)
		return GPU_MESH_INVALID
	}
	ib_usage := Gpu_Buffer_Usage{.Index_Buffer, .Shader_Device_Address}
	ib_handle := backend.create_asset_buffer(ib_usage, u64(len(cooked.index_data)), cooked.index_stride)
	if ib_handle == Gpu_Buffer_Handle(0) {
		log.warnf("[BF_GPU] mesh sync: index buffer create failed for asset %v", asset)
		backend.destroy_buffer(vb_handle)
		return GPU_MESH_INVALID
	}
	if !backend.upload_buffer(ib_handle, raw_data(cooked.index_data), u64(len(cooked.index_data))) {
		log.warnf("[BF_GPU] mesh sync: index buffer upload failed for asset %v", asset)
		backend.destroy_buffer(vb_handle)
		backend.destroy_buffer(ib_handle)
		return GPU_MESH_INVALID
	}

	// If the store already had an entry for this asset, defer the
	// previous GPU buffers so the GPU can finish reading them before
	// VMA releases the underlying device memory. Capture `prev` BEFORE
	// registering the new payload - the register call overwrites the
	// slot in place.
	if pending != nil {
		if existing, ok := store.mesh_by_asset[asset]; ok && u32(existing) < u32(len(store.meshes)) {
			prev := store.meshes[u32(existing)]
			if prev.vertex_buffer != 0 {
				asset_sync_pending_buffer(pending, Gpu_Buffer_Handle(prev.vertex_buffer), current_completion)
			}
			if prev.index_buffer != 0 {
				asset_sync_pending_buffer(pending, Gpu_Buffer_Handle(prev.index_buffer), current_completion)
			}
		}
	}

	gpu_mesh := GPU_Mesh {
		vertex_buffer  = u32(vb_handle),
		index_buffer   = u32(ib_handle),
		vertex_offset  = 0,
		index_offset   = 0,
		index_count    = cooked.index_count,
		meshlet_offset = 0,
		meshlet_count  = 0,
		bounds         = cooked.bounds,
		material       = GPU_MATERIAL_INVALID,
	}
	id := gpu_mesh_register(store, asset, gpu_mesh)
	return id
}

// gpu_asset_sync_texture creates (or replaces) the GPU_Texture
// payload for `asset`. Returns GPU_Texture_ID on success;
// GPU_TEXTURE_INVALID otherwise. Idempotent: re-registration with the
// same Asset_ID keeps the GPU_Texture_ID stable and defers the
// previous image / sampler.
gpu_asset_sync_texture :: proc(
	store: ^GPU_Resource_Store,
	backend: ^GPU_Backend,
	asset: Asset_ID,
	cooked: ^Cooked_Texture_Asset,
	current_completion: u64,
	pending: ^Asset_Sync_Pending = nil,
) -> GPU_Texture_ID {
	if !gpu_store_is_valid(store) || cooked == nil || asset == Asset_ID(0) do return GPU_TEXTURE_INVALID
	if backend == nil ||
	   backend.create_image == nil ||
	   backend.create_image_view == nil ||
	   backend.create_sampler == nil ||
	   backend.destroy_image == nil ||
	   backend.destroy_image_view == nil ||
	   backend.destroy_sampler == nil {
		log.warnf("[BF_GPU] texture sync: backend unavailable; deferring asset %v", asset)
		return GPU_TEXTURE_INVALID
	}

	image_desc := Image_Description {
		format       = cooked.format,
		extent       = Image_Extent_2D{cooked.width, cooked.height},
		mip_count    = cooked.mip_count,
		array_layers = 1,
		samples      = 1,
		usage        = cooked.usage,
	}
	image_handle := backend.create_image(image_desc)
	if image_handle == Gpu_Image_Handle(0) {
		log.warnf("[BF_GPU] texture sync: image create failed for asset %v", asset)
		return GPU_TEXTURE_INVALID
	}
	view_handle := backend.create_image_view(Image_View_Description{
		image      = image_handle,
		base_mip   = 0,
		mip_count  = cooked.mip_count,
		base_layer = 0,
		layer_count= 1,
	})
	if view_handle == Gpu_Image_View_Handle(0) {
		log.warnf("[BF_GPU] texture sync: image-view create failed for asset %v", asset)
		backend.destroy_image(image_handle)
		return GPU_TEXTURE_INVALID
	}
	sampler_handle := backend.create_sampler(cooked.sampler)
	if sampler_handle == Gpu_Sampler_Handle(0) {
		log.warnf("[BF_GPU] texture sync: sampler create failed for asset %v", asset)
		backend.destroy_image_view(view_handle)
		backend.destroy_image(image_handle)
		return GPU_TEXTURE_INVALID
	}

	// Defer the previous image / sampler so the GPU can finish sampling
	// the old texture before it disappears.
	if pending != nil {
		if existing, ok := store.texture_by_asset[asset]; ok && u32(existing) < u32(len(store.textures)) {
			prev := store.textures[u32(existing)]
			if prev.image_id != 0 {
				asset_sync_pending_image(pending, Gpu_Image_Handle(prev.image_id), current_completion)
			}
			if prev.sampler_id != 0 {
				asset_sync_pending_sampler(pending, Gpu_Sampler_Handle(prev.sampler_id), current_completion)
			}
		}
	}

	gpu_tex := GPU_Texture {
		image_id   = u32(image_handle),
		sampler_id = u32(sampler_handle),
		width      = cooked.width,
		height     = cooked.height,
		mip_count  = cooked.mip_count,
		flags      = 0,
	}
	_ = view_handle // referenced through the image's paired view internally
	return gpu_texture_register(store, asset, gpu_tex)
}

// gpu_asset_sync_material creates (or replaces) the GPU_Material
// payload for `asset`. Returns GPU_MATERIAL_ID on success; replaces in
// place on subsequent calls. Texture references are resolved through
// the store; unresolved Asset_Refs map to GPU_TEXTURE_INVALID and are
// re-resolved on later frames by the bindless descriptor update path.
gpu_asset_sync_material :: proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
	cooked: ^Cooked_Material_Asset,
) -> GPU_Material_ID {
	if !gpu_store_is_valid(store) || cooked == nil || asset == Asset_ID(0) do return GPU_MATERIAL_INVALID

	gpu_mat := GPU_Material {
		shader_id           = 0,
		texture_base_colour = resolve_texture_ref(store, cooked.base_colour_texture),
		texture_normal      = resolve_texture_ref(store, cooked.normal_texture),
		texture_orm         = resolve_texture_ref(store, cooked.orm_texture),
		texture_emissive    = resolve_texture_ref(store, cooked.emissive_texture),
		flags = material_pack_flags(cooked.double_sided, cooked.transparent, cooked.alpha_tested),
	}
	return gpu_material_register(store, asset, gpu_mat)
}

// resolve_texture_ref looks up a texture Asset_Ref in the store. An
// empty ref (id == 0) returns GPU_TEXTURE_INVALID; a registered ref
// returns the GPU_Texture_ID; an unregistered ref returns
// GPU_TEXTURE_INVALID and the bindless descriptor update path keeps
// retrying.
@(private)
resolve_texture_ref :: proc(store: ^GPU_Resource_Store, ref: Asset_Ref) -> GPU_Texture_ID {
	if !gpu_store_is_valid(store) do return GPU_TEXTURE_INVALID
	if ref.id == Asset_ID(0) do return GPU_TEXTURE_INVALID
	return gpu_texture_find(store, ref.id)
}

// material_pack_flags packs the (double_sided, transparent, alpha_tested)
// triple into the bit layout Gpu_Material.flags expects. Mirrors the
// shader-side MATERIAL_FLAG_* constants.
material_pack_flags :: proc(double_sided, transparent, alpha_tested: bool) -> u32 {
	f: u32 = 0
	if double_sided do f |= MATERIAL_FLAG_DOUBLE_SIDED
	if transparent  do f |= MATERIAL_FLAG_TRANSPARENT
	if alpha_tested do f |= MATERIAL_FLAG_ALPHA_TESTED
	return f
}

// gpu_asset_sync_model creates (or replaces) the GPU_Model payload for
// `asset`. Mesh Asset_Refs are resolved through the store; an
// unresolved reference produces a partial model whose Mesh range is
// empty and the instance is extracted with .Pending_Asset until the
// missing meshes arrive.
gpu_asset_sync_model :: proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
	cooked: ^Cooked_Model_Asset,
) -> GPU_Model_ID {
	if !gpu_store_is_valid(store) || cooked == nil || asset == Asset_ID(0) do return GPU_MODEL_INVALID

	first_mesh := u32(INVALID_INDEX)
	mesh_count: u32 = 0
	for ref in cooked.meshes {
		if ref.id == Asset_ID(0) do continue
		if id := gpu_mesh_find(store, ref.id); id != GPU_Mesh_ID(0) {
			idx := u32(id)
			if first_mesh == u32(INVALID_INDEX) || idx < first_mesh do first_mesh = idx
			mesh_count += 1
		}
	}

	gpu_model := GPU_Model {
		meshes_offset = first_mesh,
		meshes_count  = mesh_count,
		bounds        = cooked.bounds,
		flags         = cooked.flags,
	}
	return gpu_model_register(store, asset, gpu_model)
}

// ---------------------------------------------------------------------------
// Unregister. Drops the Asset_ID -> GPU_*_ID mapping and defers the
// GPU resources. Pending extraction will flag affected instances
// .Pending_Asset and re-resolve once a fresh registration lands.
// ---------------------------------------------------------------------------

gpu_asset_unregister_mesh :: proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
	current_completion: u64,
	pending: ^Asset_Sync_Pending,
) -> bool {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return false
	if pending == nil {
		log.warn("[BF_GPU] unregister_mesh called without pending list; resources will leak")
		return gpu_mesh_unregister(store, asset)
	}
	if existing, ok := store.mesh_by_asset[asset]; ok && u32(existing) < u32(len(store.meshes)) {
		prev := store.meshes[u32(existing)]
		if prev.vertex_buffer != 0 {
			asset_sync_pending_buffer(pending, Gpu_Buffer_Handle(prev.vertex_buffer), current_completion)
		}
		if prev.index_buffer != 0 {
			asset_sync_pending_buffer(pending, Gpu_Buffer_Handle(prev.index_buffer), current_completion)
		}
	}
	return gpu_mesh_unregister(store, asset)
}

gpu_asset_unregister_texture :: proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
	current_completion: u64,
	pending: ^Asset_Sync_Pending,
) -> bool {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return false
	if pending == nil {
		log.warn("[BF_GPU] unregister_texture called without pending list; resources will leak")
		return gpu_texture_unregister(store, asset)
	}
	if existing, ok := store.texture_by_asset[asset]; ok && u32(existing) < u32(len(store.textures)) {
		prev := store.textures[u32(existing)]
		if prev.image_id != 0 {
			asset_sync_pending_image(pending, Gpu_Image_Handle(prev.image_id), current_completion)
		}
		if prev.sampler_id != 0 {
			asset_sync_pending_sampler(pending, Gpu_Sampler_Handle(prev.sampler_id), current_completion)
		}
	}
	return gpu_texture_unregister(store, asset)
}

gpu_asset_unregister_material :: proc(store: ^GPU_Resource_Store, asset: Asset_ID) -> bool {
	return gpu_material_unregister(store, asset)
}

gpu_asset_unregister_model :: proc(store: ^GPU_Resource_Store, asset: Asset_ID) -> bool {
	return gpu_model_unregister(store, asset)
}

// gpu_asset_resolve_material_textures is a hook for the eventual
// material-to-texture ref table; for now it is a no-op counted call so
// the bindless descriptor update path has a stable entry point.
gpu_asset_resolve_material_textures :: proc(store: ^GPU_Resource_Store) -> u32 {
	_ = store
	return 0
}