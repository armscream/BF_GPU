// BF_GPU/Resources.odin
//
// Asset_ID -> GPU resource identity.
//
// Core/ECS owns asset identity (Asset_ID / Asset_Ref). BF_GPU owns GPU
// resource identity (GPU_Model_ID / GPU_Mesh_ID / GPU_Material_ID /
// GPU_Texture_ID). This file is the only place the two are related, and it is
// the lookup the renderer extraction path uses to resolve a Render_Model's
// Asset_Ref into the GPU model the culling pipeline can draw.
//
// Registration is idempotent per asset: registering an asset that already has
// a GPU id updates the payload in place and keeps the id stable, so instances
// that already reference it stay valid across a reload. Unregistering drops
// the mapping; extraction re-resolves and flags the affected instances as
// Pending_Asset until a new upload lands.

package BF_GPU

//* LIFECYCLE
gpu_store_init :: proc(store: ^GPU_Resource_Store, allocator := context.allocator) -> bool {
	if store == nil do return false
	if store.initialized do return true

	store.allocator = allocator
	store.models = make([dynamic]GPU_Model, 1, 64, allocator)
	store.meshes = make([dynamic]GPU_Mesh, 1, 256, allocator)
	store.materials = make([dynamic]GPU_Material, 1, 64, allocator)
	store.textures = make([dynamic]GPU_Texture, 1, 128, allocator)
	store.model_by_asset = make(map[Asset_ID]GPU_Model_ID, allocator)
	store.mesh_by_asset = make(map[Asset_ID]GPU_Mesh_ID, allocator)
	store.material_by_asset = make(map[Asset_ID]GPU_Material_ID, allocator)
	store.texture_by_asset = make(map[Asset_ID]GPU_Texture_ID, allocator)
	store.revision = 1
	store.initialized = true
	return true
}

gpu_store_destroy :: proc(store: ^GPU_Resource_Store) {
	if store == nil || !store.initialized do return
	delete(store.models)
	delete(store.meshes)
	delete(store.materials)
	delete(store.textures)
	delete(store.model_by_asset)
	delete(store.mesh_by_asset)
	delete(store.material_by_asset)
	delete(store.texture_by_asset)
	store^ = {}
}

gpu_store_is_valid :: #force_inline proc(store: ^GPU_Resource_Store) -> bool {
	return store != nil && store.initialized
}

//* MODEL
gpu_model_find :: #force_inline proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
) -> GPU_Model_ID {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return GPU_MODEL_INVALID
	if id, ok := store.model_by_asset[asset]; ok do return id
	return GPU_MODEL_INVALID
}

gpu_model_register :: proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
	model: GPU_Model,
) -> GPU_Model_ID {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return GPU_MODEL_INVALID
	if existing, ok := store.model_by_asset[asset]; ok {
		store.models[u32(existing)] = model
		store.revision += 1
		return existing
	}
	append(&store.models, model)
	id := GPU_Model_ID(len(store.models) - 1)
	store.model_by_asset[asset] = id
	store.revision += 1
	return id
}

gpu_model_unregister :: proc(store: ^GPU_Resource_Store, asset: Asset_ID) -> bool {
	if !gpu_store_is_valid(store) do return false
	id, ok := store.model_by_asset[asset]
	if !ok do return false
	store.models[u32(id)] = {}
	delete_key(&store.model_by_asset, asset)
	store.revision += 1
	return true
}

gpu_model_get :: #force_inline proc(
	store: ^GPU_Resource_Store,
	id: GPU_Model_ID,
) -> ^GPU_Model {
	if !gpu_store_is_valid(store) || id == GPU_MODEL_INVALID do return nil
	if int(id) >= len(store.models) do return nil
	return &store.models[u32(id)]
}

//* MESH
gpu_mesh_find :: #force_inline proc(store: ^GPU_Resource_Store, asset: Asset_ID) -> GPU_Mesh_ID {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return GPU_MESH_INVALID
	if id, ok := store.mesh_by_asset[asset]; ok do return id
	return GPU_MESH_INVALID
}

gpu_mesh_register :: proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
	mesh: GPU_Mesh,
) -> GPU_Mesh_ID {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return GPU_MESH_INVALID
	if existing, ok := store.mesh_by_asset[asset]; ok {
		store.meshes[u32(existing)] = mesh
		store.revision += 1
		return existing
	}
	append(&store.meshes, mesh)
	id := GPU_Mesh_ID(len(store.meshes) - 1)
	store.mesh_by_asset[asset] = id
	store.revision += 1
	return id
}

gpu_mesh_unregister :: proc(store: ^GPU_Resource_Store, asset: Asset_ID) -> bool {
	if !gpu_store_is_valid(store) do return false
	id, ok := store.mesh_by_asset[asset]
	if !ok do return false
	store.meshes[u32(id)] = {}
	delete_key(&store.mesh_by_asset, asset)
	store.revision += 1
	return true
}

gpu_mesh_get :: #force_inline proc(store: ^GPU_Resource_Store, id: GPU_Mesh_ID) -> ^GPU_Mesh {
	if !gpu_store_is_valid(store) || id == GPU_MESH_INVALID do return nil
	if int(id) >= len(store.meshes) do return nil
	return &store.meshes[u32(id)]
}

//* MATERIAL
gpu_material_find :: #force_inline proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
) -> GPU_Material_ID {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return GPU_MATERIAL_INVALID
	if id, ok := store.material_by_asset[asset]; ok do return id
	return GPU_MATERIAL_INVALID
}

gpu_material_register :: proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
	material: GPU_Material,
) -> GPU_Material_ID {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return GPU_MATERIAL_INVALID
	if existing, ok := store.material_by_asset[asset]; ok {
		store.materials[u32(existing)] = material
		store.revision += 1
		return existing
	}
	append(&store.materials, material)
	id := GPU_Material_ID(len(store.materials) - 1)
	store.material_by_asset[asset] = id
	store.revision += 1
	return id
}

gpu_material_unregister :: proc(store: ^GPU_Resource_Store, asset: Asset_ID) -> bool {
	if !gpu_store_is_valid(store) do return false
	id, ok := store.material_by_asset[asset]
	if !ok do return false
	store.materials[u32(id)] = {}
	delete_key(&store.material_by_asset, asset)
	store.revision += 1
	return true
}

gpu_material_get :: #force_inline proc(
	store: ^GPU_Resource_Store,
	id: GPU_Material_ID,
) -> ^GPU_Material {
	if !gpu_store_is_valid(store) || id == GPU_MATERIAL_INVALID do return nil
	if int(id) >= len(store.materials) do return nil
	return &store.materials[u32(id)]
}

//* TEXTURE
gpu_texture_find :: #force_inline proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
) -> GPU_Texture_ID {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return GPU_TEXTURE_INVALID
	if id, ok := store.texture_by_asset[asset]; ok do return id
	return GPU_TEXTURE_INVALID
}

gpu_texture_register :: proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
	texture: GPU_Texture,
) -> GPU_Texture_ID {
	if !gpu_store_is_valid(store) || asset == Asset_ID(0) do return GPU_TEXTURE_INVALID
	if existing, ok := store.texture_by_asset[asset]; ok {
		store.textures[u32(existing)] = texture
		store.revision += 1
		return existing
	}
	append(&store.textures, texture)
	id := GPU_Texture_ID(len(store.textures) - 1)
	store.texture_by_asset[asset] = id
	store.revision += 1
	return id
}

gpu_texture_unregister :: proc(store: ^GPU_Resource_Store, asset: Asset_ID) -> bool {
	if !gpu_store_is_valid(store) do return false
	id, ok := store.texture_by_asset[asset]
	if !ok do return false
	store.textures[u32(id)] = {}
	delete_key(&store.texture_by_asset, asset)
	store.revision += 1
	return true
}

gpu_texture_get :: #force_inline proc(
	store: ^GPU_Resource_Store,
	id: GPU_Texture_ID,
) -> ^GPU_Texture {
	if !gpu_store_is_valid(store) || id == GPU_TEXTURE_INVALID do return nil
	if int(id) >= len(store.textures) do return nil
	return &store.textures[u32(id)]
}

//* MATERIAL RESOLUTION
// The material a Render_Instance draws with: the override when one is bound
// and resolved, otherwise the first material of the model's mesh range. The
// renderer classifies buckets from this material's flags.
gpu_instance_material :: proc(
	store: ^GPU_Resource_Store,
	instance: Render_Instance,
) -> GPU_Material_ID {
	if !gpu_store_is_valid(store) do return GPU_MATERIAL_INVALID
	if instance.gpu_material_override != GPU_MATERIAL_INVALID {
		return instance.gpu_material_override
	}
	model := gpu_model_get(store, instance.gpu_model)
	if model == nil || model.meshes_count == 0 do return GPU_MATERIAL_INVALID
	mesh_index := model.meshes_offset
	if int(mesh_index) >= len(store.meshes) do return GPU_MATERIAL_INVALID
	return store.meshes[mesh_index].material
}
