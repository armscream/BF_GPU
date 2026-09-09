package BF_GPU

//* MODEL
gpu_model_find :: #force_inline proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
) -> GPU_Model_ID {
	if store == nil || asset == Asset_ID(0) do return GPU_MODEL_INVALID
	if id, ok := store.model_by_asset[asset]; ok do return id
	return GPU_MODEL_INVALID
}

//* MESH
gpu_mesh_find :: #force_inline proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
) -> GPU_Mesh_ID {
	if store == nil || asset == Asset_ID(0) do return GPU_MESH_INVALID
	if id, ok := store.mesh_by_asset[asset]; ok do return id
	return GPU_MESH_INVALID
}

//* MATERIAL
gpu_material_find :: #force_inline proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
) -> GPU_Material_ID {
	if store == nil || asset == Asset_ID(0) do return GPU_MATERIAL_INVALID
	if id, ok := store.material_by_asset[asset]; ok do return id
	return GPU_MATERIAL_INVALID
}

//* TEXTURE
gpu_texture_find :: #force_inline proc(
	store: ^GPU_Resource_Store,
	asset: Asset_ID,
) -> GPU_Texture_ID {
	if store == nil || asset == Asset_ID(0) do return GPU_TEXTURE_INVALID
	if id, ok := store.texture_by_asset[asset]; ok do return id
	return GPU_TEXTURE_INVALID
}