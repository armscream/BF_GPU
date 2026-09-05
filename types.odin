//BF_Renderer/types.odin
package BF_Renderer

// Culling ordering
// ModelCull -> MeshCull -> MeshletCull
// Chunk(spatial)(CPU) -> Frustum -> Zero-Pixel -> Hi-Z -> Cone (meshlet)
// LOD is selected per Simple Large bound box and screen space % then indexed. 


// Spitballed types to think about for rendering.
RENDER_BUCKET_COUNT :: 8

Render_Bucket_Key :: struct {
    domain: Render_Domain,
    pipeline: Render_Pipeline,
    material_class: Render_Material_Class,
}

Render_Domain :: enum { 
    Geometry,
    Particle,
    Decal,
    Terrain,
    Water,
    UI,
    Shadow,
    Post_Process,
}

Render_Bucket :: enum u8 {
    Traditional_Opaque_1Sided,
    Traditional_Opaque_2Sided,
    Traditional_Transparent_1Sided,
    Traditional_Transparent_2Sided,
    Mesh_Opaque_1Sided,
    Mesh_Opaque_2Sided,
    Mesh_Transparent_1Sided,
    Mesh_Transparent_2Sided,
}

// Render scene should ref assets and not own them.
Render_Scene :: struct {
//    render_instances: []Render_Instance,
//   transforms: []Render_Transform,
//    lights: []Render_Light,
//    cameras: []Camera,
//    particle_emitters: []Particle_Emitter,
//    material_overrides: []Material_Override,
//    spatial_render_metadata: []Spatial_Render_Metadata, //Spatial/render metadata
}

// The renderer can then maintain its own 
// GPU Model Table
// GPU Mesh Table
// GPU Material Table
// GPU Texture Table