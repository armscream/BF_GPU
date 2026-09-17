// BF_GPU/Vk_Shader.odin
//
// Shader compilation, SPIR-V caching, and VkShaderModule lifecycle for
// the BF_GPU Vulkan backend.
//
// The backend exposes the same compile_shader / destroy_shader surface
// it advertises through GPU_Backend (Renderer.odin), but the real work
// happens here: GLSL source is hashed, looked up in an in-memory cache,
// or compiled by shelling out to glslangValidator, then wrapped in a
// VkShaderModule that the pipeline layer can bind.
//
// The on-disk SPIR-V cache lives under <project_root>/bin/shader_cache/.
// Each entry is named after the FNV-1a hash of (canonicalized path +
// define string); a session reuses them so a warm start can skip the
// glslangValidator invocation entirely. A separate VkPipelineCache
// (also persisted in that directory) carries the driver-side pipeline
// blob between sessions.
//
// Extension shaders (e.g. BF_GPU_Mesh's task+mesh pair) live outside
// the BF_GPU/Shaders root. The path resolver walks a known list of
// shader roots so extensions can hand the renderer a relative path
// without baking the BF_GPU_Mesh directory into the renderer.

package BF_GPU

import "core:fmt"
import "core:log"
import "core:mem"
import "core:os"
import "core:strings"
import vk "vendor:vulkan"

Shader_Stage :: enum u8 {
	Vertex,
	Fragment,
	Compute,
	Task,
	Mesh,
}

Shader_Stage_To_Extension_Map := [Shader_Stage]string {
	.Vertex   = "vert",
	.Fragment = "frag",
	.Compute  = "comp",
	.Task     = "task",
	.Mesh     = "mesh",
}

Shader_Handle :: distinct u64
SHADER_HANDLE_INVALID :: Shader_Handle(0)

Shader_Key :: struct {
	path_hash:   u64,
	define_hash: u64,
}

Vulkan_Shader :: struct {
	handle:    Shader_Handle,
	stage:     Shader_Stage,
	key:       Shader_Key,
	module:    vk.ShaderModule,
	canonical: string,
	defines:   string,
	refcount:  int,
}

Vulkan_Shader_Map :: map[Shader_Key]^Vulkan_Shader

@(private)
VULKAN_SHADER_MAP: Vulkan_Shader_Map

@(private)
VULKAN_SHADER_NEXT_ID: u64 = 1

@(private)
VULKAN_PIPELINE_CACHE: vk.PipelineCache

@(private)
VULKAN_PIPELINE_CACHE_FILE: string = "bin/shader_cache/pipeline.cache"

// Shader_Root_Paths is the lookup order used by vulkan_resolve_shader_path
// when a caller hands in a non-absolute path. The BF_GPU root is the
// primary location for every renderer-owned shader; the extension root
// is consulted so BF_GPU_Mesh can hand its shaders to the renderer as
// relative paths.
@(private)
SHADER_ROOT_PATHS := []string {
	"Engine/src/Modules/BF_GPU/Shaders/",
	"Engine/src/Extensions/BF_GPU_Mesh/Shaders/",
}

// Precompiled_Root_Paths is the lookup order for SPIR-V files produced
// by `rune build` (see Project/rbs/shader_build.odin). The offline
// shader compile step writes `<profile.output>/shaders/<source-relative>.spv`
// preserving the source tree under the BF_GPU and BF_GPU_Mesh roots.
// At runtime, the renderer consults these paths first; only on miss
// does it fall back to SHADER_ROOT_PATHS + glslangValidator. This keeps
// cold-start startup fast (no glslang fork per shader) and matches the
// behaviour the project describes in README.md.
//
// These paths are joined with the executable directory at lookup time
// (vulkan_resolve_shader_path), not with the process working
// directory. The two typically differ: `rune build` writes shaders
// next to the executable (bin/Debug/shaders/), but the process CWD
// during a `rune run` invocation is the project root.
@(private)
PRECOMPILED_ROOT_PATHS := []string {
	"shaders/Engine/src/Modules/BF_GPU/Shaders/",
	"shaders/Engine/src/Extensions/BF_GPU_Mesh/Shaders/",
}

// Extension_To_Stage is a deferred map; populated by
// vulkan_init_shader_stage_map() the first time vulkan_resolve_shader_path
// runs, since dynamic-literal map initialisers are not allowed by
// default.
@(private)
EXTENSION_TO_STAGE: map[string]Shader_Stage

// vulkan_init_shader_stage_map populates EXTENSION_TO_STAGE. Idempotent
// so it is safe to call from both the init path and the test path.
vulkan_init_shader_stage_map :: proc() {
	if len(EXTENSION_TO_STAGE) > 0 do return
	EXTENSION_TO_STAGE["vert"] = .Vertex
	EXTENSION_TO_STAGE["frag"] = .Fragment
	EXTENSION_TO_STAGE["comp"] = .Compute
	EXTENSION_TO_STAGE["task"] = .Task
	EXTENSION_TO_STAGE["mesh"] = .Mesh
}

// ---------------------------------------------------------------------------
//* Initialisation / shutdown.

// vulkan_shader_cache_init / vulkan_shader_cache_shutdown manage the
// in-memory cache. The on-disk pipeline cache is created lazily on the
// first pipeline build; destroying it happens on shutdown after the
// device is idle.
vulkan_shader_cache_init :: proc() -> bool {
	if VULKAN_SHADER_MAP == nil {
		VULKAN_SHADER_MAP = make(Vulkan_Shader_Map, 32)
	}
	vulkan_init_shader_stage_map()
	return true
}

vulkan_shader_cache_shutdown :: proc() {
	if VULKAN_SHADER_MAP == nil do return
	for _, entry in VULKAN_SHADER_MAP {
		if entry == nil do continue
		if entry.module != {} && VULKAN_STATE.device != nil {
			vk.DestroyShaderModule(VULKAN_STATE.device, entry.module, nil)
		}
		delete(entry.canonical)
		delete(entry.defines)
		free(entry)
	}
	clear(&VULKAN_SHADER_MAP)
	VULKAN_SHADER_NEXT_ID = 1
}

// vulkan_pipeline_cache_init creates a fresh VkPipelineCache. If the
// cache file from a previous run exists, its contents are loaded into
// the cache object so a warm start can skip re-compiling pipelines.
vulkan_pipeline_cache_init :: proc() -> bool {
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] pipeline_cache_init before device creation")
		return false
	}
	if VULKAN_PIPELINE_CACHE != {} do return true

	initial_data, initial_size := vulkan_load_pipeline_cache_file()

	create_info := vk.PipelineCacheCreateInfo {
		sType           = .PIPELINE_CACHE_CREATE_INFO,
		initialDataSize = initial_size,
		pInitialData    = raw_data(initial_data) if initial_size > 0 else nil,
		flags           = {},
	}

	result := vk.CreatePipelineCache(
		VULKAN_STATE.device,
		&create_info,
		nil,
		&VULKAN_PIPELINE_CACHE,
	)
	if initial_data != nil do delete(initial_data)

	if result != .SUCCESS {
		log.errorf("[BF_GPU/Vulkan] vkCreatePipelineCache failed: %v", result)
		VULKAN_PIPELINE_CACHE = {}
		return false
	}
	return true
}

// vulkan_pipeline_cache_shutdown destroys the VkPipelineCache. When
// supported, the cache's serialised data is written to disk first so
// the next session can preload it via vulkan_pipeline_cache_init.
vulkan_pipeline_cache_shutdown :: proc() {
	if VULKAN_STATE.device == nil || VULKAN_PIPELINE_CACHE == {} do return

	data: []byte
	size: int = 0
	result := vk.GetPipelineCacheData(VULKAN_STATE.device, VULKAN_PIPELINE_CACHE, &size, nil)
	if result == .SUCCESS && size > 0 {
		data = make([]byte, size)
		result = vk.GetPipelineCacheData(VULKAN_STATE.device, VULKAN_PIPELINE_CACHE, &size, raw_data(data))
		if result != .SUCCESS {
			log.warnf("[BF_GPU/Vulkan] GetPipelineCacheData failed: %v", result)
			delete(data)
			data = nil
		}
	}
	if data != nil {
		vulkan_write_pipeline_cache_file(data)
		delete(data)
	}

	vk.DestroyPipelineCache(VULKAN_STATE.device, VULKAN_PIPELINE_CACHE, nil)
	VULKAN_PIPELINE_CACHE = {}
}

vulkan_pipeline_cache_get :: proc() -> vk.PipelineCache {
	return VULKAN_PIPELINE_CACHE
}

vulkan_load_pipeline_cache_file :: proc() -> ([]byte, int) {
	data, err := os.read_entire_file_from_path(VULKAN_PIPELINE_CACHE_FILE, context.allocator)
	if err != nil {
		return nil, 0
	}
	return data, len(data)
}

vulkan_write_pipeline_cache_file :: proc(data: []byte) {
	if len(data) == 0 do return

	dir := os.dir(VULKAN_PIPELINE_CACHE_FILE)
	if dir != "" && dir != "." && dir != "/" {
		os.make_directory_all(dir)
	}
	err := os.write_entire_file_from_bytes(VULKAN_PIPELINE_CACHE_FILE, data)
	if err != nil {
		log.warnf("[BF_GPU/Vulkan] failed to persist pipeline cache: %v", err)
	}
}

// ---------------------------------------------------------------------------
//* Path resolution.

// vulkan_resolve_shader_path turns a renderer-supplied path into an
// absolute, canonical path the compiler can read. Absolute paths are
// returned unchanged. Relative paths are looked up under each entry of
// SHADER_ROOT_PATHS until one matches. The returned string is a heap
// copy the caller owns.
//
// BF_GPU_Mesh's task/mesh pair (and any future extension shader) can
// be referenced by their relative-to-extension path; this is how the
// extension registers its shaders with the renderer without leaking
// its absolute on-disk location into Renderer.odin or the GLSL
// includes.
//
// Returns ("", false) when no root contains the path; the caller is
// expected to log and abort the compile.
//
// The resolver synthesises a canonical path even when the source file
// is not on disk. That allows the runtime to load a pre-compiled
// SPIR-V produced by `rune build` (under PRECOMPILED_ROOT_PATHS)
// without requiring the original GLSL source to be present. The
// canonical key must remain stable across runs so the in-memory
// shader cache survives a re-execute of the engine.
//
// Order of preference for the synthesised path:
//   1. The first SHADER_ROOT_PATHS entry that contains the requested
//      file on disk (preserves the on-repo dev workflow).
//   2. The first SHADER_ROOT_PATHS entry paired with a pre-compiled
//      .spv under PRECOMPILED_ROOT_PATHS (cold-start from
//      bin/Debug/shaders/ when the source was never copied).
//   3. Hard failure.
vulkan_resolve_shader_path :: proc(requested: string) -> (string, bool) {
	if requested == "" do return "", false

	if os.is_absolute_path(requested) {
		canonical, err := filepath_to_canonical(requested)
		if err != nil {
			return "", false
		}
		return canonical, true
	}

	for i in 0 ..< len(SHADER_ROOT_PATHS) {
		joined, _ := os.join_path({SHADER_ROOT_PATHS[i], requested}, context.temp_allocator)
		if _, stat_err := os.stat(joined, context.temp_allocator); stat_err == nil {
			canonical, err := filepath_to_canonical(joined)
			if err != nil {
				continue
			}
			return canonical, true
		}
	}

	// No source file matched; try the pre-compiled tree. If a
	// corresponding .spv exists under any PRECOMPILED_ROOT_PATHS
	// entry, synthesise the canonical path that the runtime cache
	// would have used for the source. The first matching root wins,
	// matching the SHADER_ROOT_PATHS / PRECOMPILED_ROOT_PATHS index
	// pairing.
	for i in 0 ..< len(PRECOMPILED_ROOT_PATHS) {
		if i >= len(SHADER_ROOT_PATHS) do break
		precompiled, _ := os.join_path(
			{PRECOMPILED_ROOT_PATHS[i], strings.concatenate({requested, ".spv"}, context.temp_allocator)},
			context.temp_allocator,
		)
		// Resolve against the executable directory first; the
		// offline `rune build` shader step writes next to the
		// exe, not next to the process CWD. Fall back to the
		// process CWD for in-repo dev where the layout matches.
		_, stat_err := os.stat(precompiled, context.temp_allocator)
		if stat_err != nil {
			exe_dir, exe_err := os.get_executable_directory(context.temp_allocator)
			if exe_err == nil {
				exe_candidate, _ := os.join_path(
					{exe_dir, precompiled},
					context.temp_allocator,
				)
				if _, exe_stat_err := os.stat(exe_candidate, context.temp_allocator); exe_stat_err == nil {
					precompiled = exe_candidate
					stat_err = nil
				}
			}
		}
		if stat_err == nil {
			synth, _ := os.join_path({SHADER_ROOT_PATHS[i], requested}, context.temp_allocator)
			canonical, err := filepath_to_canonical(synth)
			if err != nil do continue
			return canonical, true
		}
	}
	return "", false
}

// filepath_to_canonical returns the lowercase, normalised absolute path
// suitable as a stable hash input. The lowercasing matters because
// Windows file APIs treat casing as significant even though the
// underlying filesystem is case-insensitive.
filepath_to_canonical :: proc(p: string) -> (string, os.Error) {
	abs, abs_err := filepath_absolute(p)
	if abs_err != nil {
		return "", abs_err
	}
	return strings.to_lower(abs, context.temp_allocator), nil
}

filepath_absolute :: proc(p: string) -> (string, os.Error) {
	if os.is_absolute_path(p) do return p, nil
	cwd, err := os.get_working_directory(context.temp_allocator)
	if err != nil {
		return "", err
	}
	joined, _ := os.join_path({cwd, p}, context.temp_allocator)
	return joined, nil
}

// vulkan_stage_from_extension returns the matching Shader_Stage for a
// shader file's extension. Unknown extensions return false and force
// the caller to supply the stage explicitly (so the test fixtures can
// stage override without renaming the file).
vulkan_stage_from_extension :: proc(p: string) -> (Shader_Stage, bool) {
	ext := os.ext(p)
	if len(ext) > 0 && ext[0] == '.' {
		ext = ext[1:]
	}
	if ext == "" do return .Vertex, false
	vulkan_init_shader_stage_map()
	stage, found := EXTENSION_TO_STAGE[strings.to_lower(ext, context.temp_allocator)]
	return stage, found
}

// ---------------------------------------------------------------------------
//* Hashing.

// FNV-1a 64-bit. Stable across processes; matches the spirit of the
// glslang fingerprint without needing its internal hashing.
@(private)
FNV1A_OFFSET: u64 : 0xcbf29ce484222325

@(private)
FNV1A_PRIME: u64 : 0x100000001b3

fnv1a_init :: proc() -> u64 {
	return FNV1A_OFFSET
}

fnv1a_update :: proc(h: ^u64, data: []byte) {
	for b in data {
		h^ = (h^ ~ u64(b)) * FNV1A_PRIME
	}
}

fnv1a_string :: proc(s: string) -> u64 {
	h := fnv1a_init()
	for b in s {
		c: u64 = u64(b)
		h = (h ~ c) * FNV1A_PRIME
	}
	return h
}

// shader_key builds the lookup key for a shader source. path is the
// canonicalized on-disk path; defines is the renderer-supplied
// preprocessor define string (free-form, hashed as-is).
shader_key :: proc(path, defines: string) -> Shader_Key {
	return Shader_Key {
		path_hash   = fnv1a_string(path),
		define_hash = defines == "" ? 0 : fnv1a_string(defines),
	}
}

// hash_combine_u64 lives in Vk_Pipeline.odin so the pipeline-key
// builder can fold several u64 fields together without duplication.

// ---------------------------------------------------------------------------
//* GLSL -> SPIR-V.

// vulkan_compile_glsl_to_spirv runs glslangValidator and returns the
// compiled SPIR-V as a []u32 (SPIR-V is little-endian 32-bit words).
// The output is allocated with the caller's allocator.
//
// The function shells out to glslangValidator because the alternative
// (linking SPIRV-Tools / glslang as an Odin foreign library) costs
// build time and only matters for offline baking. At runtime the
// renderer's shader compilation is rare and dominated by glslang's
// own startup cost.
//
// Errors from glslang are captured and logged with the offending path
// and stage so debugging a broken shader points at the exact line.
vulkan_compile_glsl_to_spirv :: proc(
	path: string,
	stage: Shader_Stage,
	defines: string,
	allocator := context.allocator,
) -> (
	[]u32,
	bool,
) {
	if path == "" {
		log.error("[BF_GPU/Vulkan] shader compile: empty path")
		return nil, false
	}

	glslang := vulkan_locate_glslang()

	stage_ext := Shader_Stage_To_Extension_Map[stage]

	// Compose a unique output filename in the project's shader_cache
	// directory. The shader key's hash means two processes racing on
	// the same shader land in the same file; the in-memory cache
	// prevents duplicate compilation regardless.
	key := shader_key(path, defines)
	cwd, cwd_err := os.get_working_directory(context.temp_allocator)
	if cwd_err != nil {
		log.errorf("[BF_GPU/Vulkan] shader compile: cannot resolve cwd: %v", cwd_err)
		return nil, false
	}
	output_dir, _ := os.join_path({cwd, "bin", "shader_cache"}, context.temp_allocator)
	os.make_directory_all(output_dir)

	output_filename := fmt.tprintf(
		"bf_shader_%s_%s.spv",
		u64_to_hex(key.path_hash),
		u64_to_hex(key.define_hash),
	)
	output_path, _ := os.join_path({output_dir, output_filename}, context.temp_allocator)

	args: [dynamic]string
	append(&args, glslang)
	append(&args, "--target-env", "spirv1.6")
	append(&args, "-V")
	append(&args, "-S", stage_ext)
	if defines != "" {
		define_arg := strings.concatenate({"-D", defines}, context.temp_allocator)
		append(&args, define_arg)
	}
	append(&args, "-o", output_path)
	append(&args, path)

	desc := os.Process_Desc {
		command = args[:],
	}

	_, _, stderr, exec_err := os.process_exec(desc, context.temp_allocator)
	if exec_err != nil {
		log.errorf("[BF_GPU/Vulkan] glslangValidator exec failed: %v", exec_err)
		return nil, false
	}

	if len(stderr) > 0 {
		log.errorf(
			"[BF_GPU/Vulkan] glslangValidator failed for %s (stage %s, defines %q):\n%s",
			path,
			stage_ext,
			defines,
			string(stderr),
		)
		return nil, false
	}

	spirv_bytes, file_err := os.read_entire_file_from_path(output_path, allocator)
	if file_err != nil {
		log.errorf("[BF_GPU/Vulkan] could not read SPIR-V from %s: %v", output_path, file_err)
		return nil, false
	}

	if len(spirv_bytes) == 0 || len(spirv_bytes) % 4 != 0 {
		log.errorf(
			"[BF_GPU/Vulkan] SPIR-V blob for %s is misaligned (%d bytes)",
			path,
			len(spirv_bytes),
		)
		delete(spirv_bytes)
		return nil, false
	}

	spirv := mem_byte_slice_to_u32(spirv_bytes)
	return spirv, true
}

// vulkan_locate_glslang resolves the glslangValidator executable. The
// order is: $VK_SDK_PATH/Bin, $VULKAN_SDK/Bin, then whatever is on PATH.
// Returns a string with no allocator so the caller can use it as a
// short-lived cstring argument to process_exec.
vulkan_locate_glslang :: proc() -> string {
	env_vk_sdk, found_vk := os.lookup_env_alloc("VK_SDK_PATH", context.temp_allocator)
	if found_vk && env_vk_sdk != "" {
		candidate, _ := os.join_path({env_vk_sdk, "Bin", "glslangValidator.exe"}, context.temp_allocator)
		if _, err := os.stat(candidate, context.temp_allocator); err == nil {
			return candidate
		}
	}
	env_vulkan_sdk, found_vk_sdk := os.lookup_env_alloc("VULKAN_SDK", context.temp_allocator)
	if found_vk_sdk && env_vulkan_sdk != "" {
		candidate, _ := os.join_path({env_vulkan_sdk, "Bin", "glslangValidator.exe"}, context.temp_allocator)
		if _, err := os.stat(candidate, context.temp_allocator); err == nil {
			return candidate
		}
	}
	// Fall back to PATH lookup. Odin doesn't expose a portable
	// `which`; we trust PATH on Windows because glslangValidator.exe
	// is the only common glslang binary.
	return "glslangValidator.exe"
}

mem_byte_slice_to_u32 :: proc(b: []byte) -> []u32 {
	if len(b) == 0 do return nil
	out := make([]u32, len(b) / 4)
	for i in 0 ..< len(out) {
		v: u32
		v |= u32(b[i * 4 + 0])
		v |= u32(b[i * 4 + 1]) << 8
		v |= u32(b[i * 4 + 2]) << 16
		v |= u32(b[i * 4 + 3]) << 24
		out[i] = v
	}
	return out
}

// u64_to_hex renders a u64 as a 16-character lowercase hex string.
// Used in temp-file naming so the filename is stable and short.
u64_to_hex :: proc(v: u64) -> string {
	hex := "0123456789abcdef"
	out := make([]byte, 16)
	value := v
	for i in 0 ..< 16 {
		out[15 - i] = hex[value & 0xF]
		value = value >> 4
	}
	return string(out)
}

// ---------------------------------------------------------------------------
//* SPIR-V -> VkShaderModule.

// vulkan_load_spirv_module creates a VkShaderModule from a SPIR-V
// word slice. The slice is freed by the caller (the cache owns the
// resulting module separately).
vulkan_load_spirv_module :: proc(spirv: []u32, stage: Shader_Stage) -> (vk.ShaderModule, bool) {
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] create_shader_module before device init")
		return {}, false
	}
	if len(spirv) == 0 {
		log.error("[BF_GPU/Vulkan] SPIR-V blob is empty")
		return {}, false
	}

	create_info := vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(spirv) * size_of(u32),
		pCode    = raw_data(spirv),
	}
	module: vk.ShaderModule
	result := vk.CreateShaderModule(VULKAN_STATE.device, &create_info, nil, &module)
	if result != .SUCCESS {
		log.errorf(
			"[BF_GPU/Vulkan] CreateShaderModule failed for %s stage: %v",
			Shader_Stage_To_Extension_Map[stage],
			result,
		)
		return {}, false
	}
	return module, true
}

// ---------------------------------------------------------------------------
//* Public GPU_Backend hook surface.

// vulkan_backend_compile_shader_impl is the GPU_Backend.compile_shader
// implementation. It hashes the (path, defines) pair, returns the
// cached entry if one exists, otherwise compiles + caches a fresh
// module. The returned rawptr points at the heap-allocated
// Vulkan_Shader; vulkan_backend_destroy_shader_impl decrements the
// refcount and frees the entry + VkShaderModule when no pipeline
// still holds it.
//
// When `rune build` has produced a pre-compiled SPIR-V under
// bin/Debug/shaders/ the source file does NOT have to exist on disk;
// the cache lookup is keyed off the source path's stem and the
// relative-to-PRECOMPILED_ROOT_PATH location is consulted directly.
// This keeps cold-start fast on machines that only fetched the build
// artefacts and lets developers skip glslangValidator entirely when
// the build step already produced SPIR-V.
vulkan_backend_compile_shader_impl :: proc(path, stage: cstring) -> rawptr {
	if path == nil || stage == nil do return nil
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] compile_shader before Vulkan device creation")
		return nil
	}

	stage_enum, stage_ok := vulkan_stage_from_cstring(stage)
	if !stage_ok {
		log.errorf("[BF_GPU/Vulkan] compile_shader: unknown stage %s", stage)
		return nil
	}

	canonical, path_ok := vulkan_resolve_shader_path(string(path))
	if !path_ok {
		log.errorf("[BF_GPU/Vulkan] compile_shader: cannot resolve %s", string(path))
		return nil
	}

	entry := vulkan_shader_get_or_create(canonical, "", stage_enum)
	if entry == nil do return nil
	return rawptr(entry)
}

// vulkan_backend_compile_shader_precompiled_only is a fast path for
// the case where the renderer knows the source is unavailable on
// disk (only the pre-compiled SPIR-V exists under bin/Debug/shaders/).
// It walks PRECOMPILED_ROOT_PATHS for the matching <remainder>.spv
// without consulting SHADER_ROOT_PATHS at all. Returns the cached
// entry on hit, nil on miss.
vulkan_backend_compile_shader_precompiled_only :: proc(
	path, stage: cstring,
) -> rawptr {
	if path == nil || stage == nil do return nil
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] compile_shader before Vulkan device creation")
		return nil
	}
	stage_enum, stage_ok := vulkan_stage_from_cstring(stage)
	if !stage_ok {
		log.errorf("[BF_GPU/Vulkan] compile_shader: unknown stage %s", stage)
		return nil
	}
	requested := string(path)
	if requested == "" do return nil
	// Synthesize a canonical key from the requested relative path so
	// the in-memory cache keys match the source-keyed path even when
	// the source file is absent.
	cwd, cwd_err := os.get_working_directory(context.temp_allocator)
	if cwd_err != nil do return nil
	canonical := strings.to_lower(
		strings.concatenate({cwd, os.Path_Separator_String, requested}, context.temp_allocator),
		context.temp_allocator,
	)

	key := shader_key(canonical, "")
	if entry, found := VULKAN_SHADER_MAP[key]; found && entry != nil {
		entry.refcount += 1
		return rawptr(entry)
	}
	spirv, spirv_ok := vulkan_load_precompiled_spirv_by_relative(requested)
	if !spirv_ok do return nil
	module, module_ok := vulkan_load_spirv_module(spirv, stage_enum)
	delete(spirv)
	if !module_ok do return nil
	entry := vulkan_register_shader(module, key, canonical, "", stage_enum)
	if entry == nil do return nil
	return rawptr(entry)
}

// vulkan_load_precompiled_spirv_by_relative looks up a relative
// shader path (e.g. "Passes/Culling/Geometry/GeometryModelCulling.comp")
// directly under PRECOMPILED_ROOT_PATHS with a `.spv` suffix. Returns
// the file's contents as []u32 or (nil, false).
vulkan_load_precompiled_spirv_by_relative :: proc(requested: string) -> ([]u32, bool) {
	if requested == "" do return nil, false
	lower := strings.to_lower(requested, context.temp_allocator)
	for root in PRECOMPILED_ROOT_PATHS {
		joined, _ := os.join_path(
			{root, strings.concatenate({lower, ".spv"}, context.temp_allocator)},
			context.temp_allocator,
		)
		data, err := os.read_entire_file_from_path(joined, context.temp_allocator)
		if err != nil || len(data) == 0 || len(data) % 4 != 0 do continue
		spirv := make([]u32, len(data) / 4)
		mem.copy_non_overlapping(raw_data(spirv), raw_data(data), len(data))
		return spirv, true
	}
	return nil, false
}

// vulkan_backend_compile_shader_with_defines_impl is the defines-aware
// overload used by BF_GPU_Mesh (and any future extension) when its
// shaders need a feature-gate preprocessor define. The renderer-side
// GPU_Backend interface exposes the defines-less form because every
// traditional shader compiles cleanly without defines; the
// defines-aware form is wired directly into the pipeline-creation
// path so extensions can opt in.
vulkan_backend_compile_shader_with_defines_impl :: proc(
	path: cstring,
	stage: cstring,
	defines: cstring,
) -> rawptr {
	if path == nil || stage == nil do return nil
	if VULKAN_STATE.device == nil {
		log.error("[BF_GPU/Vulkan] compile_shader (defines) before Vulkan device creation")
		return nil
	}

	stage_enum, stage_ok := vulkan_stage_from_cstring(stage)
	if !stage_ok {
		log.errorf("[BF_GPU/Vulkan] compile_shader: unknown stage %s", stage)
		return nil
	}

	canonical, path_ok := vulkan_resolve_shader_path(string(path))
	if !path_ok {
		log.errorf("[BF_GPU/Vulkan] compile_shader: cannot resolve %s", string(path))
		return nil
	}

	define_str := string(defines) if defines != nil else ""
	entry := vulkan_shader_get_or_create(canonical, define_str, stage_enum)
	if entry == nil do return nil
	return rawptr(entry)
}

// vulkan_backend_destroy_shader_impl is the GPU_Backend.destroy_shader
// implementation. Decrements the refcount; the entry + VkShaderModule
// are released only when the last pipeline drops its reference.
vulkan_backend_destroy_shader_impl :: proc(s: rawptr) {
	if s == nil do return
	entry := cast(^Vulkan_Shader)s
	vulkan_shader_release(entry)
}

// vulkan_shader_get_or_create returns a refcounted cache entry for the
// (canonical_path, defines) pair. The first call pays the full cost of
// glslangValidator; subsequent calls hit the in-memory cache.
//
// Lookup order on cold start:
//   1. In-memory shader cache (refcounted)
//   2. Pre-compiled SPIR-V under PRECOMPILED_ROOT_PATHS (the
//      `rune build` shader step writes here; cold start prefers this
//      because glslangValidator startup is expensive).
//   3. GLSL source under SHADER_ROOT_PATHS, compiled via
//      vulkan_compile_glsl_to_spirv (fallback for machines without a
//      prebuilt shader cache).
vulkan_shader_get_or_create :: proc(
	canonical: string,
	defines: string,
	stage: Shader_Stage,
) -> ^Vulkan_Shader {
	key := shader_key(canonical, defines)
	if entry, found := VULKAN_SHADER_MAP[key]; found && entry != nil {
		entry.refcount += 1
		return entry
	}

	// Try the pre-compiled SPIR-V path first. The relative path
	// passed to vulkan_backend_compile_shader_impl is preserved as
	// the suffix under PRECOMPILED_ROOT_PATHS; the .comp source
	// extension is replaced with .spv because the offline step
	// writes `<source-name>.<stage-ext>.spv` (e.g.
	// GeometryModelCulling.comp.spv).
	precompiled_spirv, precompiled_ok := vulkan_load_precompiled_spirv(canonical, defines)
	if precompiled_ok {
		module, module_ok := vulkan_load_spirv_module(precompiled_spirv, stage)
		delete(precompiled_spirv)
		if !module_ok do return nil
		return vulkan_register_shader(module, key, canonical, defines, stage)
	}

	spirv, spirv_ok := vulkan_compile_glsl_to_spirv(canonical, stage, defines)
	if !spirv_ok do return nil
	defer delete(spirv)

	module, module_ok := vulkan_load_spirv_module(spirv, stage)
	if !module_ok do return nil

	return vulkan_register_shader(module, key, canonical, defines, stage)
}

// vulkan_load_precompiled_spirv searches PRECOMPILED_ROOT_PATHS for a
// SPIR-V file matching the canonical source path. Returns the file's
// contents as a []u32 SPIR-V word slice, or (nil, false) on miss.
//
// The `rune build` shader step writes files with the suffix `.spv`
// appended to the source name (e.g. GeometryModelCulling.comp
// becomes GeometryModelCulling.comp.spv) under
// `<profile.output>/shaders/`. The pre-compiled tree preserves the
// source tree under that prefix.
//
// canonical is the absolute, lower-cased source path that
// vulkan_resolve_shader_path returned. It lives under one of the
// SHADER_ROOT_PATHS entries (lowercased). To find the matching
// pre-compiled SPIR-V, we strip the lower-cased SHADER_ROOT_PATHS
// prefix and append the remainder (with `.spv` suffix) under the
// matching PRECOMPILED_ROOT_PATHS entry.
//
// Search is rooted at the executable directory (bin/Debug/) plus
// fallback to the process CWD, mirroring the resolution policy in
// vulkan_resolve_shader_path.
vulkan_load_precompiled_spirv :: proc(
	canonical: string,
	defines: string,
) -> (
	[]u32,
	bool,
) {
	_ = defines
	if canonical == "" do return nil, false

	// Iterate SHADER_ROOT_PATHS / PRECOMPILED_ROOT_PATHS in matching
	// pairs so each source-root maps to its pre-compiled twin. The
	// canonical path is absolute and lower-cased; the SHADER_ROOT_PATHS
	// literals are relative and may use `/` or `\`. We look for the
	// source-root substring anywhere in canonical (the typical
	// match is at offset `len("c:/.../<project_root>/")`).
	lower_canonical := strings.to_lower(canonical, context.temp_allocator)
	// Normalise separators so case-folded Windows absolute paths
	// (which use `\`) match the SHADER_ROOT_PATHS literals (which use
	// `/`).
	lower_canonical, _ = strings.replace_all(lower_canonical, "\\", "/", context.temp_allocator)
	for i in 0 ..< len(SHADER_ROOT_PATHS) {
		if i >= len(PRECOMPILED_ROOT_PATHS) do break
		source_root_lower := strings.to_lower(SHADER_ROOT_PATHS[i], context.temp_allocator)
		source_root_normalised := strings.trim_right(source_root_lower, "/")
		idx := strings.index(lower_canonical, source_root_normalised)
		if idx < 0 do continue
		remainder := lower_canonical[idx + len(source_root_normalised):]
		joined, _ := os.join_path(
			{PRECOMPILED_ROOT_PATHS[i], strings.concatenate({remainder, ".spv"}, context.temp_allocator)},
			context.temp_allocator,
		)
		data, err := vulkan_read_shader_bytes(joined)
		if err != nil || len(data) == 0 || len(data) % 4 != 0 do continue
		spirv := make([]u32, len(data) / 4)
		mem.copy_non_overlapping(raw_data(spirv), raw_data(data), len(data))
		return spirv, true
	}
	return nil, false
}

// vulkan_read_shader_bytes looks for a shader file first next to the
// executable (where `rune build` writes the pre-compiled output) and
// then in the process CWD. Returns the raw bytes on hit, an error on
// miss. The exe-directory lookup is mandatory because the offline
// build writes shaders to bin/Debug/shaders/, while the process CWD
// is the project root during a `rune run` invocation.
vulkan_read_shader_bytes :: proc(rel_path: string) -> ([]byte, os.Error) {
	if data, err := os.read_entire_file_from_path(rel_path, context.temp_allocator); err == nil {
		return data, nil
	}
	exe_dir, exe_err := os.get_executable_directory(context.temp_allocator)
	if exe_err != nil do return nil, os.ERROR_NONE
	abs, _ := os.join_path({exe_dir, rel_path}, context.temp_allocator)
	return os.read_entire_file_from_path(abs, context.temp_allocator)
}

// vulkan_register_shader creates a Vulkan_Shader entry, wraps the
// already-built VkShaderModule, and stores it in the cache. Pulled
// out of vulkan_shader_get_or_create so both the pre-compiled and
// runtime-compiled paths can share the cache registration.
vulkan_register_shader :: proc(
	module: vk.ShaderModule,
	key: Shader_Key,
	canonical: string,
	defines: string,
	stage: Shader_Stage,
) -> ^Vulkan_Shader {
	entry := new(Vulkan_Shader)
	entry^ = Vulkan_Shader {
		handle    = Shader_Handle(VULKAN_SHADER_NEXT_ID),
		stage     = stage,
		key       = key,
		module    = module,
		canonical = strings.clone(canonical),
		defines   = strings.clone(defines),
		refcount  = 1,
	}
	VULKAN_SHADER_MAP[key] = entry
	VULKAN_SHADER_NEXT_ID += 1

	log.infof(
		"[BF_GPU/Vulkan] compiled shader %s (stage %s, defines %q) -> handle=%d",
		canonical,
		Shader_Stage_To_Extension_Map[stage],
		defines,
		u64(entry.handle),
	)
	return entry
}

// vulkan_shader_acquire increments the refcount of an existing shader
// entry without compiling. Used by the pipeline layer when the same
// shader is bound to multiple stages.
vulkan_shader_acquire :: proc(entry: ^Vulkan_Shader) {
	if entry == nil do return
	entry.refcount += 1
}

// vulkan_shader_release decrements the refcount; when it hits zero the
// entry + VkShaderModule are released and the cache slot is freed.
vulkan_shader_release :: proc(entry: ^Vulkan_Shader) {
	if entry == nil do return
	entry.refcount -= 1
	if entry.refcount > 0 do return

	if entry.module != {} && VULKAN_STATE.device != nil {
		vk.DestroyShaderModule(VULKAN_STATE.device, entry.module, nil)
	}
	delete_key(&VULKAN_SHADER_MAP, entry.key)
	delete(entry.canonical)
	delete(entry.defines)
	free(entry)
}

// vulkan_shader_stage returns the Shader_Stage of an entry. The
// pipeline builder uses this to assemble VkPipelineShaderStageCreateInfo
// lists from the opaque rawptr handle the GPU_Backend returns.
vulkan_shader_stage :: proc(s: rawptr) -> Shader_Stage {
	if s == nil do return .Vertex
	entry := cast(^Vulkan_Shader)s
	return entry.stage
}

vulkan_shader_module :: proc(s: rawptr) -> vk.ShaderModule {
	if s == nil do return {}
	entry := cast(^Vulkan_Shader)s
	return entry.module
}

vulkan_shader_handle :: proc(s: rawptr) -> Shader_Handle {
	if s == nil do return SHADER_HANDLE_INVALID
	entry := cast(^Vulkan_Shader)s
	return entry.handle
}

// vulkan_stage_from_cstring maps the legacy cstring stage names
// ("vertex" / "compute" / ...) the renderer's GPU_Backend interface
// accepts onto the Shader_Stage enum.
vulkan_stage_from_cstring :: proc(s: cstring) -> (Shader_Stage, bool) {
	switch strings.to_lower(string(s), context.temp_allocator) {
	case "vertex":
		return .Vertex, true
	case "fragment":
		return .Fragment, true
	case "compute":
		return .Compute, true
	case "task":
		return .Task, true
	case "mesh":
		return .Mesh, true
	}
	return .Vertex, false
}

// ---------------------------------------------------------------------------
//* Cache observation helpers (tests + diagnostics).

vulkan_shader_cache_len :: proc() -> int {
	if VULKAN_SHADER_MAP == nil do return 0
	return len(VULKAN_SHADER_MAP)
}

vulkan_shader_cache_contains :: proc(canonical, defines: string) -> bool {
	if VULKAN_SHADER_MAP == nil do return false
	key := shader_key(canonical, defines)
	_, found := VULKAN_SHADER_MAP[key]
	return found
}

vulkan_shader_cache_clear :: proc() {
	vulkan_shader_cache_shutdown()
}