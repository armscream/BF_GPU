// BF_GPU/Window.odin
//
// SDL3 window + Vulkan surface ownership, plus the Input_Backend
// service that pumps SDL events for BF_Input to consume.
//
// BF_GPU is the only module that creates an SDL3 window by default.
// The window lifetime is owned here: created during module_register
// (before the Vulkan device is queried for surface support), resized
// via SDL events that the renderer listens for, destroyed during
// module_unload. Alternative renderers (BF_SDL3_GPU) own their own
// windows; BF_Input is renderer-agnostic and binds to whichever
// Input_Backend the loaded renderer registers.

package BF_GPU

import "core:c"
import "core:log"
import "base:runtime"
import sdl "vendor:sdl3"
import "../../Core"
import INPUT "../BF_Input"

// ===========================================================================
// Window state + handle. The internal pointer is typed as ^sdl.Window
// inside this file because it owns the SDL lifecycle; the public
// Window_Handle wraps it as rawptr so the rest of the renderer never
// imports SDL.
// ===========================================================================

@(private)
Window_State :: struct {
	allocator:      runtime.Allocator,
	window:         ^sdl.Window, // SDL3 window; nil when headless
	surface:        u64, // VkSurfaceKHR (opaque; backend fills)
	width:          u32,
	height:         u32,
	resize_pending: bool,
	// Pointer to the engine's shared Engine_State (exposed via
	// lib_context_query). BF_GPU is the OS-window owner, so it owns
	// the close-button translation: a registered platform pump
	// polls SDL each frame and sets engine_state.quit_requested when
	// it sees SDL_EVENT_QUIT. The engine run loop checks that flag
	// at the top of every frame.
	//
	// BF_Input's DAG system may also see the same event via
	// sdl3_input_poll_event; that's fine. The two paths are
	// idempotent — whichever runs first sets the flag.
	engine_state:   ^Core.Engine_State,
	input_backend:  ^INPUT.Input_Backend,
	lib_ctx:        ^Core.Lib_Context,
	initialized:    bool,
}

@(private)
WINDOW_STATE_VALUE: Window_State = {}

// Window_Handle is the public type used by everything in BF_GPU that
// needs the SDL window (swapchain, surface creation). It wraps the
// raw pointer so the rest of the renderer never sees SDL types.
Window_Handle :: struct {
	handle:    rawptr, // ^sdl.Window (opaque outside Window.odin)
	width:     u32,
	height:    u32,
}

// Public service name. Other modules (BF_SDL3_GPU in the alternative
// renderer, the editor) can register their own window under this name
// if they want to override BF_GPU's.
WINDOW_SERVICE_NAME :: "BF_GPU.Window"

//* Lifecycle
window_init :: proc(st: ^Window_State, title: cstring, w, h: u32, allocator := context.allocator) -> bool {
	st.allocator = allocator
	st.width  = w
	st.height = h

	// Project.toml may omit Window_Width/Height (e.g. when the user
	// didn't author that section). Fall back to a sane default.
	if st.width == 0 {
		st.width = 800
		log.info("Project settings does not list Window_State.width: ")
	}
	if st.height == 0 {
		st.height = 600
		log.info("Project settings does not list Window_State.height: ")
	}

	// SDL_Init is idempotent across repeated calls; video + events +
	// gamepad covers everything the renderer + input backend need.
	if !sdl.Init({.VIDEO, .EVENTS, .GAMEPAD}) {
		log.errorf("[BF_GPU] SDL_Init failed: %s", sdl.GetError())
		return false
	}

	st.window = sdl.CreateWindow(title, c.int(st.width), c.int(st.height), sdl.WINDOW_VULKAN | sdl.WINDOW_RESIZABLE)
	if st.window == nil {
		log.errorf("[BF_GPU] SDL_CreateWindow failed: %s", sdl.GetError())
		sdl.QuitSubSystem({.VIDEO, .EVENTS, .GAMEPAD})
		return false
	}

	// Surface creation is deferred until the Vulkan device exists
	// (VkSurfaceKHR needs an instance handle). The renderer fills
	// window_create_surface() during the device-creation step.
	st.surface = 0

	// Register the Input_Backend service so BF_Input can find it.
	st.input_backend = new(INPUT.Input_Backend, allocator)
	st.input_backend.name        = "BF_GPU.InputBackend"
	st.input_backend.poll_event  = sdl3_input_poll_event
	st.input_backend.query       = sdl3_input_query
	st.input_backend.connected   = sdl3_input_connected
	st.input_backend.shutdown    = sdl3_input_shutdown

	if ctx := st.lib_ctx; ctx != nil {
		register_input_backend_service(ctx, st.input_backend)
		// Fetch the engine's shared Engine_State once and cache it.
		// The platform pump (registered below) uses it to set
		// quit_requested when the user clicks the close button.
		if raw := Core.lib_context_query(ctx, Core.CORE_LIB_INTERFACE_ENGINE_STATE, Core.ENGINE_STATE_API_VERSION); raw != nil {
			st.engine_state = cast(^Core.Engine_State)raw
		}
	}

	st.initialized = true

	// Register the close-button pump. Runs once per frame on the
	// engine main thread, BEFORE the DAG dispatches, so the quit
	// flag is visible to engine_poll_quit on the same frame.
	//
	// IMPORTANT: route via engine_state.register_pump, not the SDK
	// helper Core.engine_register_platform_pump. The SDK helper
	// writes to the DLL's own (nil) GLOBAL_ENGINE_STATE; this
	// indirection lands the registration in the engine executable's
	// GLOBAL_ENGINE_STATE where engine.run reads it.
	if st.engine_state != nil && st.engine_state.register_pump != nil {
		if st.engine_state.register_pump("BF_GPU.SDLClose", sdl3_pump_sdl_events) {
			log.info("[BF_GPU] platform pump registered")
		}
	}

	log.infof("[BF_GPU] window created (%dx%d) + Input_Backend registered", st.width, st.height)
	return true
}

// sdl3_pump_sdl_events is the engine platform pump. Runs once per
// frame on the engine main thread; drains SDL's event queue and
// translates window-level signals into BF_GPU/Engine_State flags.
//
// The run loop calls this before the DAG dispatches, so a close
// detected here exits the loop on the SAME frame (engine_poll_quit
// sees engine_state.quit_requested at the top of the next frame).
//
// Idempotent: any other path that also drains SDL events (BF_Input's
// DAG system via sdl3_input_poll_event) will simply see no events
// left.
@(private = "file")
sdl3_pump_sdl_events :: proc() {
	if !WINDOW_STATE_VALUE.initialized do return

	ev: sdl.Event
	for sdl.PollEvent(&ev) {
		#partial switch ev.type {
		case .QUIT:
			// User clicked the close button (or OS asked the
			// window to close for any other reason: shutdown,
			// taskkill, ...). Latch the engine's shared quit
			// flag and stop polling — SDL's queue is now empty
			// for this event.
			if WINDOW_STATE_VALUE.engine_state != nil {
				WINDOW_STATE_VALUE.engine_state.quit_requested = true
			}
			log.info("[BF_GPU] SDL_EVENT_QUIT -> engine_state.quit_requested = true")
		case .WINDOW_RESIZED:
			// Latched so the renderer's swapchain rebuild sees
			// the new size next frame.
			WINDOW_STATE_VALUE.resize_pending = true
		}
	}
}

window_destroy :: proc(st: ^Window_State) {
	// Called by module_unload, which only runs after the engine has
	// exited the run loop. The run loop exits because BF_Input's quit
	// provider reported a close-window or engine-shutdown event.
	// BF_GPU never destroys the window in response to observing an
	// SDL event itself; close-window flow is:
	//
	//   SDL user clicks X
	//     -> sdl3_input_poll_event translates SDL_EVENT_QUIT to .Quit
	//     -> BF_Input.poll_events_step latches close_window = true
	//     -> engine.run loop polls engine_poll_quit() -> true
	//     -> engine_quit() sets ENGINE_RUNNING = false
	//     -> run() returns, engine.destroy() runs
	//     -> module_unload -> window_destroy (here)
	if !st.initialized do return

	if st.window != nil {
		sdl.DestroyWindow(st.window)
		st.window = nil
	}
	st.surface = 0

	if st.input_backend != nil {
		if st.input_backend.shutdown != nil do st.input_backend.shutdown(st.input_backend)
		free(st.input_backend, st.allocator)
		st.input_backend = nil
	}

	// Release the SDL3 subsystems. We don't call SDL_Quit() here in
	// case other modules still reference SDL; the OS reclaims at
	// process exit. QuitSubSystem lets the same process re-init later.
	sdl.QuitSubSystem({.VIDEO, .EVENTS, .GAMEPAD})

	st.initialized = false
	log.info("[BF_GPU] window destroyed")
}

// window_pump_events drains the SDL event queue once per frame. Called
// from BF_GPU.FrameRecord before frame_record_step runs. Resize events
// set st.resize_pending; the swapchain + HiZ image rebuilds when the
// flag is observed.
window_pump_events :: proc(st: ^Window_State) {
	if !st.initialized do return
	pump_sdl_events_into_backend(st.input_backend)
}

// ===========================================================================
//* Accessors
// ===========================================================================

window_get_handle :: proc() -> Window_Handle {
	return Window_Handle{
		handle = WINDOW_STATE_VALUE.window,
		width  = WINDOW_STATE_VALUE.width,
		height = WINDOW_STATE_VALUE.height,
	}
}

window_get_surface :: proc() -> u64 { return WINDOW_STATE_VALUE.surface }

window_set_surface :: proc(surface: u64) {
	WINDOW_STATE_VALUE.surface = surface
}

window_resize_pending :: proc() -> bool { return WINDOW_STATE_VALUE.resize_pending }
window_clear_resize :: proc() { WINDOW_STATE_VALUE.resize_pending = false }

// ===========================================================================
// Internal: SDL3 -> Raw_Input_Event translation. Pumps keyboard, mouse,
// gamepad, and window events into the Input_Backend's poll_event queue.
//
// The outer caller (BF_Input.poll_events_step) calls poll_event in a
// loop until it returns false. We drain SDL's internal queue on every
// call so events don't pile up between frames. We only RETURN true for
// kinds the input runtime currently understands; everything else is
// drained but discarded for now (per-frame motion deltas are tracked
// by SDL3 internally and queried via `query`; we don't need them on
// the event stream for v1).
// ===========================================================================

@(private = "file")
sdl3_input_poll_event :: proc(backend: ^INPUT.Input_Backend, out: ^INPUT.Raw_Input_Event) -> bool {
	_ = backend
	if out == nil do return false

	ev: sdl.Event
	for sdl.PollEvent(&ev) {
		if ev.type == .QUIT {
			// SDL_EVENT_QUIT is the user-requested quit (window
			// close button, OS terminate, ...). Translate to a
			// .Quit raw event; BF_Input.poll_events_step latches
			// the close_window flag on its own state.
			out^ = INPUT.Raw_Input_Event{
				kind      = .Quit,
				binding   = INPUT.Raw_Input_Binding{},
				value     = INPUT.Input_Action_Value{},
				timestamp = 0,
			}
			return true
		} else if ev.type == .WINDOW_RESIZED {
			// Latched on the window state so the renderer's
			// swapchain rebuild sees the new size next frame.
			// Not an input action — keep draining.
			WINDOW_STATE_VALUE.resize_pending = true
		}
		// All other SDL3 event kinds are drained and discarded.
		// TODO: translate KEY/MOUSE/GAMEPAD events when needed.
	}
	return false
}

@(private = "file")
sdl3_input_query :: proc(backend: ^INPUT.Input_Backend, binding: INPUT.Raw_Input_Binding) -> INPUT.Input_Action_Value {
	_ = backend
	// Stub: query SDL_GetKeyboardState / SDL_GetGamepadAxis etc.
	// and translate to Input_Action_Value.
	_ = binding
	return INPUT.Input_Action_Value{}
}

@(private = "file")
sdl3_input_connected :: proc(backend: ^INPUT.Input_Backend, binding: INPUT.Raw_Input_Binding) -> bool {
	_ = backend
	// Stub: SDL_GamepadConnected for gamepad sources; true for keyboard/mouse.
	_ = binding
	return true
}

@(private = "file")
sdl3_input_shutdown :: proc(backend: ^INPUT.Input_Backend) {
	_ = backend
	// window_destroy handles the matching SDL_QuitSubSystem call so
	// the subsystems stay alive for the lifetime of the window. This
	// callback fires from window_destroy too, but it's safe to call
	// QuitSubSystem redundantly — SDL3 makes it a no-op for
	// subsystems that aren't initialised.
}

@(private = "file")
pump_sdl_events_into_backend :: proc(backend: ^INPUT.Input_Backend) {
	_ = backend
	// SDL3's SDL_PollEvent pumps events internally; there's no
	// separate "pump then poll" split as in SDL2. The poll_event
	// callback above is the single drain point. Kept as a no-op so
	// legacy callers don't crash if they invoke it.
}

// ===========================================================================
//* Internal: lib-context plumbing. Mirror of Renderer.odin's pattern.
// ===========================================================================

@(private = "file")
WINDOW_LIB_CTX: struct {
	service_reg: ^Core.Service_Registry,
	ctx:         ^Core.Lib_Context,
}

window_set_lib_context :: proc(reg: ^Core.Service_Registry, ctx: ^Core.Lib_Context) {
	WINDOW_LIB_CTX.service_reg = reg
	WINDOW_LIB_CTX.ctx = ctx
	WINDOW_STATE_VALUE.lib_ctx = ctx
}

@(private = "file")
register_input_backend_service :: proc(ctx: ^Core.Lib_Context, backend: ^INPUT.Input_Backend) {
	api_raw := Core.lib_context_query(
		ctx,
		Core.CORE_LIB_INTERFACE_COMPONENT_REGISTRATION,
		Core.COMPONENT_REGISTRATION_API_VERSION,
	)
	if api_raw == nil {
		log.warn("[BF_GPU] component_registration unavailable; Input_Backend service not registered")
		return
	}
	api := cast(^Core.Component_Registration_API)api_raw

	sreg := Core.Service_Registration {
		name     = INPUT.INPUT_BACKEND_SERVICE_NAME,
		instance = cast(rawptr)backend,
		destroy  = nil, // owned by WINDOW_STATE
	}
	if !api.add_service(ctx, sreg) {
		log.warn("[BF_GPU] failed to register Input_Backend service")
	}
}

vulkan_create_surface :: proc() -> bool {
	window := window_get_handle()
	if window.handle == nil {log.error("[BF_GPU] window handle is null"); return false}

	sdl_window := cast(^sdl.Window)window.handle
	if !sdl.Vulkan_CreateSurface(sdl_window, VULKAN_STATE.instance, nil, &VULKAN_STATE.surface){
		log.errorf("[BF_GPU/Vulkan] ")
	}
	return false
}

window_vulkan_instance_extensions :: proc() -> []cstring {
	count: u32
	names := sdl.Vulkan_GetInstanceExtensions(&count)
	if names == nil || count == 0 {
		log.error("[BF_GPU/Vulkan] SDL returned no Vulkan instance extensions")
		return nil
	}
	result := make([]cstring, count)
	for i in 0..<count {result[i] = names[i]}
	return result
}