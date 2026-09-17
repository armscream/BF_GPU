// BF_GPU/Vk_Resource.odin
//
// Deferred destruction queue for resources whose GPU work has not yet
// completed.
//
// Vulkan spec: vkDestroyBuffer / vkDestroyImage must not be called while
// the GPU is still reading or writing the resource. The renderer can ask
// for a resource to be destroyed immediately (renderer shutdown,
// vkDeviceWaitIdle before exit) or queued until a known GPU completion
// fence / timeline value has signaled.
//
// `vulkan_defer_*` push a Vulkan_Deferred_Destruction onto the queue
// tagged with the GPU completion value that was last submitted. The
// submit step records the current graphics_timeline_value into
// `frame.completion_value`; vulkan_frame() blocks on it before
// re-recording, and vulkan_collect_garbage(value) reaps every entry
// tagged with a completion value <= value.
//
// Entries are taken from the queue in FIFO order; they are never
// reordered. Each tag is a u64 monotonic counter; the renderer passes
// the latest `VULKAN_STATE.graphics_timeline_value` to flush.

package BF_GPU

import vma "../../dependencies/odin-vma"
import "core:log"
import "core:sync"
import vk "vendor:vulkan"

@(private)
Vulkan_Deferred_Destruction :: struct {
	kind:    Vulkan_Deferred_Kind,
	tag:     u64, // GPU completion value at the time of deferral
	// At most one of these is set, depending on `kind`.
	image:   ^Vulkan_Image,
	view:    vk.ImageView,
	sampler: vk.Sampler,
	buffer:  ^Vulkan_Buffer,
}

Vulkan_Deferred_Kind :: enum {
	Buffer,
	Image,
	Image_View,
	Sampler,
}

@(private)
VULKAN_DEFER_QUEUE: [dynamic]Vulkan_Deferred_Destruction

// VULKAN_DEFER_QUEUE_LOCKED is set while the queue is being walked.
// Producers (frame submit path) refuse new entries while it is set;
// the flag is local to the GC path and not a thread-safety primitive.
@(private)
VULKAN_DEFER_QUEUE_LOCKED: bool

// VULKAN_DEFER_QUEUE_MUTEX serializes producers (frame submit, asset
// unload) and the GC consumer (top of vulkan_frame). The resource
// lifetime model assumes both can run concurrently: asset unloads
// can fire any frame, and the GC runs before the next submit so any
// in-flight work that referenced a deferred resource has completed.
@(private)
VULKAN_DEFER_QUEUE_MUTEX: sync.Mutex

// vulkan_defer_init / vulkan_defer_destroy manage the deferred queue
// lifecycle. Called from vulkan_init / vulkan_shutdown.
vulkan_defer_init :: proc() {
	sync.mutex_lock(&VULKAN_DEFER_QUEUE_MUTEX)
	defer sync.mutex_unlock(&VULKAN_DEFER_QUEUE_MUTEX)
	clear(&VULKAN_DEFER_QUEUE)
	VULKAN_DEFER_QUEUE_LOCKED = false
}

vulkan_defer_shutdown :: proc() {
	// Best-effort drain. Anything still queued is freed unconditionally;
	// vulkan_shutdown calls vkDeviceWaitIdle before this so the queue
	// is guaranteed to be empty in practice.
	sync.mutex_lock(&VULKAN_DEFER_QUEUE_MUTEX)
	defer sync.mutex_unlock(&VULKAN_DEFER_QUEUE_MUTEX)
	for entry in VULKAN_DEFER_QUEUE {
		switch entry.kind {
		case .Buffer:
			if entry.buffer != nil {
				vulkan_destroy_buffer_now(entry.buffer)
				free(entry.buffer)
			}
		case .Image:
			if entry.image != nil {
				vulkan_destroy_image_now(entry.image)
				free(entry.image)
			}
		case .Image_View:
			vulkan_destroy_image_view_now(entry.view)
		case .Sampler:
			vulkan_destroy_sampler_now(entry.sampler)
		}
	}
	clear(&VULKAN_DEFER_QUEUE)
}

// vulkan_defer_image_destruction pushes a Vulkan_Image onto the queue.
// The backend takes ownership of the heap allocation; vulkan_collect_garbage
// is the only path that frees it.
vulkan_defer_image_destruction :: proc(image: ^Vulkan_Image, tag: u64) {
	if image == nil do return
	sync.mutex_lock(&VULKAN_DEFER_QUEUE_MUTEX)
	defer sync.mutex_unlock(&VULKAN_DEFER_QUEUE_MUTEX)
	if VULKAN_DEFER_QUEUE_LOCKED {
		// Defensive: refuse new entries while the queue is being
		// walked so the iterator does not see a moving target.
		log.warn("[BF_GPU/Vulkan] defer_image_destruction called while queue is locked; leaking")
		return
	}
	append(&VULKAN_DEFER_QUEUE, Vulkan_Deferred_Destruction{
		kind  = .Image,
		tag   = tag,
		image = image,
	})
}

// vulkan_defer_image_view_destruction queues an image view for
// deferred destruction. The view handle being zero is allowed (tests
// use it as a placeholder); the actual vk.DestroyImageView call at
// reap time is a no-op when the handle is zero.
vulkan_defer_image_view_destruction :: proc(view: vk.ImageView, tag: u64) {
	sync.mutex_lock(&VULKAN_DEFER_QUEUE_MUTEX)
	defer sync.mutex_unlock(&VULKAN_DEFER_QUEUE_MUTEX)
	if VULKAN_DEFER_QUEUE_LOCKED {
		log.warn("[BF_GPU/Vulkan] defer_image_view_destruction called while queue is locked; leaking")
		return
	}
	append(&VULKAN_DEFER_QUEUE, Vulkan_Deferred_Destruction{
		kind = .Image_View,
		tag  = tag,
		view = view,
	})
}

// vulkan_defer_sampler_destruction queues a sampler for deferred
// destruction. As with image_view, a zero handle is allowed so tests
// can drive the queue lifecycle without a live device.
vulkan_defer_sampler_destruction :: proc(sampler: vk.Sampler, tag: u64) {
	sync.mutex_lock(&VULKAN_DEFER_QUEUE_MUTEX)
	defer sync.mutex_unlock(&VULKAN_DEFER_QUEUE_MUTEX)
	if VULKAN_DEFER_QUEUE_LOCKED {
		log.warn("[BF_GPU/Vulkan] defer_sampler_destruction called while queue is locked; leaking")
		return
	}
	append(&VULKAN_DEFER_QUEUE, Vulkan_Deferred_Destruction{
		kind    = .Sampler,
		tag     = tag,
		sampler = sampler,
	})
}

// vulkan_defer_buffer_destruction queues a buffer for deferred
// destruction. The backend takes ownership of the heap allocation;
// vulkan_collect_garbage is the only path that frees it. Used when an
// asset is reloaded: the previous mesh's vertex / index buffer is
// queued so the GPU can finish reading it before VMA releases the
// underlying VkDeviceMemory.
vulkan_defer_buffer_destruction :: proc(buffer: ^Vulkan_Buffer, tag: u64) {
	if buffer == nil do return
	sync.mutex_lock(&VULKAN_DEFER_QUEUE_MUTEX)
	defer sync.mutex_unlock(&VULKAN_DEFER_QUEUE_MUTEX)
	if VULKAN_DEFER_QUEUE_LOCKED {
		log.warn("[BF_GPU/Vulkan] defer_buffer_destruction called while queue is locked; leaking")
		return
	}
	append(&VULKAN_DEFER_QUEUE, Vulkan_Deferred_Destruction{
		kind   = .Buffer,
		tag    = tag,
		buffer = buffer,
	})
}

// vulkan_collect_garbage reaps every queued entry whose tag is <=
// up_to. up_to typically comes from VULKAN_STATE.graphics_timeline_value
// at the top of vulkan_frame() after the wait proves the GPU has finished
// the prior submission.
//
// Returns the number of entries actually destroyed (useful for stats).
vulkan_collect_garbage :: proc(up_to: u64) -> int {
	sync.mutex_lock(&VULKAN_DEFER_QUEUE_MUTEX)
	defer sync.mutex_unlock(&VULKAN_DEFER_QUEUE_MUTEX)
	if len(VULKAN_DEFER_QUEUE) == 0 do return 0
	VULKAN_DEFER_QUEUE_LOCKED = true
	defer VULKAN_DEFER_QUEUE_LOCKED = false

	destroyed := 0
	keep: [dynamic]Vulkan_Deferred_Destruction
	defer delete(keep)

	for entry in VULKAN_DEFER_QUEUE {
		if entry.tag <= up_to {
			switch entry.kind {
			case .Buffer:
				if entry.buffer != nil {
					vulkan_destroy_buffer_now(entry.buffer)
					free(entry.buffer)
				}
			case .Image:
				if entry.image != nil {
					vulkan_destroy_image_now(entry.image)
					free(entry.image)
				}
			case .Image_View:
				vulkan_destroy_image_view_now(entry.view)
			case .Sampler:
				vulkan_destroy_sampler_now(entry.sampler)
			}
			destroyed += 1
		} else {
			append(&keep, entry)
		}
	}

	// Replace the queue contents with the survivors (or empty it
	// outright when nothing survived). Mutating VULKAN_DEFER_QUEUE
	// in-place while iterating it would invalidate the iterator.
	clear(&VULKAN_DEFER_QUEUE)
	for entry in keep {
		append(&VULKAN_DEFER_QUEUE, entry)
	}

	if destroyed > 0 {
		log.infof("[BF_GPU/Vulkan] GC'd %d deferred destructions (tag<= %d)", destroyed, up_to)
	}
	return destroyed
}

// vulkan_defer_queue_len is exposed for tests + diagnostics. The
// snapshot is taken under the queue mutex so concurrent producers do
// not race with the caller.
vulkan_defer_queue_len :: proc() -> int {
	sync.mutex_lock(&VULKAN_DEFER_QUEUE_MUTEX)
	defer sync.mutex_unlock(&VULKAN_DEFER_QUEUE_MUTEX)
	return len(VULKAN_DEFER_QUEUE)
}