package BF_GPU

import vk "vendor:vulkan"

Vulkan_Image :: struct {
    image: vk.Image,
    memory: vk.DeviceMemory,
    view: vk.ImageView,
    format: vk.Format,
    width: u32,
    height: u32,
    mip_count: u32,
}

