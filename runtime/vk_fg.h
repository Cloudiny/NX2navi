// Frame generation with AMD FSR 3.1 (vk_fg.cpp): AMD's frame generation swapchain in gfx_present.c's place.
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

int vk_fg_wanted(void);                      // NX2NAVI_FRAMEGEN = fsr3 with DLSS or FSR 3.1 upscaling on (live), where FSR 3.1 loads
uint32_t vk_fg_queue_count(void);            // graphics-family queues the device creates
void vk_fg_queues(VkQueue present, uint32_t present_family, VkQueue acquire, uint32_t acquire_family);   // vk_device.c, after the device exists
int vk_fg_active(void);                      // the frame generation swapchain exists
int vk_fg_generating(void);                  // ... and the last frame had generation on (not paused for the menu)
// gfx_present.c's swapchain calls while vk_fg_wanted: the first create makes AMD's swapchain context, later ones
// (window resizes) its swapchain again. Non-success from the first: the plain swapchain path stays.
VkResult vk_fg_create_swapchain(const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc);
void vk_fg_destroy_swapchain(VkSwapchainKHR sc);
VkResult vk_fg_get_images(VkSwapchainKHR sc, uint32_t *n, VkImage *img);
VkResult vk_fg_acquire(VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem, uint32_t *idx);
VkResult vk_fg_present(const VkPresentInfoKHR *pi);   // on the game's queue, its lock held by the caller
// Each present, before its command buffer is submitted: configures generation for this frame and records its
// preparation (the frame's depth and motion vectors, vk_dlaa_fg_inputs) into cb.
// hudless: this frame's picture without the HUD at the swapchain's size and format, in GENERAL (VK_NULL_HANDLE: none),
// from which generation tells the HUD apart and keeps it out of the generated frames.
// pic: the picture's rectangle in the swapchain {x, y, w, h} (black bars around it when the window has another shape);
// frames are generated inside it only.
void vk_fg_frame(VkCommandBuffer cb, VkSwapchainKHR sc, VkFormat fmt, VkExtent2D ext, VkImage hudless, VkImageView hudless_view,
                 const int pic[4]);
uint64_t vk_fg_presents(VkSwapchainKHR sc);   // frames the swapchain presented, generated ones included (0: off)
PFN_vkSetHdrMetadataEXT vk_fg_hdr_metadata_fn(void);   // AMD's replacement for vkSetHdrMetadataEXT while active, else NULL
void vk_fg_release(void);

// The same through AMD's FSR SDK 2.4 (vk_fg4.cpp), for vk_fg.cpp alone: it takes this path while FSR 4 is the upscaler,
// and every call above goes to these while this path holds the swapchain.
int vk_fg4_available(void);                  // its libraries are beside the executable and list a frame generation for the device
int vk_fg4_failed(void);                     // it failed in this run: FSR 3.1's library generates the frames
const char *vk_fg4_name(void);               // the frame generation version the loader lists first for the device ("3.1.6"), "" unknown
int vk_fg4_active(void);
int vk_fg4_generating(void);
void vk_fg4_queues(VkQueue present, uint32_t present_family, VkQueue acquire, uint32_t acquire_family);
VkResult vk_fg4_create_swapchain(const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc);
void vk_fg4_destroy_swapchain(VkSwapchainKHR sc);
VkResult vk_fg4_get_images(VkSwapchainKHR sc, uint32_t *n, VkImage *img);
VkResult vk_fg4_acquire(VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem, uint32_t *idx);
VkResult vk_fg4_present(const VkPresentInfoKHR *pi);
void vk_fg4_frame(VkCommandBuffer cb, VkSwapchainKHR sc, VkFormat fmt, VkExtent2D ext, VkImage hudless, const int pic[4]);
uint64_t vk_fg4_presents(VkSwapchainKHR sc);
PFN_vkSetHdrMetadataEXT vk_fg4_hdr_metadata_fn(void);
void vk_fg4_release(void);
