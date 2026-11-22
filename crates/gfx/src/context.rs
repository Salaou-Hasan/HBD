//! Vulkan instance, surface, device, swapchain, and the first frame loop.
//!
//! This is intentionally a single clear-screen path. It proves the native
//! surface, queue selection, swapchain recreation, presentation, and object
//! lifetime handling before shaders and pipelines are added.

use std::ffi::CStr;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tracing::{debug, error, info, warn};

use crate::memory::{GpuBuffer, GpuImage, SharedAllocator};
use crate::vulkan::{GfxError, version_string};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct InFlightFrame {
    command_buffer: vk::CommandBuffer,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
}

struct DebugMessenger {
    loader: ash::ext::debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = if callback_data.is_null() {
        None
    } else {
        let raw_message = unsafe { (*callback_data).p_message };
        if raw_message.is_null() {
            None
        } else {
            Some(unsafe { CStr::from_ptr(raw_message) }.to_string_lossy())
        }
    };
    let message = message
        .as_deref()
        .unwrap_or("validation message unavailable");
    let message_type = format!("{message_type:?}");

    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        error!(%message_type, %message, "Vulkan validation");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        warn!(%message_type, %message, "Vulkan validation");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        info!(%message_type, %message, "Vulkan validation");
    } else {
        debug!(%message_type, %message, "Vulkan validation");
    }

    vk::FALSE
}

/// Owns the Vulkan objects needed to present a window.
///
/// The context is created and used on the thread that owns the window. It is
/// not `Send`; the render thread will own its own command recording later.
pub struct VulkanContext {
    // Keeps the loader alive for the lifetime of the instance.
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_messenger: Option<DebugMessenger>,
    surface: vk::SurfaceKHR,
    surface_loader: ash::khr::surface::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    // Dropped explicitly before `destroy_device`: the allocator frees its
    // remaining `vkDeviceMemory` blocks through the device on drop.
    allocator: Option<SharedAllocator>,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    graphics_queue_family: u32,
    present_queue_family: u32,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    /// Incremented on every swapchain (re)creation, including the implicit
    /// recreations triggered by `ERROR_OUT_OF_DATE_KHR`/suboptimal presents.
    /// Consumers use this to know when their per-image resources are stale.
    swapchain_generation: u64,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    in_flight_frames: Vec<InFlightFrame>,
    current_frame: usize,
    width: u32,
    height: u32,
    /// `0.0` means timestamp queries are unsupported; callers must check.
    timestamp_period: f32,
    timestamps_supported: bool,
    /// Presentation policy restored on every swapchain (re)creation.
    present_policy: PresentPolicy,
}

/// Presentation policy. `Vsync` is the default and never tears. `Uncapped`
/// prefers MAILBOX when available, otherwise IMMEDIATE: it removes the refresh
/// cap but may tear on displays without MAILBOX. Both are logged explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PresentPolicy {
    #[default]
    Vsync,
    Uncapped,
}

impl VulkanContext {
    pub fn new<W>(window: &W, width: u32, height: u32) -> Result<Self, GfxError>
    where
        W: HasWindowHandle + HasDisplayHandle + ?Sized,
    {
        Self::new_with_present_policy(window, width, height, PresentPolicy::Vsync)
    }

    /// Constructor with an explicit presentation policy. The engine and apps
    /// must opt in to uncapped presentation; the default is vsync.
    pub fn new_with_present_policy<W>(
        window: &W,
        width: u32,
        height: u32,
        present_policy: PresentPolicy,
    ) -> Result<Self, GfxError>
    where
        W: HasWindowHandle + HasDisplayHandle + ?Sized,
    {
        if width == 0 || height == 0 {
            return Err(GfxError::ZeroSizedWindow);
        }

        let display_handle = window
            .display_handle()
            .map_err(|error| GfxError::WindowHandle(error.to_string()))?
            .as_raw();
        let window_handle = window
            .window_handle()
            .map_err(|error| GfxError::WindowHandle(error.to_string()))?
            .as_raw();

        unsafe {
            let entry = ash::Entry::load().map_err(|error| GfxError::Loader(error.to_string()))?;
            let loader_version = entry
                .try_enumerate_instance_version()
                .map_err(GfxError::Vulkan)?
                .unwrap_or(vk::API_VERSION_1_0);
            let requested_version = loader_version.min(vk::API_VERSION_1_3);

            let base_extensions = ash_window::enumerate_required_extensions(display_handle)
                .map_err(GfxError::Vulkan)?;

            let validation_requested = match std::env::var("HBD_VALIDATION").ok().as_deref() {
                Some("0") => false,
                Some(_) => true,
                None => cfg!(debug_assertions),
            };
            let validation_layer = c"VK_LAYER_KHRONOS_validation";
            let validation_available = entry
                .enumerate_instance_layer_properties()
                .map_err(GfxError::Vulkan)?
                .iter()
                .any(|layer| CStr::from_ptr(layer.layer_name.as_ptr()) == validation_layer);
            let debug_utils_available = entry
                .enumerate_instance_extension_properties(None)
                .map_err(GfxError::Vulkan)?
                .iter()
                .any(|extension| {
                    CStr::from_ptr(extension.extension_name.as_ptr()) == ash::ext::debug_utils::NAME
                });
            let enable_validation =
                validation_requested && validation_available && debug_utils_available;

            if validation_requested && !enable_validation {
                warn!(
                    layer = validation_available,
                    debug_utils = debug_utils_available,
                    "Vulkan validation requested but unavailable; continuing without it"
                );
            }

            let mut layer_names = Vec::new();
            if enable_validation {
                layer_names.push(validation_layer.as_ptr());
            }

            let mut extension_names = base_extensions.to_vec();
            if enable_validation {
                extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
            }

            let application_info = vk::ApplicationInfo::default()
                .application_name(c"hbd")
                .application_version(vk::make_api_version(0, 0, 1, 0))
                .engine_name(c"hbd")
                .engine_version(vk::make_api_version(0, 0, 1, 0))
                .api_version(requested_version);

            let instance_info = vk::InstanceCreateInfo::default()
                .application_info(&application_info)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&extension_names);

            let instance = entry
                .create_instance(&instance_info, None)
                .map_err(GfxError::Vulkan)?;

            let debug_messenger = if enable_validation {
                let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(debug_callback));

                let messenger = loader
                    .create_debug_utils_messenger(&create_info, None)
                    .map_err(GfxError::Vulkan)?;
                info!("Vulkan validation layers enabled");

                Some(DebugMessenger { loader, messenger })
            } else {
                None
            };

            let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
            let surface =
                ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                    .map_err(GfxError::Vulkan)?;

            let (physical_device, graphics_queue_family, present_queue_family) =
                pick_physical_device(&instance, &surface_loader, surface)?;

            let properties = instance.get_physical_device_properties(physical_device);
            let device_name = CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();

            info!(
                device = %device_name,
                api = %version_string(properties.api_version),
                loader = %version_string(loader_version),
                graphics_queue = graphics_queue_family,
                present_queue = present_queue_family,
                "Vulkan device selected"
            );

            let timestamp_period = properties.limits.timestamp_period;
            let timestamp_valid_bits = instance
                .get_physical_device_queue_family_properties(physical_device)
                [graphics_queue_family as usize]
                .timestamp_valid_bits;
            let timestamps_supported = timestamp_valid_bits > 0 && timestamp_period > 0.0;
            info!(
                supported = timestamps_supported,
                valid_bits = timestamp_valid_bits,
                period_ns = timestamp_period,
                "GPU timestamp capability"
            );

            let mut unique_families = vec![graphics_queue_family];
            if present_queue_family != graphics_queue_family {
                unique_families.push(present_queue_family);
            }

            let queue_priorities = [1.0f32];
            let queue_create_infos: Vec<_> = unique_families
                .iter()
                .map(|&family| {
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(family)
                        .queue_priorities(&queue_priorities)
                })
                .collect();

            let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
            let device_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_create_infos)
                .enabled_extension_names(&device_extensions);

            let device = instance
                .create_device(physical_device, &device_info, None)
                .map_err(GfxError::Vulkan)?;

            let allocator = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .map_err(|error| GfxError::Allocation(error.to_string()))?;
            let allocator = Arc::new(Mutex::new(allocator));

            let graphics_queue = device.get_device_queue(graphics_queue_family, 0);
            let present_queue = if graphics_queue_family == present_queue_family {
                graphics_queue
            } else {
                device.get_device_queue(present_queue_family, 0)
            };

            let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

            let command_pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(graphics_queue_family);
            let command_pool = device
                .create_command_pool(&command_pool_info, None)
                .map_err(GfxError::Vulkan)?;

            let command_buffer_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);
            let command_buffers = device
                .allocate_command_buffers(&command_buffer_info)
                .map_err(GfxError::Vulkan)?;

            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let mut in_flight_frames = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

            for command_buffer in command_buffers {
                in_flight_frames.push(InFlightFrame {
                    command_buffer,
                    image_available: device
                        .create_semaphore(&semaphore_info, None)
                        .map_err(GfxError::Vulkan)?,
                    render_finished: device
                        .create_semaphore(&semaphore_info, None)
                        .map_err(GfxError::Vulkan)?,
                    in_flight: device
                        .create_fence(&fence_info, None)
                        .map_err(GfxError::Vulkan)?,
                });
            }

            let mut context = Self {
                _entry: entry,
                instance,
                debug_messenger,
                surface,
                surface_loader,
                physical_device,
                device,
                allocator: Some(allocator),
                graphics_queue,
                present_queue,
                graphics_queue_family,
                present_queue_family,
                swapchain_loader,
                swapchain: vk::SwapchainKHR::null(),
                swapchain_format: vk::Format::UNDEFINED,
                swapchain_extent: vk::Extent2D::default(),
                swapchain_images: Vec::new(),
                swapchain_image_views: Vec::new(),
                swapchain_generation: 0,
                render_pass: vk::RenderPass::null(),
                framebuffers: Vec::new(),
                command_pool,
                in_flight_frames,
                current_frame: 0,
                width,
                height,
                timestamp_period,
                timestamps_supported,
                present_policy,
            };

            context.create_swapchain()?;
            Ok(context)
        }
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<GpuBuffer, GfxError> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let allocator = self
            .allocator
            .as_ref()
            .expect("allocator is available while the context is alive");
        GpuBuffer::new(&self.device, allocator, &create_info, location, name)
    }

    pub fn create_image(
        &self,
        create_info: &vk::ImageCreateInfo<'_>,
        name: &str,
    ) -> Result<GpuImage, GfxError> {
        let allocator = self
            .allocator
            .as_ref()
            .expect("allocator is available while the context is alive");
        GpuImage::new(
            &self.device,
            allocator,
            create_info,
            MemoryLocation::GpuOnly,
            name,
        )
    }

    /// Uploads immutable data into a device-local buffer through staging memory.
    pub fn upload_buffer(
        &self,
        usage: vk::BufferUsageFlags,
        data: &[u8],
        name: &str,
    ) -> Result<GpuBuffer, GfxError> {
        if data.is_empty() {
            return Err(GfxError::EmptyBuffer);
        }

        let size = data.len() as vk::DeviceSize;
        let staging = self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "HBD upload staging",
        )?;
        staging.write(data)?;

        let buffer = self.create_buffer(
            size,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
            name,
        )?;
        self.copy_buffer(&staging, &buffer, size)?;

        Ok(buffer)
    }

    fn copy_buffer(
        &self,
        source: &GpuBuffer,
        destination: &GpuBuffer,
        size: vk::DeviceSize,
    ) -> Result<(), GfxError> {
        unsafe {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = self
                .device
                .allocate_command_buffers(&allocate_info)
                .map_err(GfxError::Vulkan)?;
            let command_buffer = command_buffers[0];

            let result = (|| {
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device
                    .begin_command_buffer(command_buffer, &begin_info)
                    .map_err(GfxError::Vulkan)?;

                let copy = [vk::BufferCopy::default()
                    .src_offset(0)
                    .dst_offset(0)
                    .size(size)];
                self.device.cmd_copy_buffer(
                    command_buffer,
                    source.buffer(),
                    destination.buffer(),
                    &copy,
                );
                self.device
                    .end_command_buffer(command_buffer)
                    .map_err(GfxError::Vulkan)?;

                let command_buffers = [command_buffer];
                let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
                self.device
                    .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                    .map_err(GfxError::Vulkan)?;
                self.device
                    .queue_wait_idle(self.graphics_queue)
                    .map_err(GfxError::Vulkan)
            })();

            self.device
                .free_command_buffers(self.command_pool, &command_buffers);
            result
        }
    }

    /// Resizes the swapchain. Zero-sized frames (minimized window) are ignored.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), GfxError> {
        if width == 0 || height == 0 {
            return Ok(());
        }
        if width == self.width && height == self.height {
            return Ok(());
        }

        self.width = width;
        self.height = height;
        self.recreate_swapchain()
    }

    /// Records off-screen work, then presents the swapchain image.
    ///
    /// `before_present` runs before the presentation render pass begins; it is
    /// where geometry passes record into their own framebuffers. `present`
    /// runs inside the presentation render pass and draws the final image.
    pub fn render<G, P>(
        &mut self,
        clear_color: [f32; 4],
        before_present: G,
        present: P,
    ) -> Result<(), GfxError>
    where
        G: FnOnce(&ash::Device, vk::CommandBuffer, u32, vk::Extent2D),
        P: FnOnce(&ash::Device, vk::CommandBuffer, u32, vk::Extent2D),
    {
        let frame_index = self.current_frame;
        let frame = &self.in_flight_frames[frame_index];
        let command_buffer = frame.command_buffer;
        let image_available = frame.image_available;
        let render_finished = frame.render_finished;
        let in_flight = frame.in_flight;

        unsafe {
            self.device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .map_err(GfxError::Vulkan)?;

            let acquire_result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            );

            let (image_index, _suboptimal) = match acquire_result {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain()?;
                    return Ok(());
                }
                Err(error) => return Err(GfxError::Vulkan(error)),
            };

            self.device
                .reset_fences(&[in_flight])
                .map_err(GfxError::Vulkan)?;
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(GfxError::Vulkan)?;

            let begin_info = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(GfxError::Vulkan)?;

            before_present(
                &self.device,
                command_buffer,
                image_index,
                self.swapchain_extent,
            );

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: clear_color,
                },
            }];
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain_extent,
                })
                .clear_values(&clear_values);

            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            present(
                &self.device,
                command_buffer,
                image_index,
                self.swapchain_extent,
            );

            self.device.cmd_end_render_pass(command_buffer);
            self.device
                .end_command_buffer(command_buffer)
                .map_err(GfxError::Vulkan)?;

            let wait_semaphores = [image_available];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [render_finished];
            let command_buffers = [command_buffer];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.device
                .queue_submit(self.graphics_queue, &[submit_info], in_flight)
                .map_err(GfxError::Vulkan)?;

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            match self
                .swapchain_loader
                .queue_present(self.present_queue, &present_info)
            {
                Ok(false) => {}
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain()?;
                }
                Err(error) => return Err(GfxError::Vulkan(error)),
            }
        }

        self.current_frame = (self.current_frame + 1) % self.in_flight_frames.len();
        Ok(())
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Vulkan instance, exposed for editor-side renderer backends.
    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    /// Graphics queue used by the engine; the editor's ImGui renderer shares it.
    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    /// Command pool for transient CPU-side uploads.
    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.swapchain_extent
    }

    pub fn swapchain_image_views(&self) -> &[vk::ImageView] {
        &self.swapchain_image_views
    }

    /// Monotonic counter for swapchain recreation. Render passes bound to
    /// per-image views must be rebuilt whenever this changes.
    pub fn swapchain_generation(&self) -> u64 {
        self.swapchain_generation
    }

    pub fn swapchain_format(&self) -> vk::Format {
        self.swapchain_format
    }

    /// Nanoseconds per timestamp tick. Callers must check `supports_timestamps`.
    pub fn timestamp_period(&self) -> f32 {
        self.timestamp_period
    }

    pub fn supports_timestamps(&self) -> bool {
        self.timestamps_supported
    }

    /// Executes a one-time command buffer and waits for it to finish.
    pub fn one_time_commands<F>(&self, f: F) -> Result<(), GfxError>
    where
        F: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        unsafe {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = self
                .device
                .allocate_command_buffers(&allocate_info)
                .map_err(GfxError::Vulkan)?;
            let command_buffer = command_buffers[0];

            let result = (|| {
                let begin_info = vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.device
                    .begin_command_buffer(command_buffer, &begin_info)
                    .map_err(GfxError::Vulkan)?;
                f(&self.device, command_buffer);
                self.device
                    .end_command_buffer(command_buffer)
                    .map_err(GfxError::Vulkan)?;

                let command_buffers = [command_buffer];
                let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
                self.device
                    .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                    .map_err(GfxError::Vulkan)?;
                self.device
                    .queue_wait_idle(self.graphics_queue)
                    .map_err(GfxError::Vulkan)
            })();

            self.device
                .free_command_buffers(self.command_pool, &command_buffers);
            result
        }
    }

    fn recreate_swapchain(&mut self) -> Result<(), GfxError> {
        unsafe {
            self.device.device_wait_idle().map_err(GfxError::Vulkan)?;
        }
        self.cleanup_swapchain();
        self.create_swapchain()
    }

    /// Chooses a depth format suitable for depth-stencil attachments.
    pub fn pick_depth_format(&self) -> Result<vk::Format, GfxError> {
        for format in [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ] {
            let properties = unsafe {
                self.instance
                    .get_physical_device_format_properties(self.physical_device, format)
            };
            if properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            {
                return Ok(format);
            }
        }

        Err(GfxError::NoDepthFormat)
    }

    fn create_swapchain(&mut self) -> Result<(), GfxError> {
        unsafe {
            let capabilities = self
                .surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .map_err(GfxError::Vulkan)?;

            let formats = self
                .surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)
                .map_err(GfxError::Vulkan)?;

            let format = formats
                .iter()
                .copied()
                .find(|format| {
                    format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                        && matches!(
                            format.format,
                            vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                        )
                })
                .or_else(|| {
                    formats
                        .iter()
                        .copied()
                        .find(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                })
                .or_else(|| formats.first().copied())
                .ok_or(GfxError::NoSwapchainFormat)?;

            let present_modes = self
                .surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
                .map_err(GfxError::Vulkan)?;

            let present_mode = choose_present_mode(&present_modes, self.present_policy);
            info!(
                modes = ?present_modes,
                chosen = ?present_mode,
                "swapchain present modes"
            );

            let extent = if capabilities.current_extent.width != u32::MAX {
                capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: self.width.clamp(
                        capabilities.min_image_extent.width,
                        capabilities.max_image_extent.width,
                    ),
                    height: self.height.clamp(
                        capabilities.min_image_extent.height,
                        capabilities.max_image_extent.height,
                    ),
                }
            };

            let mut image_count = capabilities.min_image_count + 1;
            if capabilities.max_image_count > 0 {
                image_count = image_count.min(capabilities.max_image_count);
            }

            let (sharing_mode, queue_family_indices) =
                if self.graphics_queue_family != self.present_queue_family {
                    (
                        vk::SharingMode::CONCURRENT,
                        vec![self.graphics_queue_family, self.present_queue_family],
                    )
                } else {
                    (vk::SharingMode::EXCLUSIVE, Vec::new())
                };

            let create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(sharing_mode)
                .queue_family_indices(&queue_family_indices)
                .pre_transform(capabilities.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(vk::SwapchainKHR::null());

            let swapchain = self
                .swapchain_loader
                .create_swapchain(&create_info, None)
                .map_err(GfxError::Vulkan)?;
            let images = self
                .swapchain_loader
                .get_swapchain_images(swapchain)
                .map_err(GfxError::Vulkan)?;

            self.swapchain = swapchain;
            self.swapchain_images = images;

            let image_subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            for index in 0..self.swapchain_images.len() {
                let image_view_info = vk::ImageViewCreateInfo::default()
                    .image(self.swapchain_images[index])
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(image_subresource_range);

                self.swapchain_image_views.push(
                    self.device
                        .create_image_view(&image_view_info, None)
                        .map_err(GfxError::Vulkan)?,
                );
            }

            let attachments = [vk::AttachmentDescription::default()
                .format(format.format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

            let color_attachment_reference = vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(std::slice::from_ref(&color_attachment_reference))];

            let dependencies = [
                vk::SubpassDependency::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                // Geometry passes write the HDR image before this pass samples it.
                vk::SubpassDependency::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ),
            ];

            let render_pass_info = vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies);

            self.render_pass = self
                .device
                .create_render_pass(&render_pass_info, None)
                .map_err(GfxError::Vulkan)?;

            for image_view in &self.swapchain_image_views {
                let attachments = [*image_view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(self.render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                self.framebuffers.push(
                    self.device
                        .create_framebuffer(&framebuffer_info, None)
                        .map_err(GfxError::Vulkan)?,
                );
            }

            self.swapchain_format = format.format;
            self.swapchain_extent = extent;
            self.swapchain_generation += 1;

            info!(
                format = ?format.format,
                present_mode = ?present_mode,
                width = extent.width,
                height = extent.height,
                images = self.swapchain_images.len(),
                "swapchain created"
            );
        }

        Ok(())
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for image_view in self.swapchain_image_views.drain(..) {
                self.device.destroy_image_view(image_view, None);
            }
            if self.render_pass != vk::RenderPass::null() {
                self.device.destroy_render_pass(self.render_pass, None);
                self.render_pass = vk::RenderPass::null();
            }
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader
                    .destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
            self.swapchain_images.clear();
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }

        self.cleanup_swapchain();

        // gpu-allocator's Drop frees its blocks through the device, so it must
        // run before `destroy_device`. Renderer resources are declared before
        // the context and are already gone at this point.
        if let Some(allocator) = self.allocator.take() {
            if Arc::strong_count(&allocator) > 1 {
                warn!("allocator still has live resource handles during context drop");
            }
            drop(allocator);
        }

        unsafe {
            for frame in self.in_flight_frames.drain(..) {
                self.device.destroy_semaphore(frame.image_available, None);
                self.device.destroy_semaphore(frame.render_finished, None);
                self.device.destroy_fence(frame.in_flight, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some(debug_messenger) = self.debug_messenger.take() {
                debug_messenger
                    .loader
                    .destroy_debug_utils_messenger(debug_messenger.messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32, u32), GfxError> {
    unsafe {
        let devices = instance
            .enumerate_physical_devices()
            .map_err(GfxError::Vulkan)?;
        let mut fallback = None;

        for device in devices {
            let properties = instance.get_physical_device_properties(device);
            if properties.api_version < vk::API_VERSION_1_3 {
                continue;
            }

            let families = instance.get_physical_device_queue_family_properties(device);
            let mut graphics_family = None;
            let mut present_family = None;

            for (index, family) in families.iter().enumerate() {
                let index = index as u32;

                if graphics_family.is_none()
                    && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                {
                    graphics_family = Some(index);
                }

                if present_family.is_none()
                    && surface_loader
                        .get_physical_device_surface_support(device, index, surface)
                        .map_err(GfxError::Vulkan)?
                {
                    present_family = Some(index);
                }
            }

            let (Some(graphics_family), Some(present_family)) = (graphics_family, present_family)
            else {
                continue;
            };

            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                return Ok((device, graphics_family, present_family));
            }

            if fallback.is_none() {
                fallback = Some((device, graphics_family, present_family));
            }
        }

        fallback.ok_or(GfxError::NoSuitableDevice)
    }
}

/// Chooses the presentation mode for a given surface and policy. Kept pure so
/// the FPS behaviour can be regression-tested without a GPU.
pub(crate) fn choose_present_mode(
    modes: &[vk::PresentModeKHR],
    policy: PresentPolicy,
) -> vk::PresentModeKHR {
    if modes.contains(&vk::PresentModeKHR::MAILBOX) {
        // Tear-free and uncapped: always preferable when available.
        vk::PresentModeKHR::MAILBOX
    } else if policy == PresentPolicy::Uncapped && modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
        // Explicit opt-in: no refresh cap, but tearing is possible.
        vk::PresentModeKHR::IMMEDIATE
    } else {
        vk::PresentModeKHR::FIFO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_vulkan_versions() {
        assert_eq!(version_string(vk::make_api_version(0, 1, 2, 3)), "1.2.3");
    }

    #[test]
    fn vsync_never_leaves_fifo_without_mailbox() {
        let modes = [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::FIFO];
        assert_eq!(
            choose_present_mode(&modes, PresentPolicy::Vsync),
            vk::PresentModeKHR::FIFO
        );
    }

    #[test]
    fn uncapped_prefers_mailbox_over_immediate() {
        let modes = [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::FIFO,
            vk::PresentModeKHR::MAILBOX,
        ];
        assert_eq!(
            choose_present_mode(&modes, PresentPolicy::Uncapped),
            vk::PresentModeKHR::MAILBOX
        );
    }

    #[test]
    fn uncapped_falls_back_to_immediate_then_fifo() {
        let modes = [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::FIFO];
        assert_eq!(
            choose_present_mode(&modes, PresentPolicy::Uncapped),
            vk::PresentModeKHR::IMMEDIATE
        );
        assert_eq!(
            choose_present_mode(&[vk::PresentModeKHR::FIFO], PresentPolicy::Uncapped),
            vk::PresentModeKHR::FIFO
        );
    }
}
