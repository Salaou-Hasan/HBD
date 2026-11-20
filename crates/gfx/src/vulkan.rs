use std::ffi::CStr;

use ash::{Entry, vk};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GfxError {
    #[error("failed to load the Vulkan loader: {0}")]
    Loader(String),

    #[error("Vulkan call failed: {0:?}")]
    Vulkan(vk::Result),

    #[error("no Vulkan physical device supports graphics and presentation")]
    NoSuitableDevice,

    #[error("the window has a zero-sized drawable area")]
    ZeroSizedWindow,

    #[error("no supported swapchain format")]
    NoSwapchainFormat,

    #[error("no supported depth format")]
    NoDepthFormat,

    #[error("failed to read the native window handle: {0}")]
    WindowHandle(String),

    #[error("GPU allocation failed: {0}")]
    Allocation(String),

    #[error("buffer is not host-visible")]
    NotHostVisible,

    #[error("write exceeds the buffer size")]
    BufferOverflow,

    #[error("cannot upload an empty buffer")]
    EmptyBuffer,
}

#[derive(Debug, Clone)]
pub struct VulkanDeviceInfo {
    pub name: String,
    pub api_version: u32,
    pub device_type: vk::PhysicalDeviceType,
}

#[derive(Debug, Clone)]
pub struct VulkanInfo {
    pub loader_version: u32,
    pub devices: Vec<VulkanDeviceInfo>,
}

/// Loads the Vulkan loader, creates a temporary instance, and enumerates devices.
///
/// This is the M0 smoke test. It deliberately creates no device and no queues.
pub fn probe() -> Result<VulkanInfo, GfxError> {
    unsafe {
        let entry = Entry::load().map_err(|error| GfxError::Loader(error.to_string()))?;

        let loader_version = entry
            .try_enumerate_instance_version()
            .map_err(GfxError::Vulkan)?
            .unwrap_or(vk::API_VERSION_1_0);
        let requested_version = loader_version.min(vk::API_VERSION_1_3);

        let application_info = vk::ApplicationInfo::default()
            .application_name(c"hbd-probe")
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(c"hbd")
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(requested_version);

        let create_info = vk::InstanceCreateInfo::default().application_info(&application_info);
        let instance = entry
            .create_instance(&create_info, None)
            .map_err(GfxError::Vulkan)?;

        let physical_devices = instance
            .enumerate_physical_devices()
            .map_err(GfxError::Vulkan)?;

        let mut devices = Vec::with_capacity(physical_devices.len());
        for device in physical_devices {
            let properties = instance.get_physical_device_properties(device);
            let name = CStr::from_ptr(properties.device_name.as_ptr())
                .to_string_lossy()
                .into_owned();

            devices.push(VulkanDeviceInfo {
                name,
                api_version: properties.api_version,
                device_type: properties.device_type,
            });
        }

        instance.destroy_instance(None);

        Ok(VulkanInfo {
            loader_version,
            devices,
        })
    }
}

pub fn version_string(version: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(version),
        vk::api_version_minor(version),
        vk::api_version_patch(version)
    )
}
