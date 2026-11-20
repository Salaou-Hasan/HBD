//! RAII GPU buffers and images backed by `gpu-allocator`.
//!
//! The allocator is shared through an `Arc<Mutex<_>>` so resources can free
//! their own suballocations when they are dropped. All resources must outlive
//! the device that created them; `VulkanContext` is ordered after renderer
//! resources for exactly this reason.

use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::vulkan::GfxError;

pub type SharedAllocator = Arc<Mutex<Allocator>>;

pub struct GpuBuffer {
    device: ash::Device,
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    allocator: SharedAllocator,
    size: u64,
    mapped: Option<NonNull<u8>>,
    host_coherent: bool,
}

impl GpuBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &SharedAllocator,
        create_info: &vk::BufferCreateInfo<'_>,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self, GfxError> {
        let buffer =
            unsafe { device.create_buffer(create_info, None) }.map_err(GfxError::Vulkan)?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = match allocator.lock() {
            Ok(mut allocator) => allocator.allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }),
            Err(_) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(GfxError::Allocation("allocator mutex poisoned".into()));
            }
        };

        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(GfxError::Allocation(error.to_string()));
            }
        };

        let memory = unsafe { allocation.memory() };
        let offset = allocation.offset();
        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, offset) } {
            if let Ok(mut allocator) = allocator.lock() {
                let _ = allocator.free(allocation);
            }
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(GfxError::Vulkan(error));
        }

        let mapped = allocation.mapped_ptr().map(|pointer| pointer.cast::<u8>());
        let host_coherent = allocation
            .memory_properties()
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT);

        Ok(Self {
            device: device.clone(),
            buffer,
            allocation: Some(allocation),
            allocator: allocator.clone(),
            size: create_info.size,
            mapped,
            host_coherent,
        })
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Copies bytes into persistently mapped host-visible memory.
    pub fn write(&self, data: &[u8]) -> Result<(), GfxError> {
        if data.len() as u64 > self.size {
            return Err(GfxError::BufferOverflow);
        }
        let Some(mapped) = self.mapped else {
            return Err(GfxError::NotHostVisible);
        };

        // The mapped pointer is offset-adjusted by gpu-allocator.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped.as_ptr(), data.len());
        }

        if !self.host_coherent {
            let Some(allocation) = self.allocation.as_ref() else {
                return Err(GfxError::NotHostVisible);
            };
            let range = [vk::MappedMemoryRange::default()
                .memory(unsafe { allocation.memory() })
                .offset(allocation.offset())
                .size(allocation.size())];
            unsafe { self.device.flush_mapped_memory_ranges(&range) }.map_err(GfxError::Vulkan)?;
        }

        Ok(())
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe { self.device.destroy_buffer(self.buffer, None) };
        if let Some(allocation) = self.allocation.take()
            && let Ok(mut allocator) = self.allocator.lock()
        {
            let _ = allocator.free(allocation);
        }
    }
}

pub struct GpuImage {
    device: ash::Device,
    image: vk::Image,
    allocation: Option<Allocation>,
    allocator: SharedAllocator,
}

impl GpuImage {
    pub fn new(
        device: &ash::Device,
        allocator: &SharedAllocator,
        create_info: &vk::ImageCreateInfo<'_>,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self, GfxError> {
        let image = unsafe { device.create_image(create_info, None) }.map_err(GfxError::Vulkan)?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = match allocator.lock() {
            Ok(mut allocator) => allocator.allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }),
            Err(_) => {
                unsafe { device.destroy_image(image, None) };
                return Err(GfxError::Allocation("allocator mutex poisoned".into()));
            }
        };

        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(error) => {
                unsafe { device.destroy_image(image, None) };
                return Err(GfxError::Allocation(error.to_string()));
            }
        };

        let memory = unsafe { allocation.memory() };
        let offset = allocation.offset();
        if let Err(error) = unsafe { device.bind_image_memory(image, memory, offset) } {
            if let Ok(mut allocator) = allocator.lock() {
                let _ = allocator.free(allocation);
            }
            unsafe { device.destroy_image(image, None) };
            return Err(GfxError::Vulkan(error));
        }

        Ok(Self {
            device: device.clone(),
            image,
            allocation: Some(allocation),
            allocator: allocator.clone(),
        })
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }
}

impl Drop for GpuImage {
    fn drop(&mut self) {
        unsafe { self.device.destroy_image(self.image, None) };
        if let Some(allocation) = self.allocation.take()
            && let Ok(mut allocator) = self.allocator.lock()
        {
            let _ = allocator.free(allocation);
        }
    }
}
