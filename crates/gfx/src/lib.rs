//! Vulkan layer for HBD.
//!
//! `hbd-gfx` owns the instance, surface, physical/logical device, queues, and
//! memory allocator. It knows nothing about passes, materials, or ECS. The
//! renderer builds on top of this crate.

pub mod context;
pub mod memory;
pub mod vulkan;

pub use context::VulkanContext;
pub use gpu_allocator::MemoryLocation;
pub use memory::{GpuBuffer, GpuImage, SharedAllocator};
pub use vulkan::{GfxError, VulkanDeviceInfo, VulkanInfo, probe, version_string};
