//! Off-screen HDR render targets for the geometry pass.
//!
//! One HDR color image and one depth image are kept per swapchain image so
//! frame N+1 cannot overwrite targets that frame N is still presenting. The
//! HDR color attachment ends in `SHADER_READ_ONLY_OPTIMAL`, ready for the
//! tonemap pass.

use ash::vk;
use hbd_gfx::{GfxError, GpuImage, VulkanContext};

use crate::RenderError;

pub(crate) const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub(crate) const VELOCITY_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

pub(crate) struct RenderTargets {
    device: ash::Device,
    // Kept alive so the image views and framebuffers remain backed by memory.
    _hdr_images: Vec<GpuImage>,
    _depth_images: Vec<GpuImage>,
    _velocity_images: Vec<GpuImage>,
    hdr_views: Vec<vk::ImageView>,
    velocity_views: Vec<vk::ImageView>,
    depth_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
}

impl RenderTargets {
    pub(crate) fn new(
        context: &VulkanContext,
        extent: vk::Extent2D,
        image_count: usize,
    ) -> Result<Self, RenderError> {
        let device = context.device().clone();
        let depth_format = context.pick_depth_format()?;

        let color_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let depth_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let mut hdr_images = Vec::with_capacity(image_count);
        let mut depth_images = Vec::with_capacity(image_count);
        let mut velocity_images = Vec::with_capacity(image_count);
        let mut hdr_views = Vec::with_capacity(image_count);
        let mut velocity_views = Vec::with_capacity(image_count);
        let mut depth_views = Vec::with_capacity(image_count);

        for _ in 0..image_count {
            let hdr_image = context.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(HDR_FORMAT)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                "HBD HDR color",
            )?;
            let hdr_view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(hdr_image.image())
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(HDR_FORMAT)
                        .subresource_range(color_range),
                    None,
                )
            }
            .map_err(GfxError::Vulkan)?;

            let velocity_image = context.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(VELOCITY_FORMAT)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                "HBD velocity",
            )?;
            let velocity_view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(velocity_image.image())
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(VELOCITY_FORMAT)
                        .subresource_range(color_range),
                    None,
                )
            }
            .map_err(GfxError::Vulkan)?;

            let depth_image = context.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(depth_format)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                "HBD geometry depth",
            )?;
            let depth_view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(depth_image.image())
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(depth_format)
                        .subresource_range(depth_range),
                    None,
                )
            }
            .map_err(GfxError::Vulkan)?;

            hdr_images.push(hdr_image);
            depth_images.push(depth_image);
            velocity_images.push(velocity_image);
            hdr_views.push(hdr_view);
            velocity_views.push(velocity_view);
            depth_views.push(depth_view);
        }

        // Early failures would leave the images allocated but the local
        // vectors drop them, so their GpuImage destructors free the memory.
        let attachments = [
            vk::AttachmentDescription::default()
                .format(HDR_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentDescription::default()
                .format(VELOCITY_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentDescription::default()
                .format(depth_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];

        let color_reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let velocity_reference = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_reference = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let color_references = [color_reference, velocity_reference];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_references)
            .depth_stencil_attachment(&depth_reference)];
        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
                .dst_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                )
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
            // The shadow pass writes depth before this pass samples it.
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
        ];

        let render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(&attachments)
                    .subpasses(&subpasses)
                    .dependencies(&dependencies),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;

        let mut framebuffers = Vec::with_capacity(image_count);
        for index in 0..image_count {
            let framebuffer_attachments =
                [hdr_views[index], velocity_views[index], depth_views[index]];
            framebuffers.push(
                unsafe {
                    device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(&framebuffer_attachments)
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                }
                .map_err(GfxError::Vulkan)?,
            );
        }

        Ok(Self {
            device,
            _hdr_images: hdr_images,
            _depth_images: depth_images,
            _velocity_images: velocity_images,
            hdr_views,
            velocity_views,
            depth_views,
            render_pass,
            framebuffers,
        })
    }

    pub(crate) fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub(crate) fn hdr_views(&self) -> &[vk::ImageView] {
        &self.hdr_views
    }

    pub(crate) fn velocity_views(&self) -> &[vk::ImageView] {
        &self.velocity_views
    }

    pub(crate) fn depth_views(&self) -> &[vk::ImageView] {
        &self.depth_views
    }

    pub(crate) fn hdr_format(&self) -> vk::Format {
        HDR_FORMAT
    }

    pub(crate) fn framebuffer(&self, image_index: u32) -> vk::Framebuffer {
        self.framebuffers[image_index as usize]
    }

    pub(crate) fn clear_values(&self) -> [vk::ClearValue; 3] {
        [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ]
    }
}

impl Drop for RenderTargets {
    fn drop(&mut self) {
        unsafe {
            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            for view in &self.hdr_views {
                self.device.destroy_image_view(*view, None);
            }
            for view in &self.velocity_views {
                self.device.destroy_image_view(*view, None);
            }
            for view in &self.depth_views {
                self.device.destroy_image_view(*view, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
        }
    }
}
