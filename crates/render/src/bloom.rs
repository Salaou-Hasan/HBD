//! Two-pass bloom: bright extraction + horizontal blur, then vertical blur.
//!
//! One ping-pong pair of half-resolution HDR images is kept per swapchain
//! image so overlapping frames never write a target another frame samples.

use std::ffi::CStr;
use std::mem::size_of;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use hbd_gfx::{GfxError, GpuBuffer, GpuImage, VulkanContext};

use crate::{RenderError, render_targets::HDR_FORMAT, shader};

const BLOOM_WGSL: &str = include_str!("bloom.wgsl");
const VERTEX_ENTRY: &CStr = c"vs_main";
const HORIZONTAL_ENTRY: &CStr = c"fs_horizontal";
const VERTICAL_ENTRY: &CStr = c"fs_vertical";

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomParams {
    data: [f32; 4],
}

pub(crate) struct BloomPass {
    device: ash::Device,
    // Kept alive for the image views and framebuffers.
    _images: Vec<GpuImage>,
    a_views: Vec<vk::ImageView>,
    b_views: Vec<vk::ImageView>,
    sampler: vk::Sampler,
    render_pass: vk::RenderPass,
    framebuffers_a: Vec<vk::Framebuffer>,
    framebuffers_b: Vec<vk::Framebuffer>,
    pipeline_horizontal: vk::Pipeline,
    pipeline_vertical: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    sets_hdr: Vec<vk::DescriptorSet>,
    sets_a: Vec<vk::DescriptorSet>,
    _params_buffer: GpuBuffer,
    half_extent: vk::Extent2D,
}

impl BloomPass {
    pub(crate) fn new(
        context: &VulkanContext,
        hdr_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<Self, RenderError> {
        let device = context.device().clone();
        let half_extent = vk::Extent2D {
            width: (extent.width / 2).max(1),
            height: (extent.height / 2).max(1),
        };
        let image_count = hdr_views.len();

        let subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let mut images = Vec::with_capacity(image_count * 2);
        let mut a_views = Vec::with_capacity(image_count);
        let mut b_views = Vec::with_capacity(image_count);

        for _ in 0..image_count {
            for _ in 0..2 {
                let image = context.create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(HDR_FORMAT)
                        .extent(vk::Extent3D {
                            width: half_extent.width,
                            height: half_extent.height,
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    "HBD bloom target",
                )?;
                let view = unsafe {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image.image())
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(HDR_FORMAT)
                            .subresource_range(subresource),
                        None,
                    )
                }
                .map_err(GfxError::Vulkan)?;
                images.push(image);
                if images.len() % 2 == 1 {
                    a_views.push(view);
                } else {
                    b_views.push(view);
                }
            }
        }

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;

        let params_buffer = context.create_buffer(
            size_of::<BloomParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            hbd_gfx::MemoryLocation::CpuToGpu,
            "HBD bloom params",
        )?;
        params_buffer
            .write(bytemuck::bytes_of(&BloomParams {
                data: [
                    1.0 / half_extent.width as f32,
                    1.0 / half_extent.height as f32,
                    1.0, // luminance threshold
                    0.0,
                ],
            }))
            .map_err(RenderError::Gfx)?;

        let attachments = [vk::AttachmentDescription::default()
            .format(HDR_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let color_reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_reference))];
        let dependencies = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
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

        let mut framebuffers_a = Vec::with_capacity(image_count);
        let mut framebuffers_b = Vec::with_capacity(image_count);
        for index in 0..image_count {
            for (views, target) in [
                (&a_views, &mut framebuffers_a),
                (&b_views, &mut framebuffers_b),
            ] {
                let framebuffer_attachments = [views[index]];
                target.push(
                    unsafe {
                        device.create_framebuffer(
                            &vk::FramebufferCreateInfo::default()
                                .render_pass(render_pass)
                                .attachments(&framebuffer_attachments)
                                .width(half_extent.width)
                                .height(half_extent.height)
                                .layers(1),
                            None,
                        )
                    }
                    .map_err(GfxError::Vulkan)?,
                );
            }
        }

        let vertex_spv = shader::compile_wgsl(
            &shader::source("bloom.wgsl", BLOOM_WGSL),
            "vs_main",
            naga::ShaderStage::Vertex,
        )?;
        let horizontal_spv = shader::compile_wgsl(
            &shader::source("bloom.wgsl", BLOOM_WGSL),
            "fs_horizontal",
            naga::ShaderStage::Fragment,
        )?;
        let vertical_spv = shader::compile_wgsl(
            &shader::source("bloom.wgsl", BLOOM_WGSL),
            "fs_vertical",
            naga::ShaderStage::Fragment,
        )?;
        let vertex_module = create_shader_module(&device, &vertex_spv)?;
        let horizontal_module = create_shader_module(&device, &horizontal_spv)?;
        let vertical_module = create_shader_module(&device, &vertical_spv)?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let set_layouts = [descriptor_set_layout];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let make_pipeline =
            |fragment_module, fragment_entry| -> Result<vk::Pipeline, RenderError> {
                let stages = [
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(vertex_module)
                        .name(VERTEX_ENTRY),
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(fragment_module)
                        .name(fragment_entry),
                ];
                let info = vk::GraphicsPipelineCreateInfo::default()
                    .stages(&stages)
                    .input_assembly_state(&input_assembly)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization)
                    .multisample_state(&multisample)
                    .color_blend_state(&color_blend)
                    .dynamic_state(&dynamic_state)
                    .layout(layout)
                    .render_pass(render_pass)
                    .subpass(0);
                unsafe {
                    device
                        .create_graphics_pipelines(
                            vk::PipelineCache::null(),
                            std::slice::from_ref(&info),
                            None,
                        )
                        .map_err(|(_, error)| RenderError::Pipeline(error))?
                        .into_iter()
                        .next()
                        .ok_or(RenderError::MissingPipeline)
                }
            };
        let pipeline_horizontal = make_pipeline(horizontal_module, HORIZONTAL_ENTRY)?;
        let pipeline_vertical = make_pipeline(vertical_module, VERTICAL_ENTRY)?;

        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(horizontal_module, None);
            device.destroy_shader_module(vertical_module, None);
        }

        let total_sets = (image_count * 2) as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(total_sets),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(total_sets),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(total_sets),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(total_sets),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let set_layouts = vec![descriptor_set_layout; image_count * 2];
        let all_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(GfxError::Vulkan)?;
        let (sets_hdr, sets_a) = all_sets.split_at(image_count);

        let sampler_infos = [vk::DescriptorImageInfo::default().sampler(sampler)];
        let params_infos = [vk::DescriptorBufferInfo::default()
            .buffer(params_buffer.buffer())
            .offset(0)
            .range(params_buffer.size())];
        for (index, view) in hdr_views.iter().copied().enumerate() {
            write_set(
                &device,
                sets_hdr[index],
                view,
                &sampler_infos,
                &params_infos,
            );
        }
        for (index, view) in a_views.iter().copied().enumerate() {
            write_set(&device, sets_a[index], view, &sampler_infos, &params_infos);
        }

        tracing::info!(
            width = half_extent.width,
            height = half_extent.height,
            "bloom pass ready"
        );

        Ok(Self {
            device,
            _images: images,
            a_views,
            b_views,
            sampler,
            render_pass,
            framebuffers_a,
            framebuffers_b,
            pipeline_horizontal,
            pipeline_vertical,
            layout,
            descriptor_set_layout,
            descriptor_pool,
            sets_hdr: sets_hdr.to_vec(),
            sets_a: sets_a.to_vec(),
            _params_buffer: params_buffer,
            half_extent,
        })
    }

    pub(crate) fn b_views(&self) -> &[vk::ImageView] {
        &self.b_views
    }

    pub(crate) fn record(&self, command_buffer: vk::CommandBuffer, image_index: u32) {
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let index = image_index as usize;

        unsafe {
            let pass_a = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers_a[index])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.half_extent,
                })
                .clear_values(&clear_values);
            self.device
                .cmd_begin_render_pass(command_buffer, &pass_a, vk::SubpassContents::INLINE);
            self.draw_blur(
                command_buffer,
                self.pipeline_horizontal,
                self.sets_hdr[index],
            );
            self.device.cmd_end_render_pass(command_buffer);

            let pass_b = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers_b[index])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.half_extent,
                })
                .clear_values(&clear_values);
            self.device
                .cmd_begin_render_pass(command_buffer, &pass_b, vk::SubpassContents::INLINE);
            self.draw_blur(command_buffer, self.pipeline_vertical, self.sets_a[index]);
            self.device.cmd_end_render_pass(command_buffer);
        }
    }

    fn draw_blur(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        descriptor_set: vk::DescriptorSet,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            let viewport = vk::Viewport::default()
                .width(self.half_extent.width as f32)
                .height(self.half_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            self.device
                .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));
            let scissor = vk::Rect2D::default().extent(self.half_extent);
            self.device
                .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
    }
}

impl Drop for BloomPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline_horizontal, None);
            self.device.destroy_pipeline(self.pipeline_vertical, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for framebuffer in &self.framebuffers_a {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            for framebuffer in &self.framebuffers_b {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            for view in &self.a_views {
                self.device.destroy_image_view(*view, None);
            }
            for view in &self.b_views {
                self.device.destroy_image_view(*view, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

fn write_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    sampler_infos: &[vk::DescriptorImageInfo],
    params_infos: &[vk::DescriptorBufferInfo],
) {
    let image_infos = [vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(sampler_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&image_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(params_infos),
    ];
    unsafe {
        device.update_descriptor_sets(&writes, &[]);
    }
}

fn create_shader_module(
    device: &ash::Device,
    spirv: &[u32],
) -> Result<vk::ShaderModule, RenderError> {
    unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
            .map_err(GfxError::Vulkan)
    }
    .map_err(RenderError::Gfx)
}
