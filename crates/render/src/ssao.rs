//! Half-resolution screen-space ambient occlusion.
//!
//! Reconstructs view-space positions from the geometry depth buffer, builds a
//! normal from screen-space derivatives, and accumulates a 16-tap hemisphere.

use std::ffi::CStr;
use std::mem::size_of;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use hbd_gfx::{GfxError, GpuBuffer, GpuImage, MemoryLocation, VulkanContext};

use crate::{Camera, RenderError, shader};

const SSAO_WGSL: &str = include_str!("ssao.wgsl");
const VERTEX_ENTRY: &CStr = c"vs_main";
const FRAGMENT_ENTRY: &CStr = c"fs_main";
const AO_FORMAT: vk::Format = vk::Format::R8_UNORM;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsaoParams {
    projection: [[f32; 4]; 4],
    inverse_projection: [[f32; 4]; 4],
    settings: [f32; 4],
}

pub(crate) struct SsaoPass {
    device: ash::Device,
    _images: Vec<GpuImage>,
    views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    sampler: vk::Sampler,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_buffers: Vec<GpuBuffer>,
    half_extent: vk::Extent2D,
}

impl SsaoPass {
    pub(crate) fn new(
        context: &VulkanContext,
        depth_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<Self, RenderError> {
        let device = context.device().clone();
        let half_extent = vk::Extent2D {
            width: (extent.width / 2).max(1),
            height: (extent.height / 2).max(1),
        };
        let image_count = depth_views.len();

        let subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let mut images = Vec::with_capacity(image_count);
        let mut views = Vec::with_capacity(image_count);
        for _ in 0..image_count {
            let image = context.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(AO_FORMAT)
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
                "HBD SSAO target",
            )?;
            let view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image.image())
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(AO_FORMAT)
                        .subresource_range(subresource),
                    None,
                )
            }
            .map_err(GfxError::Vulkan)?;
            images.push(image);
            views.push(view);
        }

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;

        let attachments = [vk::AttachmentDescription::default()
            .format(AO_FORMAT)
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
            // Depth was written by the geometry pass and is sampled here.
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
        for view in &views {
            framebuffers.push(
                unsafe {
                    device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(std::slice::from_ref(view))
                            .width(half_extent.width)
                            .height(half_extent.height)
                            .layers(1),
                        None,
                    )
                }
                .map_err(GfxError::Vulkan)?,
            );
        }

        let vertex_spv = shader::compile_wgsl(SSAO_WGSL, "vs_main", naga::ShaderStage::Vertex)?;
        let fragment_spv = shader::compile_wgsl(SSAO_WGSL, "fs_main", naga::ShaderStage::Fragment)?;
        let vertex_module = create_shader_module(&device, &vertex_spv)?;
        let fragment_module = create_shader_module(&device, &fragment_spv)?;

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
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vertex_module)
                .name(VERTEX_ENTRY),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fragment_module)
                .name(FRAGMENT_ENTRY),
        ];
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
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
        let pipeline = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|(_, error)| RenderError::Pipeline(error))?
        .into_iter()
        .next()
        .ok_or(RenderError::MissingPipeline)?;
        unsafe {
            device.destroy_shader_module(vertex_module, None);
            device.destroy_shader_module(fragment_module, None);
        }

        let mut uniform_buffers = Vec::with_capacity(image_count);
        for _ in 0..image_count {
            uniform_buffers.push(context.create_buffer(
                size_of::<SsaoParams>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                "HBD SSAO params",
            )?);
        }

        let set_count = image_count as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(set_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(set_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(set_count),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(set_count),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let set_layouts = vec![descriptor_set_layout; image_count];
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(GfxError::Vulkan)?;

        let sampler_infos = [vk::DescriptorImageInfo::default().sampler(sampler)];
        for ((set, depth_view), uniform) in descriptor_sets
            .iter()
            .zip(depth_views.iter().copied())
            .zip(uniform_buffers.iter())
        {
            let depth_infos = [vk::DescriptorImageInfo::default()
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let uniform_infos = [vk::DescriptorBufferInfo::default()
                .buffer(uniform.buffer())
                .offset(0)
                .range(uniform.size())];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&depth_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&uniform_infos),
            ];
            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }
        }

        tracing::info!(
            width = half_extent.width,
            height = half_extent.height,
            "SSAO pass ready"
        );

        Ok(Self {
            device,
            _images: images,
            views,
            framebuffers,
            sampler,
            render_pass,
            pipeline,
            layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            uniform_buffers,
            half_extent,
        })
    }

    pub(crate) fn views(&self) -> &[vk::ImageView] {
        &self.views
    }

    pub(crate) fn update_all(&self, camera: &Camera, aspect: f32) -> Result<(), RenderError> {
        for index in 0..self.uniform_buffers.len() as u32 {
            self.update(camera, aspect, index)?;
        }
        Ok(())
    }

    pub(crate) fn update(
        &self,
        camera: &Camera,
        aspect: f32,
        index: u32,
    ) -> Result<(), RenderError> {
        let projection = camera.projection(aspect);
        let params = SsaoParams {
            projection: projection.to_cols_array_2d(),
            inverse_projection: projection.inverse().to_cols_array_2d(),
            settings: [1.4, 0.02, 1.4, 0.0],
        };
        self.uniform_buffers[index as usize]
            .write(bytemuck::bytes_of(&params))
            .map_err(RenderError::Gfx)
    }

    pub(crate) fn record(&self, command_buffer: vk::CommandBuffer, image_index: u32) {
        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [1.0, 1.0, 1.0, 1.0],
            },
        }];
        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.half_extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
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
                std::slice::from_ref(&self.descriptor_sets[image_index as usize]),
                &[],
            );
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(command_buffer);
        }
    }
}

impl Drop for SsaoPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_sampler(self.sampler, None);
            for view in &self.views {
                self.device.destroy_image_view(*view, None);
            }
        }
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
