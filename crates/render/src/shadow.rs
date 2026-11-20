//! Three-cascade directional shadow map. Depth-only passes render the scene
//! from the light for three nested orthographic extents; the mesh pass selects
//! a cascade by camera distance and samples it with PCF.

use std::ffi::CStr;
use std::mem::{offset_of, size_of};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use hbd_gfx::{GfxError, GpuBuffer, GpuImage, MemoryLocation, VulkanContext};
use tracing::info;

use crate::{DirectionalLight, RenderError, mesh, shader};

const SHADOW_WGSL: &str = include_str!("shadow.wgsl");
const VERTEX_ENTRY: &CStr = c"vs_main";
const FRAGMENT_ENTRY: &CStr = c"fs_main";
const SHADOW_SIZE: u32 = 1024;
const CASCADE_EXTENTS: [f32; 3] = [12.0, 32.0, 80.0];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowFrame {
    light_view_projection: [[f32; 4]; 4],
}

pub(crate) struct ShadowPass {
    device: ash::Device,
    // Kept alive for the image views and framebuffers.
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
    // One uniform buffer per cascade, bound through its own descriptor set.
    _uniform_buffers: Vec<GpuBuffer>,
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    _instance_buffer: GpuBuffer,
    index_count: u32,
    instance_count: u32,
}

impl ShadowPass {
    pub(crate) fn new(
        context: &VulkanContext,
        light: &DirectionalLight,
    ) -> Result<Self, RenderError> {
        let device = context.device().clone();
        let depth_format = context.pick_depth_format()?;
        let cascade_count = CASCADE_EXTENTS.len();

        let mut images = Vec::with_capacity(cascade_count);
        let mut views = Vec::with_capacity(cascade_count);
        let mut framebuffers = Vec::with_capacity(cascade_count);

        for _ in 0..cascade_count {
            let image = context.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(depth_format)
                    .extent(vk::Extent3D {
                        width: SHADOW_SIZE,
                        height: SHADOW_SIZE,
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
                "HBD shadow cascade",
            )?;
            let view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image.image())
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(depth_format)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::DEPTH,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
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
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .compare_enable(true)
                    .compare_op(vk::CompareOp::LESS_OR_EQUAL),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;

        let attachments = [vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let depth_reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&depth_reference)];
        let dependencies = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS)
            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)];
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

        for view in &views {
            framebuffers.push(
                unsafe {
                    device.create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(std::slice::from_ref(view))
                            .width(SHADOW_SIZE)
                            .height(SHADOW_SIZE)
                            .layers(1),
                        None,
                    )
                }
                .map_err(GfxError::Vulkan)?,
            );
        }

        let vertex_spv = shader::compile_wgsl(SHADOW_WGSL, "vs_main", naga::ShaderStage::Vertex)?;
        let fragment_spv =
            shader::compile_wgsl(SHADOW_WGSL, "fs_main", naga::ShaderStage::Fragment)?;
        let vertex_module = create_shader_module(&device, &vertex_spv)?;
        let fragment_module = create_shader_module(&device, &fragment_spv)?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
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

        let binding_description = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<mesh::Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
        let attribute_descriptions = [vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(mesh::Vertex, position) as u32)];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .depth_bias_enable(true)
            .depth_bias_constant_factor(1.5)
            .depth_bias_slope_factor(2.0)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);
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
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
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

        let (vertices, indices) = mesh::cube_mesh();
        let instances = mesh::build_scene();
        let vertex_buffer = context.upload_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            bytemuck::cast_slice(&vertices),
            "HBD shadow vertex buffer",
        )?;
        let index_buffer = context.upload_buffer(
            vk::BufferUsageFlags::INDEX_BUFFER,
            bytemuck::cast_slice(&indices),
            "HBD shadow index buffer",
        )?;
        let instance_buffer = context.upload_buffer(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            bytemuck::cast_slice(&instances),
            "HBD shadow instance buffer",
        )?;

        let mut uniform_buffers = Vec::with_capacity(cascade_count);
        for extent in CASCADE_EXTENTS {
            let buffer = context.create_buffer(
                size_of::<ShadowFrame>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                "HBD shadow cascade uniform",
            )?;
            buffer
                .write(bytemuck::bytes_of(&ShadowFrame {
                    light_view_projection: light.view_projection(extent).to_cols_array_2d(),
                }))
                .map_err(RenderError::Gfx)?;
            uniform_buffers.push(buffer);
        }

        let set_count = cascade_count as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(set_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
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
        let set_layouts = vec![descriptor_set_layout; cascade_count];
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(GfxError::Vulkan)?;

        let instance_info = [vk::DescriptorBufferInfo::default()
            .buffer(instance_buffer.buffer())
            .offset(0)
            .range(instance_buffer.size())];
        for (set, uniform) in descriptor_sets.iter().zip(uniform_buffers.iter()) {
            let uniform_info = [vk::DescriptorBufferInfo::default()
                .buffer(uniform.buffer())
                .offset(0)
                .range(uniform.size())];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&uniform_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&instance_info),
            ];
            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }
        }

        info!(
            cascades = cascade_count,
            size = SHADOW_SIZE,
            "cascaded directional shadow map ready"
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
            _uniform_buffers: uniform_buffers,
            vertex_buffer,
            index_buffer,
            _instance_buffer: instance_buffer,
            index_count: indices.len() as u32,
            instance_count: instances.len() as u32,
        })
    }

    pub(crate) fn views(&self) -> &[vk::ImageView] {
        &self.views
    }

    pub(crate) fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    pub(crate) fn record(&self, command_buffer: vk::CommandBuffer) {
        let clear_values = [vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        }];

        for (index, framebuffer) in self.framebuffers.iter().enumerate() {
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(*framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: SHADOW_SIZE,
                        height: SHADOW_SIZE,
                    },
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
                    .width(SHADOW_SIZE as f32)
                    .height(SHADOW_SIZE as f32)
                    .min_depth(0.0)
                    .max_depth(1.0);
                self.device
                    .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));
                let scissor = vk::Rect2D::default().extent(vk::Extent2D {
                    width: SHADOW_SIZE,
                    height: SHADOW_SIZE,
                });
                self.device
                    .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    std::slice::from_ref(&self.descriptor_sets[index]),
                    &[],
                );
                self.device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    std::slice::from_ref(&self.vertex_buffer.buffer()),
                    &[0],
                );
                self.device.cmd_bind_index_buffer(
                    command_buffer,
                    self.index_buffer.buffer(),
                    0,
                    vk::IndexType::UINT32,
                );
                self.device.cmd_draw_indexed(
                    command_buffer,
                    self.index_count,
                    self.instance_count,
                    0,
                    0,
                    0,
                );
                self.device.cmd_end_render_pass(command_buffer);
            }
        }
    }
}

impl Drop for ShadowPass {
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
