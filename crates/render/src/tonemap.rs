//! Final presentation pass: samples the HDR target and tonemaps it into the
//! swapchain image. A fullscreen triangle avoids a vertex buffer.

use std::ffi::CStr;
use std::mem::size_of;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use hbd_gfx::{GfxError, GpuBuffer, MemoryLocation, VulkanContext};
use tracing::info;

use crate::{RenderError, shader};

const TONEMAP_WGSL: &str = include_str!("tonemap.wgsl");
const VERTEX_ENTRY: &CStr = c"vs_main";
const FRAGMENT_ENTRY: &CStr = c"fs_main";

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DebugParams {
    mode: u32,
    pad: [u32; 3],
}

pub(crate) struct TonemapPass {
    device: ash::Device,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    descriptor_sets: Vec<vk::DescriptorSet>,
    debug_buffer: GpuBuffer,
}

impl TonemapPass {
    pub(crate) fn new(
        context: &VulkanContext,
        hdr_views: &[vk::ImageView],
        bloom_views: &[vk::ImageView],
        ao_views: &[vk::ImageView],
        depth_views: &[vk::ImageView],
        shadow_views: &[vk::ImageView],
        history_views: &[vk::ImageView],
    ) -> Result<Self, RenderError> {
        let device = context.device().clone();

        let vertex_spv = shader::compile_wgsl(TONEMAP_WGSL, "vs_main", naga::ShaderStage::Vertex)?;
        let fragment_spv =
            shader::compile_wgsl(TONEMAP_WGSL, "fs_main", naga::ShaderStage::Fragment)?;
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
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
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
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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
            .render_pass(context.render_pass())
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

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(0.0),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;

        let debug_buffer = context.create_buffer(
            size_of::<DebugParams>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "HBD debug params",
        )?;
        debug_buffer
            .write(bytemuck::bytes_of(&DebugParams {
                mode: 0,
                pad: [0, 0, 0],
            }))
            .map_err(RenderError::Gfx)?;

        let set_count = hdr_views.len() as u32;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(set_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(set_count * 6),
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
        let set_layouts = vec![descriptor_set_layout; hdr_views.len()];
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(GfxError::Vulkan)?;

        assert!(!shadow_views.is_empty(), "a shadow cascade is required");
        let debug_shadow_view = shadow_views[0];
        for (
            (((((set, hdr_view), bloom_view), ao_view), depth_view), _shadow_view),
            history_view,
        ) in descriptor_sets
            .iter()
            .zip(hdr_views.iter().copied())
            .zip(bloom_views.iter().copied())
            .zip(ao_views.iter().copied())
            .zip(depth_views.iter().copied())
            .zip(shadow_views.iter().copied())
            .zip(history_views.iter().copied())
        {
            let sampler_infos = [vk::DescriptorImageInfo::default().sampler(sampler)];
            let image_infos = [vk::DescriptorImageInfo::default()
                .image_view(hdr_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let bloom_infos = [vk::DescriptorImageInfo::default()
                .image_view(bloom_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let ao_infos = [vk::DescriptorImageInfo::default()
                .image_view(ao_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let debug_infos = [vk::DescriptorBufferInfo::default()
                .buffer(debug_buffer.buffer())
                .offset(0)
                .range(debug_buffer.size())];
            let depth_infos = [vk::DescriptorImageInfo::default()
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let shadow_infos = [vk::DescriptorImageInfo::default()
                .image_view(debug_shadow_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let history_infos = [vk::DescriptorImageInfo::default()
                .image_view(history_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
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
                    .image_info(&image_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&bloom_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&ao_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&debug_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&depth_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&shadow_infos),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&history_infos),
            ];
            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }
        }

        info!(sets = set_count, "HDR tonemap pass ready");

        Ok(Self {
            device,
            pipeline,
            layout,
            descriptor_set_layout,
            descriptor_pool,
            sampler,
            descriptor_sets,
            debug_buffer,
        })
    }

    pub(crate) fn set_mode(&self, mode: u32) -> Result<(), RenderError> {
        self.debug_buffer
            .write(bytemuck::bytes_of(&DebugParams {
                mode,
                pad: [0, 0, 0],
            }))
            .map_err(RenderError::Gfx)
    }

    pub(crate) fn record(
        &self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        image_index: u32,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            let viewport = vk::Viewport::default()
                .x(0.0)
                .y(0.0)
                .width(extent.width as f32)
                .height(extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            self.device
                .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

            let scissor = vk::Rect2D::default().extent(extent);
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
        }
    }
}

impl Drop for TonemapPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_sampler(self.sampler, None);
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
