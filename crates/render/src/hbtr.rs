//! HBD Temporal Resolve (HBTR).
//!
//! Jitters the camera projection each frame, then resolves the current HDR
//! sample against a reprojected history buffer with neighborhood clamping.
//! History is ping-ponged per swapchain image.

use std::ffi::CStr;
use std::mem::size_of;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use hbd_core::math::Mat4;
use hbd_gfx::{GfxError, GpuBuffer, GpuImage, MemoryLocation, VulkanContext};

use crate::RenderError;
use crate::render_targets::HDR_FORMAT;
use crate::shader;

const RESOLVE_WGSL: &str = include_str!("taa_resolve.wgsl");
const VERTEX_ENTRY: &CStr = c"vs_main";
const FRAGMENT_ENTRY: &CStr = c"fs_main";

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TaaParams {
    inverse_view_projection: [[f32; 4]; 4],
    previous_view_projection: [[f32; 4]; 4],
    settings: [f32; 4],
}

pub(crate) struct TaaPass {
    device: ash::Device,
    _resolved_images: Vec<GpuImage>,
    _history_images: Vec<GpuImage>,
    resolved_views: Vec<vk::ImageView>,
    history_views: Vec<Vec<vk::ImageView>>,
    /// Per image, per history parity: attachment 0 is the stable presented
    /// image, attachment 1 is that parity's history slot. Both are written by
    /// the resolve draw, so the old full-screen copy pass is gone.
    resolve_framebuffers: Vec<Vec<vk::Framebuffer>>,
    sampler: vk::Sampler,
    render_pass: vk::RenderPass,
    resolve_pipeline: vk::Pipeline,
    resolve_layout: vk::PipelineLayout,
    resolve_set_layout: vk::DescriptorSetLayout,
    resolve_pool: vk::DescriptorPool,
    resolve_sets: Vec<Vec<vk::DescriptorSet>>,
    uniform_buffers: Vec<GpuBuffer>,
    /// Reprojection pose for each swapchain image's own history pair. An image
    /// is only revisited every `image_count` frames, so one shared matrix would
    /// reproject against a completely different camera pose.
    previous_view_projections: Vec<Mat4>,
    initialized: Vec<bool>,
    read_history: Vec<usize>,
    read_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
}

impl TaaPass {
    pub(crate) fn new(
        context: &VulkanContext,
        hdr_views: &[vk::ImageView],
        depth_views: &[vk::ImageView],
        velocity_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<Self, RenderError> {
        let device = context.device().clone();
        let image_count = hdr_views.len();

        let subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let create_target = |name: &str| -> Result<(GpuImage, vk::ImageView), RenderError> {
            let image = context.create_image(
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
                name,
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
            Ok((image, view))
        };

        let mut resolved_images = Vec::with_capacity(image_count);
        let mut resolved_views = Vec::with_capacity(image_count);
        let mut history_images = Vec::with_capacity(image_count * 2);
        let mut history_views = vec![Vec::with_capacity(2); image_count];

        for history_slot in history_views.iter_mut() {
            let (image, view) = create_target("HBD TAA resolved")?;
            resolved_images.push(image);
            resolved_views.push(view);
            for _ in 0..2 {
                let (image, view) = create_target("HBD TAA history")?;
                history_images.push(image);
                history_slot.push(view);
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

        let attachment = || {
            vk::AttachmentDescription::default()
                .format(HDR_FORMAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        };
        let attachments = [attachment(), attachment()];
        let color_references = [
            vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(1)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        ];
        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_references)];
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
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
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

        let make_resolve_framebuffer = |presented: vk::ImageView, history: vk::ImageView| {
            let attachments = [presented, history];
            unsafe {
                device
                    .create_framebuffer(
                        &vk::FramebufferCreateInfo::default()
                            .render_pass(render_pass)
                            .attachments(&attachments)
                            .width(extent.width)
                            .height(extent.height)
                            .layers(1),
                        None,
                    )
                    .map_err(|error| RenderError::Gfx(GfxError::Vulkan(error)))
            }
        };

        let mut resolve_framebuffers = vec![Vec::with_capacity(2); image_count];
        for index in 0..image_count {
            for history in history_views[index].iter().take(2) {
                resolve_framebuffers[index]
                    .push(make_resolve_framebuffer(resolved_views[index], *history)?);
            }
        }

        let resolve_vertex_spv =
            shader::compile_wgsl(RESOLVE_WGSL, "vs_main", naga::ShaderStage::Vertex)?;
        let resolve_fragment_spv =
            shader::compile_wgsl(RESOLVE_WGSL, "fs_main", naga::ShaderStage::Fragment)?;
        let resolve_vertex_module = create_shader_module(&device, &resolve_vertex_spv)?;
        let resolve_fragment_module = create_shader_module(&device, &resolve_fragment_spv)?;

        let resolve_bindings = [
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
        ];
        let resolve_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&resolve_bindings),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let resolve_layout = create_pipeline_layout(&device, resolve_set_layout)?;
        let resolve_pipeline = create_fullscreen_pipeline(
            &device,
            resolve_layout,
            render_pass,
            resolve_vertex_module,
            resolve_fragment_module,
        )?;

        unsafe {
            device.destroy_shader_module(resolve_vertex_module, None);
            device.destroy_shader_module(resolve_fragment_module, None);
        }

        let mut uniform_buffers = Vec::with_capacity(image_count);
        for _ in 0..image_count {
            uniform_buffers.push(context.create_buffer(
                size_of::<TaaParams>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                "HBD TAA params",
            )?);
        }

        let resolve_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count((image_count * 2) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count((image_count * 8) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count((image_count * 2) as u32),
        ];
        let resolve_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&resolve_pool_sizes)
                    .max_sets((image_count * 2) as u32),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let resolve_set_layouts = vec![resolve_set_layout; image_count * 2];
        let all_resolve_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(resolve_pool)
                    .set_layouts(&resolve_set_layouts),
            )
        }
        .map_err(GfxError::Vulkan)?;

        let sampler_infos = [vk::DescriptorImageInfo::default().sampler(sampler)];
        let mut resolve_sets = vec![Vec::with_capacity(2); image_count];
        for index in 0..image_count {
            for parity in 0..2 {
                let set = all_resolve_sets[index * 2 + parity];
                let current_infos = [vk::DescriptorImageInfo::default()
                    .image_view(hdr_views[index])
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let history_infos = [vk::DescriptorImageInfo::default()
                    .image_view(history_views[index][parity])
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let depth_infos = [vk::DescriptorImageInfo::default()
                    .image_view(depth_views[index])
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let velocity_infos = [vk::DescriptorImageInfo::default()
                    .image_view(velocity_views[index])
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let uniform_infos = [vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffers[index].buffer())
                    .offset(0)
                    .range(uniform_buffers[index].size())];
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLER)
                        .image_info(&sampler_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&current_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&history_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(3)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&depth_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(4)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&uniform_infos),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(5)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&velocity_infos),
                ];
                unsafe {
                    device.update_descriptor_sets(&writes, &[]);
                }
                resolve_sets[index].push(set);
            }
        }

        tracing::info!(
            width = extent.width,
            height = extent.height,
            "TAA pass ready"
        );

        let read_views = history_views.iter().map(|views| views[0]).collect();

        Ok(Self {
            device,
            _resolved_images: resolved_images,
            _history_images: history_images,
            resolved_views,
            history_views,
            resolve_framebuffers,
            sampler,
            render_pass,
            resolve_pipeline,
            resolve_layout,
            resolve_set_layout,
            resolve_pool,
            resolve_sets,
            uniform_buffers,
            previous_view_projections: vec![Mat4::IDENTITY; image_count],
            initialized: vec![false; image_count],
            read_history: vec![0; image_count],
            read_views,
            extent,
        })
    }

    pub(crate) fn resolved_views(&self) -> &[vk::ImageView] {
        &self.resolved_views
    }

    pub(crate) fn history_views(&self) -> &[vk::ImageView] {
        &self.read_views
    }

    /// Uploads the resolve parameters for the acquired swapchain image after
    /// its in-flight fence has been waited on. The history pose belongs to this
    /// image alone; see `end_frame_for_index`.
    pub(crate) fn begin_frame(
        &self,
        index: u32,
        current_view_projection: Mat4,
    ) -> Result<(), RenderError> {
        let index = index as usize;
        let inverse = current_view_projection.inverse();
        let valid = self.initialized[index];
        let blend = if valid { 0.9 } else { 0.0 };
        let params = TaaParams {
            inverse_view_projection: inverse.to_cols_array_2d(),
            previous_view_projection: self.previous_view_projections[index].to_cols_array_2d(),
            settings: [
                blend,
                1.0 / self.extent.width as f32,
                1.0 / self.extent.height as f32,
                // The resolve shader must not sample a freshly allocated (or
                // post-resize) history image: its contents are undefined and a
                // single NaN poisons the whole weighted blend into black.
                if valid { 1.0 } else { 0.0 },
            ],
        };
        self.uniform_buffers[index]
            .write(bytemuck::bytes_of(&params))
            .map_err(RenderError::Gfx)
    }

    /// Records the pose that produced this image's newest history slot. It must
    /// be per image: with three swapchain images the next visit to this slot is
    /// three frames later, not one.
    pub(crate) fn end_frame_for_index(&mut self, current_view_projection: Mat4, index: u32) {
        let index = index as usize;
        self.previous_view_projections[index] = current_view_projection;
        self.initialized[index] = true;
    }

    pub(crate) fn record(&mut self, command_buffer: vk::CommandBuffer, index: u32) {
        let index = index as usize;
        let read = self.read_history[index];
        let write = 1 - read;

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let clear_values = [clear_color, clear_color];
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };

        unsafe {
            // One draw writes both the presented image and the new history
            // slot. The old full-screen copy pass is gone; same pixels, fewer
            // passes and less bandwidth.
            let resolve_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.resolve_framebuffers[index][write])
                .render_area(render_area)
                .clear_values(&clear_values);
            self.device.cmd_begin_render_pass(
                command_buffer,
                &resolve_info,
                vk::SubpassContents::INLINE,
            );
            self.draw(
                command_buffer,
                self.resolve_pipeline,
                self.resolve_layout,
                self.resolve_sets[index][read],
            );
            self.device.cmd_end_render_pass(command_buffer);
        }

        self.read_history[index] = write;
        self.read_views[index] = self.history_views[index][write];
    }

    fn draw(
        &self,
        command_buffer: vk::CommandBuffer,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        descriptor_set: vk::DescriptorSet,
    ) {
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            let viewport = vk::Viewport::default()
                .width(self.extent.width as f32)
                .height(self.extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            self.device
                .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));
            let scissor = vk::Rect2D::default().extent(self.extent);
            self.device
                .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
    }
}

impl Drop for TaaPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.resolve_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.resolve_layout, None);
            self.device.destroy_descriptor_pool(self.resolve_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.resolve_set_layout, None);
            for framebuffers in &self.resolve_framebuffers {
                for framebuffer in framebuffers {
                    self.device.destroy_framebuffer(*framebuffer, None);
                }
            }
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_sampler(self.sampler, None);
            for view in &self.resolved_views {
                self.device.destroy_image_view(*view, None);
            }
            for views in &self.history_views {
                for view in views {
                    self.device.destroy_image_view(*view, None);
                }
            }
        }
    }
}

fn create_pipeline_layout(
    device: &ash::Device,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, RenderError> {
    unsafe {
        device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&layout)),
                None,
            )
            .map_err(GfxError::Vulkan)
    }
    .map_err(RenderError::Gfx)
}

fn create_fullscreen_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
) -> Result<vk::Pipeline, RenderError> {
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
    let color_blend_attachments = [color_blend_attachment, color_blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
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
            .create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&info), None)
            .map_err(|(_, error)| RenderError::Pipeline(error))?
            .into_iter()
            .next()
            .ok_or(RenderError::MissingPipeline)
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
