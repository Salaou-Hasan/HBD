//! First real 3D pass: an instanced, indexed cube scene with depth testing.
//!
//! This replaces the M0 bootstrap triangle. Geometry is one shared cube mesh;
//! per-instance transforms and colors live in a read-only storage buffer. The
//! frame uniform carries the camera and light. Materials, instancing culling,
//! and the deferred G-buffer arrive in later milestones.

use std::ffi::CStr;
use std::mem::{offset_of, size_of};
use std::path::Path;

use ash::vk;
use bytemuck::{Pod, Zeroable};
use hbd_core::math::{Mat4, Vec2, Vec3};
use hbd_gfx::{GfxError, GpuBuffer, GpuImage, MemoryLocation, VulkanContext};
use tracing::info;

use crate::{Camera, DirectionalLight, RenderError, shader};

const MESH_WGSL: &str = include_str!("mesh.wgsl");
const SKY_WGSL: &str = include_str!("sky.wgsl");
const MESH_GENERATE_WGSL: &str = include_str!("mesh_generate.wgsl");
const CULL_WGSL: &str = include_str!("cull.wgsl");
const CLUSTER_WGSL: &str = include_str!("cluster.wgsl");
const HZB_WGSL: &str = include_str!("hzb.wgsl");
const HZB_MIPS: u32 = 3;
const VERTEX_ENTRY: &CStr = c"vs_main";
const FRAGMENT_ENTRY: &CStr = c"fs_main";
const GRID_HALF_EXTENT: i32 = 4;
const GRID_SPACING: f32 = 1.25;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_projection: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    light_vp0: [[f32; 4]; 4],
    light_vp1: [[f32; 4]; 4],
    light_vp2: [[f32; 4]; 4],
    eye_position: [f32; 4],
    light_direction: [f32; 4],
    light_color: [f32; 4],
    shadow_splits: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Instance {
    model: [[f32; 4]; 4],
    base_color: [f32; 4],
    /// x: metallic, y: roughness, z: ambient occlusion, w: unused.
    material: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniform {
    planes: [[f32; 4]; 6],
    count: u32,
    cluster_count: u32,
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Occlusion {
    view_projection: [[f32; 4]; 4],
    proj_scale: [f32; 2],
    screen: [f32; 2],
    mip_count: u32,
    enabled: u32,
    pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HzbLevel {
    src_mip: u32,
    src_width: u32,
    src_height: u32,
    pad: u32,
}

pub(crate) struct MeshPass {
    device: ash::Device,
    pipeline: vk::Pipeline,
    sky_pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    /// Per-swapchain-image graphics sets: each binds that image's compacted
    /// instance buffer. The extra set binds the source buffer for `--no-cull`.
    graphics_sets: Vec<vk::DescriptorSet>,
    graphics_set_no_cull: vk::DescriptorSet,
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    // Kept alive for the lifetime of the descriptor set; not read after setup.
    _instance_buffer: GpuBuffer,
    /// GPU-readable draw parameters for `vkCmdDrawIndexedIndirect`.
    /// Per-image variants are written by the cull shader; the single no-cull
    /// variant is written by the CPU.
    indirect_buffer_no_cull: GpuBuffer,
    indirect_buffers: Vec<GpuBuffer>,
    visible_buffers: Vec<GpuBuffer>,
    cull_pipeline: vk::Pipeline,
    cull_layout: vk::PipelineLayout,
    cull_set_layout: vk::DescriptorSetLayout,
    cull_pool: vk::DescriptorPool,
    cull_sets: Vec<vk::DescriptorSet>,
    cull_uniform: GpuBuffer,
    /// Host-visible copy of the GPU-written count for optional debug readback.
    cull_readback: GpuBuffer,
    cull_enabled: bool,
    count_override: Option<u32>,
    /// Cluster hierarchy: conservative bounds + per-image visibility flags.
    cluster_count: u32,
    _cluster_bounds: GpuBuffer,
    cluster_flags: Vec<GpuBuffer>,
    cluster_init_pipeline: vk::Pipeline,
    cluster_cull_pipeline: vk::Pipeline,
    cluster_layout: vk::PipelineLayout,
    cluster_set_layout: vk::DescriptorSetLayout,
    cluster_pool: vk::DescriptorPool,
    _cluster_init_set: vk::DescriptorSet,
    cluster_sets: Vec<vk::DescriptorSet>,
    /// HZB: max-depth mip chain rebuilt from the previous frame's depth.
    _material_image: GpuImage,
    _material_view: vk::ImageView,
    _material_sampler: vk::Sampler,
    hzb_image: GpuImage,
    _hzb_storage_views: Vec<vk::ImageView>,
    _hzb_sampled_view: vk::ImageView,
    hzb_from_depth_pipeline: vk::Pipeline,
    hzb_reduce_pipeline: vk::Pipeline,
    hzb_layout: vk::PipelineLayout,
    hzb_set_layout: vk::DescriptorSetLayout,
    hzb_pool: vk::DescriptorPool,
    hzb_sets: Vec<Vec<vk::DescriptorSet>>,
    _hzb_uniforms: Vec<GpuBuffer>,
    occlusion_uniforms: Vec<GpuBuffer>,
    hzb_width: u32,
    hzb_height: u32,
    uniform_buffer: GpuBuffer,
    index_count: u32,
    base_count: u32,
    /// GPU bowl generator, present only for the explicit `--instances` stress mode.
    generate_pipeline: Option<vk::Pipeline>,
    generate_layout: Option<vk::PipelineLayout>,
    generate_set_layout: Option<vk::DescriptorSetLayout>,
    generate_pool: Option<vk::DescriptorPool>,
}

impl MeshPass {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        shadow_views: &[vk::ImageView],
        shadow_sampler: vk::Sampler,
        obj_path: Option<&Path>,
        instance_count: Option<u32>,
        image_count: u32,
        depth_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<Self, RenderError> {
        let device = context.device().clone();
        assert_eq!(shadow_views.len(), 3, "CSM requires three shadow views");

        let vertex_spv = shader::compile_wgsl(MESH_WGSL, "vs_main", naga::ShaderStage::Vertex)?;
        let fragment_spv = shader::compile_wgsl(MESH_WGSL, "fs_main", naga::ShaderStage::Fragment)?;

        let vertex_module = create_shader_module(&device, &vertex_spv)?;
        let fragment_module = create_shader_module(&device, &fragment_spv)?;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
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
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::SAMPLER)
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
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;

        let binding_description = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX);
        let attribute_descriptions = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, normal) as u32),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&binding_description))
            .vertex_attribute_descriptions(&attribute_descriptions);

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
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0);
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
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
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

        let sky_vertex_spv = shader::compile_wgsl(SKY_WGSL, "vs_main", naga::ShaderStage::Vertex)?;
        let sky_fragment_spv =
            shader::compile_wgsl(SKY_WGSL, "fs_main", naga::ShaderStage::Fragment)?;
        let sky_vertex_module = create_shader_module(&device, &sky_vertex_spv)?;
        let sky_fragment_module = create_shader_module(&device, &sky_fragment_spv)?;
        let sky_depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);
        let sky_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(sky_vertex_module)
                .name(VERTEX_ENTRY),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(sky_fragment_module)
                .name(FRAGMENT_ENTRY),
        ];
        let sky_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&sky_stages)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&sky_depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);
        let sky_pipeline = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&sky_pipeline_info),
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
            device.destroy_shader_module(sky_vertex_module, None);
            device.destroy_shader_module(sky_fragment_module, None);
        }

        let (vertices, indices, instances, source) = match obj_path {
            Some(path) => {
                let (vertices, indices, instance) = load_obj(path)?;
                (vertices, indices, vec![instance], "obj")
            }
            None => {
                let (vertices, indices) = cube_mesh();
                (vertices, indices, build_scene(), "cube grid")
            }
        };
        let vertex_buffer = context.upload_buffer(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            bytemuck::cast_slice(&vertices),
            "HBD vertex buffer",
        )?;
        let index_buffer = context.upload_buffer(
            vk::BufferUsageFlags::INDEX_BUFFER,
            bytemuck::cast_slice(&indices),
            "HBD index buffer",
        )?;
        let index_count = indices.len() as u32;
        let (instance_buffer, generated, base_count, instance_source) = if let Some(count) =
            instance_count
        {
            if count == 0 {
                return Err(RenderError::Obj(
                    "--instances must be greater than zero".to_owned(),
                ));
            }
            // Device-local storage buffer large enough for the generated
            // bowl. The compute pass fills it once at construction.
            let buffer = context.create_buffer(
                u64::from(count) * size_of::<Instance>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                "HBD generated instance buffer",
            )?;

            let generate_spv =
                shader::compile_wgsl(MESH_GENERATE_WGSL, "cs_main", naga::ShaderStage::Compute)?;
            let generate_module = create_shader_module(&device, &generate_spv)?;
            let generate_bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let generate_set_layout = unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&generate_bindings),
                    None,
                )
            }
            .map_err(GfxError::Vulkan)?;
            let generate_layout = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(std::slice::from_ref(&generate_set_layout)),
                    None,
                )
            }
            .map_err(GfxError::Vulkan)?;
            let generate_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(generate_module)
                .name(c"cs_main");
            let generate_info = vk::ComputePipelineCreateInfo::default()
                .layout(generate_layout)
                .stage(generate_stage);
            let generate_pipeline = unsafe {
                device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&generate_info),
                    None,
                )
            }
            .map_err(|(_, error)| RenderError::Pipeline(error))?
            .into_iter()
            .next()
            .ok_or(RenderError::MissingPipeline)?;
            unsafe { device.destroy_shader_module(generate_module, None) };

            let generate_pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&[vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(1)])
                        .max_sets(1),
                    None,
                )
            }
            .map_err(GfxError::Vulkan)?;
            let generate_set = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(generate_pool)
                        .set_layouts(std::slice::from_ref(&generate_set_layout)),
                )
            }
            .map_err(GfxError::Vulkan)?[0];
            let generated_info = [vk::DescriptorBufferInfo::default()
                .buffer(buffer.buffer())
                .offset(0)
                .range(buffer.size())];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(generate_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&generated_info)],
                    &[],
                );
            }
            let workgroups = count.div_ceil(256);
            context.one_time_commands(|device, command_buffer| unsafe {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    generate_pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    generate_layout,
                    0,
                    std::slice::from_ref(&generate_set),
                    &[],
                );
                device.cmd_dispatch(command_buffer, workgroups, 1, 1);
            })?;
            info!(count, workgroups, "GPU bowl instance generation dispatched");
            (
                buffer,
                Some((
                    generate_pipeline,
                    generate_layout,
                    generate_set_layout,
                    generate_pool,
                )),
                count,
                "GPU bowl generator",
            )
        } else {
            let buffer = context.upload_buffer(
                vk::BufferUsageFlags::STORAGE_BUFFER,
                bytemuck::cast_slice(&instances),
                "HBD instance buffer",
            )?;
            let count = instances.len() as u32;
            (buffer, None, count, source)
        };
        let draw = [index_count, base_count, 0u32, 0u32, 0u32];
        // The no-cull draw args are CPU-written. Per-image args are written by
        // the cull shader (atomic count) and reset with vkCmdFillBuffer.
        let indirect_buffer_no_cull = context.create_buffer(
            size_of::<[u32; 5]>() as u64,
            vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "HBD indirect draw (no cull)",
        )?;
        indirect_buffer_no_cull.write(bytemuck::cast_slice(&draw))?;

        let mut indirect_buffers = Vec::with_capacity(image_count as usize);
        let mut visible_buffers = Vec::with_capacity(image_count as usize);
        for _ in 0..image_count {
            let indirect = context.create_buffer(
                size_of::<[u32; 5]>() as u64,
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "HBD indirect draw (culled)",
            )?;
            indirect.write(bytemuck::cast_slice(&draw))?;
            let visible = context.create_buffer(
                u64::from(base_count) * size_of::<Instance>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                "HBD visible instances",
            )?;
            indirect_buffers.push(indirect);
            visible_buffers.push(visible);
        }
        info!(
            image_count,
            bytes_per_image = base_count as u64 * size_of::<Instance>() as u64,
            "per-image cull buffers allocated"
        );

        let uniform_buffer = context.create_buffer(
            size_of::<FrameUniform>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "HBD frame uniform",
        )?;

        // --- GPU frustum culling resources ---------------------------------
        let cull_spv = shader::compile_wgsl(CULL_WGSL, "cs_main", naga::ShaderStage::Compute)?;
        let cull_module = create_shader_module(&device, &cull_spv)?;
        let cull_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let cull_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&cull_bindings),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let cull_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[cull_set_layout]),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let cull_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(cull_module)
            .name(c"cs_main");
        let cull_info = vk::ComputePipelineCreateInfo::default()
            .layout(cull_layout)
            .stage(cull_stage);
        let cull_pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&cull_info),
                None,
            )
        }
        .map_err(|(_, error)| RenderError::Pipeline(error))?
        .into_iter()
        .next()
        .ok_or(RenderError::MissingPipeline)?;
        unsafe { device.destroy_shader_module(cull_module, None) };

        let cull_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(image_count),
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(image_count * 4),
                    ])
                    .max_sets(image_count),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let cull_layouts_vec = vec![cull_set_layout; image_count as usize];
        let cull_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(cull_pool)
                    .set_layouts(&cull_layouts_vec),
            )
        }
        .map_err(GfxError::Vulkan)?;
        let cull_uniform = context.create_buffer(
            size_of::<CullUniform>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "HBD cull uniform",
        )?;
        let cull_readback = context.create_buffer(
            4,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "HBD cull readback",
        )?;

        // --- Cluster hierarchy --------------------------------------------
        let cluster_count = base_count.div_ceil(256);
        let cluster_bounds = context.create_buffer(
            u64::from(cluster_count) * 16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
            "HBD cluster bounds",
        )?;
        let mut cluster_flags = Vec::with_capacity(image_count as usize);
        for _ in 0..image_count {
            cluster_flags.push(context.create_buffer(
                u64::from(cluster_count) * 4,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuOnly,
                "HBD cluster flags",
            )?);
        }
        let init_spv = shader::compile_wgsl(CLUSTER_WGSL, "init_cs", naga::ShaderStage::Compute)?;
        let cull_spv =
            shader::compile_wgsl(CLUSTER_WGSL, "cluster_cull_cs", naga::ShaderStage::Compute)?;
        let init_module = create_shader_module(&device, &init_spv)?;
        let cull_module = create_shader_module(&device, &cull_spv)?;
        let cluster_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let cluster_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&cluster_bindings),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let cluster_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[cluster_set_layout]),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let cluster_pipeline = |module: vk::ShaderModule,
                                entry: &std::ffi::CStr|
         -> Result<vk::Pipeline, RenderError> {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            let info = vk::ComputePipelineCreateInfo::default()
                .layout(cluster_layout)
                .stage(stage);
            unsafe {
                device
                    .create_compute_pipelines(
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
        let cluster_init_pipeline = cluster_pipeline(init_module, c"init_cs")?;
        let cluster_cull_pipeline = cluster_pipeline(cull_module, c"cluster_cull_cs")?;
        unsafe {
            device.destroy_shader_module(init_module, None);
            device.destroy_shader_module(cull_module, None);
        }

        let cluster_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(image_count + 1),
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_BUFFER)
                            .descriptor_count(4 * (image_count + 1)),
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(image_count + 1),
                    ])
                    .max_sets(image_count + 1),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let cluster_alloc_layouts = vec![cluster_set_layout; (image_count + 1) as usize];
        let cluster_allocated = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(cluster_pool)
                    .set_layouts(&cluster_alloc_layouts),
            )
        }
        .map_err(GfxError::Vulkan)?;
        let cluster_init_set = cluster_allocated[0];
        let cluster_sets: Vec<vk::DescriptorSet> = cluster_allocated[1..].to_vec();

        // Bounds init needs count/cluster_count only; zeros in the plane slots.
        cull_uniform
            .write(bytemuck::bytes_of(&CullUniform {
                planes: [[0.0; 4]; 6],
                count: base_count,
                cluster_count,
                pad: [0; 2],
            }))
            .map_err(RenderError::Gfx)?;
        let cull_uniform_info = [vk::DescriptorBufferInfo::default()
            .buffer(cull_uniform.buffer())
            .offset(0)
            .range(cull_uniform.size())];
        let source_info = [vk::DescriptorBufferInfo::default()
            .buffer(instance_buffer.buffer())
            .offset(0)
            .range(instance_buffer.size())];
        let bounds_info = [vk::DescriptorBufferInfo::default()
            .buffer(cluster_bounds.buffer())
            .offset(0)
            .range(cluster_bounds.size())];
        let mut occlusion_uniforms = Vec::with_capacity(image_count as usize);
        for _ in 0..image_count {
            occlusion_uniforms.push(context.create_buffer(
                size_of::<Occlusion>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                "HBD occlusion params",
            )?);
        }
        let write_cluster_set =
            |set: vk::DescriptorSet, flags: &GpuBuffer, occlusion: &GpuBuffer| {
                let flags_info = [vk::DescriptorBufferInfo::default()
                    .buffer(flags.buffer())
                    .offset(0)
                    .range(flags.size())];
                let occlusion_info = [vk::DescriptorBufferInfo::default()
                    .buffer(occlusion.buffer())
                    .offset(0)
                    .range(occlusion.size())];
                let hzb_sampled_info = [vk::DescriptorImageInfo::default()
                    .image_view(depth_views[0])
                    .image_layout(vk::ImageLayout::GENERAL)];
                unsafe {
                    device.update_descriptor_sets(
                        &[
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(0)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(&source_info),
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(1)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(&bounds_info),
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(2)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(&cull_uniform_info),
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(3)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(&flags_info),
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(4)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(&occlusion_info),
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(5)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(&hzb_sampled_info),
                        ],
                        &[],
                    );
                }
            };
        write_cluster_set(cluster_init_set, &cluster_flags[0], &occlusion_uniforms[0]);
        for (index, set) in cluster_sets.iter().enumerate() {
            write_cluster_set(*set, &cluster_flags[index], &occlusion_uniforms[index]);
        }
        context.one_time_commands(|device, command_buffer| unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                cluster_init_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                cluster_layout,
                0,
                std::slice::from_ref(&cluster_init_set),
                &[],
            );
            device.cmd_dispatch(command_buffer, cluster_count, 1, 1);
        })?;
        info!(
            cluster_count,
            cluster_size = 256,
            "cluster bounds initialized"
        );

        // --- HZB (max-depth mip chain) ------------------------------------
        let hzb_width = (extent.width / 2).max(1);
        let hzb_height = (extent.height / 2).max(1);
        let hzb_image = context.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R32_SFLOAT)
                .extent(vk::Extent3D {
                    width: hzb_width,
                    height: hzb_height,
                    depth: 1,
                })
                .mip_levels(HZB_MIPS)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            "HBD HZB",
        )?;
        let sampled_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(hzb_image.image())
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R32_SFLOAT)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(HZB_MIPS)
                            .base_array_layer(0)
                            .layer_count(1),
                    ),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let mut hzb_storage_views = Vec::with_capacity(HZB_MIPS as usize);
        for mip in 0..HZB_MIPS {
            hzb_storage_views.push(
                unsafe {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(hzb_image.image())
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::R32_SFLOAT)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .base_mip_level(mip)
                                    .level_count(1)
                                    .base_array_layer(0)
                                    .layer_count(1),
                            ),
                        None,
                    )
                }
                .map_err(GfxError::Vulkan)?,
            );
        }
        context.one_time_commands(|device, command_buffer| unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(hzb_image.image())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(HZB_MIPS)
                            .base_array_layer(0)
                            .layer_count(1),
                    )],
            );
        })?;

        let hzb_bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let hzb_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&hzb_bindings),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let hzb_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[hzb_set_layout]),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let hzb_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let make_hzb_pipeline = |entry: &std::ffi::CStr, module: vk::ShaderModule| {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            let info = vk::ComputePipelineCreateInfo::default()
                .layout(hzb_layout)
                .stage(stage);
            unsafe {
                device
                    .create_compute_pipelines(
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
        let from_depth_spv =
            shader::compile_wgsl(HZB_WGSL, "from_depth_cs", naga::ShaderStage::Compute)?;
        let reduce_spv = shader::compile_wgsl(HZB_WGSL, "reduce_cs", naga::ShaderStage::Compute)?;
        let from_depth_module = create_shader_module(&device, &from_depth_spv)?;
        let reduce_module = create_shader_module(&device, &reduce_spv)?;
        let hzb_from_depth_pipeline = make_hzb_pipeline(c"from_depth_cs", from_depth_module)?;
        let hzb_reduce_pipeline = make_hzb_pipeline(c"reduce_cs", reduce_module)?;
        unsafe {
            device.destroy_shader_module(from_depth_module, None);
            device.destroy_shader_module(reduce_module, None);
        }

        let total_sets = image_count * HZB_MIPS;
        let hzb_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&[
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(total_sets),
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(total_sets),
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(total_sets),
                        vk::DescriptorPoolSize::default()
                            .ty(vk::DescriptorType::UNIFORM_BUFFER)
                            .descriptor_count(total_sets),
                    ])
                    .max_sets(total_sets),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let hzb_layouts = vec![hzb_set_layout; total_sets as usize];
        let hzb_allocated = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(hzb_pool)
                    .set_layouts(&hzb_layouts),
            )
        }
        .map_err(GfxError::Vulkan)?;

        let mut hzb_uniforms = Vec::with_capacity(total_sets as usize);
        let mut hzb_sets = Vec::with_capacity(image_count as usize);
        for image in 0..image_count {
            let mut per_image = Vec::with_capacity(HZB_MIPS as usize);
            for mip in 0..HZB_MIPS {
                let set = hzb_allocated[(image * HZB_MIPS + mip) as usize];
                let src_mip = mip.saturating_sub(1);
                let src_width = if mip == 0 {
                    extent.width
                } else {
                    (hzb_width >> src_mip).max(1)
                };
                let src_height = if mip == 0 {
                    extent.height
                } else {
                    (hzb_height >> src_mip).max(1)
                };
                let uniform = context.create_buffer(
                    size_of::<HzbLevel>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                    "HBD HZB level",
                )?;
                uniform
                    .write(bytemuck::bytes_of(&HzbLevel {
                        src_mip,
                        src_width,
                        src_height,
                        pad: 0,
                    }))
                    .map_err(RenderError::Gfx)?;
                let depth_info = [vk::DescriptorImageInfo::default()
                    .sampler(hzb_sampler)
                    .image_view(depth_views[image as usize])
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let hzb_sampled_info = [vk::DescriptorImageInfo::default()
                    .image_view(sampled_view)
                    .image_layout(vk::ImageLayout::GENERAL)];
                let output_info = [vk::DescriptorImageInfo::default()
                    .image_view(hzb_storage_views[mip as usize])
                    .image_layout(vk::ImageLayout::GENERAL)];
                let uniform_info = [vk::DescriptorBufferInfo::default()
                    .buffer(uniform.buffer())
                    .offset(0)
                    .range(uniform.size())];
                unsafe {
                    device.update_descriptor_sets(
                        &[
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(0)
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .image_info(&depth_info),
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(1)
                                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                .image_info(&hzb_sampled_info),
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(2)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                .image_info(&output_info),
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(3)
                                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                .buffer_info(&uniform_info),
                        ],
                        &[],
                    );
                }
                hzb_uniforms.push(uniform);
                per_image.push(set);
            }
            hzb_sets.push(per_image);
        }
        info!(
            hzb_width,
            hzb_height,
            mips = HZB_MIPS,
            "HZB occlusion chain initialized"
        );
        let hzb_bind = [vk::DescriptorImageInfo::default()
            .image_view(sampled_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        for set in cluster_sets
            .iter()
            .copied()
            .chain(std::iter::once(cluster_init_set))
        {
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(5)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&hzb_bind)],
                    &[],
                );
            }
        }

        let (material_image, material_view, material_sampler) =
            Self::create_material_table(context)?;

        let cull_uniform_info = [vk::DescriptorBufferInfo::default()
            .buffer(cull_uniform.buffer())
            .offset(0)
            .range(cull_uniform.size())];
        let source_info = [vk::DescriptorBufferInfo::default()
            .buffer(instance_buffer.buffer())
            .offset(0)
            .range(instance_buffer.size())];
        for (index, set) in cull_sets.iter().enumerate() {
            let visible_info = [vk::DescriptorBufferInfo::default()
                .buffer(visible_buffers[index].buffer())
                .offset(0)
                .range(visible_buffers[index].size())];
            let indirect_info = [vk::DescriptorBufferInfo::default()
                .buffer(indirect_buffers[index].buffer())
                .offset(0)
                .range(indirect_buffers[index].size())];
            let cluster_flags_info = [vk::DescriptorBufferInfo::default()
                .buffer(cluster_flags[index].buffer())
                .offset(0)
                .range(cluster_flags[index].size())];
            unsafe {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&cull_uniform_info),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&source_info),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&visible_info),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(3)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&indirect_info),
                        vk::WriteDescriptorSet::default()
                            .dst_set(*set)
                            .dst_binding(4)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&cluster_flags_info),
                    ],
                    &[],
                );
            }
        }
        info!(
            instances = base_count,
            image_count, "GPU frustum culling ready"
        );

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(image_count + 1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(image_count + 1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(4 * (image_count + 1)),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(2 * (image_count + 1)),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(image_count + 1),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let allocated_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&vec![descriptor_set_layout; (image_count + 1) as usize]),
            )
        }
        .map_err(GfxError::Vulkan)?;
        let graphics_sets: Vec<vk::DescriptorSet> = allocated_sets[..image_count as usize].to_vec();
        let graphics_set_no_cull = allocated_sets[image_count as usize];

        let source_info = [vk::DescriptorBufferInfo::default()
            .buffer(instance_buffer.buffer())
            .offset(0)
            .range(instance_buffer.size())];
        let shadow_image_infos: Vec<_> = shadow_views
            .iter()
            .copied()
            .map(|view| {
                vk::DescriptorImageInfo::default()
                    .image_view(view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            })
            .collect();
        let shadow_sampler_info = [vk::DescriptorImageInfo::default().sampler(shadow_sampler)];
        let material_image_info = [vk::DescriptorImageInfo::default()
            .image_view(material_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let material_sampler_info = [vk::DescriptorImageInfo::default().sampler(material_sampler)];
        let uniform_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer.buffer())
            .offset(0)
            .range(uniform_buffer.size())];

        for (index, set) in graphics_sets.iter().enumerate() {
            let instance_info = [vk::DescriptorBufferInfo::default()
                .buffer(visible_buffers[index].buffer())
                .offset(0)
                .range(visible_buffers[index].size())];
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
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&shadow_image_infos[0..1]),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&shadow_image_infos[1..2]),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&shadow_image_infos[2..3]),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&shadow_sampler_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&material_image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&material_sampler_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let no_cull_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(graphics_set_no_cull)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_info),
            vk::WriteDescriptorSet::default()
                .dst_set(graphics_set_no_cull)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&source_info),
            vk::WriteDescriptorSet::default()
                .dst_set(graphics_set_no_cull)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&shadow_image_infos[0..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(graphics_set_no_cull)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&shadow_image_infos[1..2]),
            vk::WriteDescriptorSet::default()
                .dst_set(graphics_set_no_cull)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&shadow_image_infos[2..3]),
            vk::WriteDescriptorSet::default()
                .dst_set(graphics_set_no_cull)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&shadow_sampler_info),
            vk::WriteDescriptorSet::default()
                .dst_set(graphics_set_no_cull)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&material_image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(graphics_set_no_cull)
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&material_sampler_info),
        ];
        unsafe { device.update_descriptor_sets(&no_cull_writes, &[]) };

        info!(
            instances = base_count,
            triangles = indices.len() / 3,
            source = instance_source,
            "mesh pipeline ready"
        );

        Ok(Self {
            device,
            pipeline,
            sky_pipeline,
            layout: pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            graphics_sets,
            graphics_set_no_cull,
            vertex_buffer,
            index_buffer,
            _instance_buffer: instance_buffer,
            indirect_buffer_no_cull,
            indirect_buffers,
            visible_buffers,
            cull_pipeline,
            cull_layout,
            cull_set_layout,
            cull_pool,
            cull_sets,
            cull_uniform,
            cull_readback,
            cull_enabled: true,
            count_override: None,
            cluster_count,
            _cluster_bounds: cluster_bounds,
            cluster_flags,
            cluster_init_pipeline,
            cluster_cull_pipeline,
            cluster_layout,
            cluster_set_layout,
            cluster_pool,
            _cluster_init_set: cluster_init_set,
            cluster_sets,
            _material_image: material_image,
            _material_view: material_view,
            _material_sampler: material_sampler,
            hzb_image,
            _hzb_storage_views: hzb_storage_views,
            _hzb_sampled_view: sampled_view,
            hzb_from_depth_pipeline,
            hzb_reduce_pipeline,
            hzb_layout,
            hzb_set_layout,
            hzb_pool,
            hzb_sets,
            _hzb_uniforms: hzb_uniforms,
            occlusion_uniforms,
            hzb_width,
            hzb_height,
            uniform_buffer,
            index_count,
            base_count,
            generate_pipeline: generated.as_ref().map(|g| g.0),
            generate_layout: generated.as_ref().map(|g| g.1),
            generate_set_layout: generated.as_ref().map(|g| g.2),
            generate_pool: generated.as_ref().map(|g| g.3),
        })
    }

    pub(crate) fn update(
        &self,
        camera: &Camera,
        light: &DirectionalLight,
        aspect: f32,
        jitter: Vec2,
    ) -> Result<(), RenderError> {
        let view_projection = camera.view_projection_with_jitter(aspect, jitter);
        let uniform = FrameUniform {
            view_projection: view_projection.to_cols_array_2d(),
            inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
            light_vp0: light.view_projection(12.0).to_cols_array_2d(),
            light_vp1: light.view_projection(32.0).to_cols_array_2d(),
            light_vp2: light.view_projection(80.0).to_cols_array_2d(),
            eye_position: [camera.position.x, camera.position.y, camera.position.z, 1.0],
            light_direction: [light.direction.x, light.direction.y, light.direction.z, 0.0],
            light_color: [
                light.color.x * light.intensity,
                light.color.y * light.intensity,
                light.color.z * light.intensity,
                0.06,
            ],
            shadow_splits: [12.0, 32.0, 80.0, self.base_count as f32],
        };

        // Frustum planes for the cull pass. Use the unjittered pose so subpixel
        // jitter does not churn the visible set. Vulkan clip depth is 0..1.
        let unjittered = camera.view_projection(aspect);
        let c0 = unjittered.col(0);
        let c1 = unjittered.col(1);
        let c2 = unjittered.col(2);
        let c3 = unjittered.col(3);
        let raw_planes = [c3 + c0, c3 - c0, c3 + c1, c3 - c1, c2, c3 - c2];
        let planes = raw_planes.map(|plane| {
            let length = plane.truncate().length().max(1.0e-8);
            [
                plane.x / length,
                plane.y / length,
                plane.z / length,
                plane.w / length,
            ]
        });
        let cull_uniform = CullUniform {
            planes,
            count: self.base_count,
            cluster_count: self.cluster_count,
            pad: [0; 2],
        };
        self.cull_uniform
            .write(bytemuck::bytes_of(&cull_uniform))
            .map_err(RenderError::Gfx)?;

        self.uniform_buffer
            .write(bytemuck::bytes_of(&uniform))
            .map_err(RenderError::Gfx)
    }

    /// Override the instance count for the explicit no-cull stress path.
    pub(crate) fn set_instance_count(&mut self, count: u32) -> Result<(), RenderError> {
        self.count_override = Some(count);
        let draw = [self.index_count, count, 0u32, 0u32, 0u32];
        self.indirect_buffer_no_cull
            .write(bytemuck::cast_slice(&draw))
            .map_err(RenderError::Gfx)
    }

    /// Explicit A/B switch for GPU frustum culling. Always logged by renderer.
    pub(crate) fn set_culling_enabled(&mut self, enabled: bool) {
        self.cull_enabled = enabled;
    }

    pub(crate) fn total_count(&self) -> u32 {
        self.base_count
    }

    pub(crate) fn triangles_per_instance(&self) -> u32 {
        self.index_count / 3
    }

    /// Dispatches culling for one swapchain image. Both the compacted instance
    /// buffer and the indirect draw args are per-image, so no other in-flight
    /// frame can read them while this one writes.
    pub(crate) fn dispatch_cull(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: u32,
        view_projection: hbd_core::math::Mat4,
    ) {
        if !self.cull_enabled {
            return;
        }
        let index = image_index as usize;
        let device = &self.device;
        let aspect = self.hzb_width as f32 / self.hzb_height.max(1) as f32;
        let projection = view_projection;
        let occlusion = Occlusion {
            view_projection: projection.to_cols_array_2d(),
            proj_scale: [
                projection.col(0).x * self.hzb_width as f32,
                projection.col(1).y * self.hzb_height as f32,
            ],
            screen: [self.hzb_width as f32 * 2.0, self.hzb_height as f32 * 2.0],
            mip_count: HZB_MIPS,
            enabled: 1,
            pad: [0; 2],
        };
        let _ = aspect;
        self.occlusion_uniforms[index]
            .write(bytemuck::bytes_of(&occlusion))
            .map_err(RenderError::Gfx)
            .ok();
        unsafe {
            // Reset the atomic instanceCount as a transfer op, then make it
            // visible to the compute shader before it atomically increments.
            device.cmd_fill_buffer(
                command_buffer,
                self.indirect_buffers[index].buffer(),
                4,
                4,
                0,
            );
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .buffer(self.indirect_buffers[index].buffer())
                    .offset(0)
                    .size(vk::WHOLE_SIZE)],
                &[],
            );

            // Rebuild the HZB from this image's previous-frame depth, then
            // reduce 2x2 max into each coarser mip.
            for mip in 0..HZB_MIPS {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    if mip == 0 {
                        self.hzb_from_depth_pipeline
                    } else {
                        self.hzb_reduce_pipeline
                    },
                );
                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.hzb_layout,
                    0,
                    std::slice::from_ref(&self.hzb_sets[index][mip as usize]),
                    &[],
                );
                let width = (self.hzb_width >> mip).max(1);
                let height = (self.hzb_height >> mip).max(1);
                device.cmd_dispatch(command_buffer, width.div_ceil(8), height.div_ceil(8), 1);
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(self.hzb_image.image())
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .base_mip_level(mip)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        )],
                );
            }

            // Cluster pass first: one sphere test per 256 instances publishes
            // visibility flags that the instance pass consults.
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.cluster_cull_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.cluster_layout,
                0,
                std::slice::from_ref(&self.cluster_sets[index]),
                &[],
            );
            device.cmd_dispatch(command_buffer, self.cluster_count.div_ceil(64), 1, 1);
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .buffer(self.cluster_flags[index].buffer())
                    .offset(0)
                    .size(vk::WHOLE_SIZE)],
                &[],
            );

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.cull_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.cull_layout,
                0,
                std::slice::from_ref(&self.cull_sets[index]),
                &[],
            );
            device.cmd_dispatch(command_buffer, self.base_count.div_ceil(256), 1, 1);

            let barriers = [
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .buffer(self.visible_buffers[index].buffer())
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ)
                    .buffer(self.indirect_buffers[index].buffer())
                    .offset(0)
                    .size(vk::WHOLE_SIZE),
            ];
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
        }
    }

    /// Explicit, stalled debug readback of the last culled count for an image.
    pub(crate) fn read_visible_count(
        &self,
        context: &VulkanContext,
        image_index: u32,
    ) -> Result<u32, RenderError> {
        context
            .one_time_commands(|device, command_buffer| unsafe {
                device.cmd_copy_buffer(
                    command_buffer,
                    self.indirect_buffers[image_index as usize].buffer(),
                    self.cull_readback.buffer(),
                    &[vk::BufferCopy::default()
                        .src_offset(4)
                        .dst_offset(0)
                        .size(4)],
                );
            })
            .map_err(RenderError::Gfx)?;
        let mut bytes = [0u8; 4];
        self.cull_readback
            .read(&mut bytes)
            .map_err(RenderError::Gfx)?;
        Ok(u32::from_le_bytes(bytes))
    }

    pub(crate) fn record(
        &self,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        image_index: u32,
    ) {
        // Culling on: this image's compacted buffer. Off: the source buffer and
        // its own CPU-written draw args. They are never mixed.
        let (descriptor_set, indirect_buffer) = if self.cull_enabled {
            (
                self.graphics_sets[image_index as usize],
                &self.indirect_buffers[image_index as usize],
            )
        } else {
            (self.graphics_set_no_cull, &self.indirect_buffer_no_cull)
        };
        unsafe {
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

            // Sky first, with depth test/write disabled; geometry overwrites it.
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.sky_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                std::slice::from_ref(&descriptor_set),
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
            // One indirect draw: the GPU reads index/instance counts from a
            // buffer, so future compute culling can write the draw without a
            // CPU round trip. No visual change from the previous direct draw.
            self.device.cmd_draw_indexed_indirect(
                command_buffer,
                indirect_buffer.buffer(),
                0,
                1,
                0,
            );
        }
    }
}

impl Drop for MeshPass {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline(self.sky_pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            if let Some(pipeline) = self.generate_pipeline {
                self.device.destroy_pipeline(pipeline, None);
            }
            if let Some(layout) = self.generate_layout {
                self.device.destroy_pipeline_layout(layout, None);
            }
            if let Some(pool) = self.generate_pool {
                self.device.destroy_descriptor_pool(pool, None);
            }
            if let Some(layout) = self.generate_set_layout {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device.destroy_pipeline(self.cull_pipeline, None);
            self.device.destroy_pipeline_layout(self.cull_layout, None);
            self.device.destroy_descriptor_pool(self.cull_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.cull_set_layout, None);
            self.device
                .destroy_pipeline(self.cluster_init_pipeline, None);
            self.device
                .destroy_pipeline(self.cluster_cull_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.cluster_layout, None);
            self.device.destroy_descriptor_pool(self.cluster_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.cluster_set_layout, None);
            self.device
                .destroy_pipeline(self.hzb_from_depth_pipeline, None);
            self.device.destroy_pipeline(self.hzb_reduce_pipeline, None);
            self.device.destroy_pipeline_layout(self.hzb_layout, None);
            self.device.destroy_descriptor_pool(self.hzb_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.hzb_set_layout, None);
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

pub(crate) fn cube_mesh() -> (Vec<Vertex>, Vec<u32>) {
    // Four vertices per face keep normals flat without a geometry pass.
    let faces = [
        (
            Vec3::X,
            [
                Vec3::new(0.5, -0.5, -0.5),
                Vec3::new(0.5, -0.5, 0.5),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(0.5, 0.5, -0.5),
            ],
        ),
        (
            Vec3::NEG_X,
            [
                Vec3::new(-0.5, -0.5, 0.5),
                Vec3::new(-0.5, -0.5, -0.5),
                Vec3::new(-0.5, 0.5, -0.5),
                Vec3::new(-0.5, 0.5, 0.5),
            ],
        ),
        (
            Vec3::Y,
            [
                Vec3::new(-0.5, 0.5, -0.5),
                Vec3::new(0.5, 0.5, -0.5),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(-0.5, 0.5, 0.5),
            ],
        ),
        (
            Vec3::NEG_Y,
            [
                Vec3::new(-0.5, -0.5, 0.5),
                Vec3::new(0.5, -0.5, 0.5),
                Vec3::new(0.5, -0.5, -0.5),
                Vec3::new(-0.5, -0.5, -0.5),
            ],
        ),
        (
            Vec3::Z,
            [
                Vec3::new(-0.5, -0.5, 0.5),
                Vec3::new(0.5, -0.5, 0.5),
                Vec3::new(0.5, 0.5, 0.5),
                Vec3::new(-0.5, 0.5, 0.5),
            ],
        ),
        (
            Vec3::NEG_Z,
            [
                Vec3::new(0.5, -0.5, -0.5),
                Vec3::new(-0.5, -0.5, -0.5),
                Vec3::new(-0.5, 0.5, -0.5),
                Vec3::new(0.5, 0.5, -0.5),
            ],
        ),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (normal, corners) in faces {
        let base = vertices.len() as u32;
        for corner in corners {
            vertices.push(Vertex {
                position: corner.to_array(),
                normal: normal.to_array(),
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    (vertices, indices)
}

pub(crate) fn build_scene() -> Vec<Instance> {
    let mut instances = Vec::new();

    // Saturated, distinct colors make the lighting and shadow response easy to
    // read. A washed-out white grid is hard to judge visually.
    const PALETTE: [[f32; 3]; 6] = [
        [0.86, 0.12, 0.14], // red
        [0.12, 0.74, 0.26], // green
        [0.12, 0.34, 0.92], // blue
        [0.94, 0.74, 0.12], // yellow
        [0.74, 0.14, 0.84], // magenta
        [0.10, 0.82, 0.82], // cyan
    ];

    // A single flat cube acts as the ground plane.
    instances.push(Instance {
        model: (Mat4::from_translation(Vec3::new(0.0, -0.85, 0.0))
            * Mat4::from_scale(Vec3::new(14.0, 0.2, 14.0)))
        .to_cols_array_2d(),
        base_color: [0.12, 0.13, 0.15, 1.0],
        material: [0.0, 0.92, 1.0, 0.0],
    });

    for x in -GRID_HALF_EXTENT..=GRID_HALF_EXTENT {
        for z in -GRID_HALF_EXTENT..=GRID_HALF_EXTENT {
            let pattern = (x * 7 + z * 13).rem_euclid(5);
            let height = 0.5 + pattern as f32 * 0.22;
            let position = Vec3::new(
                x as f32 * GRID_SPACING,
                -0.75 + height * 0.5,
                z as f32 * GRID_SPACING,
            );
            let model =
                Mat4::from_translation(position) * Mat4::from_scale(Vec3::new(0.52, height, 0.52));

            let cool = (z + GRID_HALF_EXTENT) as f32 / (GRID_HALF_EXTENT * 2) as f32;
            let metallic = pattern as f32 / 4.0 * 0.85;
            let roughness = 0.18 + cool * 0.65;
            let color = PALETTE[(x * 3 + z * 5).rem_euclid(PALETTE.len() as i32) as usize];
            instances.push(Instance {
                model: model.to_cols_array_2d(),
                base_color: [color[0], color[1], color[2], 1.0],
                material: [metallic, roughness, 1.0, (x + z).rem_euclid(4) as f32],
            });
        }
    }

    instances
}

/// Temporary early-visualization loader for `--obj`.
///
/// This is deliberately not the asset pipeline: it reads only positions and
/// faces, derives smooth normals from the triangles, ignores materials,
/// textures and vertex normals, and normalizes the whole mesh into the M1 test
/// scene as a single instance. See `docs/EARLY-OBJ-TEST.md`.
pub(crate) fn load_obj(path: &Path) -> Result<(Vec<Vertex>, Vec<u32>, Instance), RenderError> {
    let location = path.display().to_string();
    let text = std::fs::read_to_string(path)
        .map_err(|error| RenderError::Obj(format!("{location}: {error}")))?;

    let mut positions: Vec<Vec3> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();

    for (line_number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = || format!("{location} line {}", line_number + 1);
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("v") => {
                let mut position = [0.0f32; 3];
                for component in &mut position {
                    let token = parts
                        .next()
                        .ok_or_else(|| RenderError::Obj(format!("{}: incomplete vertex", at())))?;
                    *component = token.parse::<f32>().map_err(|error| {
                        RenderError::Obj(format!(
                            "{}: bad vertex component '{token}': {error}",
                            at()
                        ))
                    })?;
                }
                positions.push(Vec3::from_array(position));
            }
            Some("f") => {
                let mut corners = Vec::new();
                for token in parts {
                    let raw = token.split('/').next().unwrap_or(token);
                    let index: i64 = raw.parse().map_err(|error| {
                        RenderError::Obj(format!("{}: bad face index '{token}': {error}", at()))
                    })?;
                    let resolved = if index < 0 {
                        positions.len() as i64 + index
                    } else {
                        index - 1
                    };
                    if resolved < 0 || resolved as usize >= positions.len() {
                        return Err(RenderError::Obj(format!(
                            "{}: face index out of range",
                            at()
                        )));
                    }
                    corners.push(resolved as u32);
                }
                for i in 1..corners.len().saturating_sub(1) {
                    faces.push([corners[0], corners[i], corners[i + 1]]);
                }
            }
            _ => {}
        }
    }

    if positions.is_empty() || faces.is_empty() {
        return Err(RenderError::Obj(format!(
            "{location}: no vertices or faces found"
        )));
    }

    let mut normals = vec![Vec3::ZERO; positions.len()];
    for face in &faces {
        let a = positions[face[0] as usize];
        let b = positions[face[1] as usize];
        let c = positions[face[2] as usize];
        let normal = (b - a).cross(c - a);
        for index in face {
            normals[*index as usize] += normal;
        }
    }
    for normal in &mut normals {
        if normal.length_squared() > f32::EPSILON {
            *normal = normal.normalize();
        } else {
            *normal = Vec3::Y;
        }
    }

    let vertices: Vec<Vertex> = positions
        .iter()
        .zip(normals.iter())
        .map(|(position, normal)| Vertex {
            position: position.to_array(),
            normal: normal.to_array(),
        })
        .collect();
    let indices: Vec<u32> = faces.into_iter().flatten().collect();

    let minimum = positions
        .iter()
        .fold(Vec3::splat(f32::INFINITY), |acc, p| acc.min(*p));
    let maximum = positions
        .iter()
        .fold(Vec3::splat(f32::NEG_INFINITY), |acc, p| acc.max(*p));
    let center = (minimum + maximum) * 0.5;
    let extent = maximum - minimum;
    let largest = extent.x.max(extent.y).max(extent.z);
    if !largest.is_finite() || largest <= 0.0 {
        return Err(RenderError::Obj(format!("{location}: degenerate bounds")));
    }
    let scale = 5.0 / largest;
    let model = Mat4::from_translation(Vec3::new(0.0, 1.6, 0.0))
        * Mat4::from_rotation_y(std::f32::consts::PI)
        * Mat4::from_scale(Vec3::splat(scale))
        * Mat4::from_translation(-center);
    let instance = Instance {
        model: model.to_cols_array_2d(),
        base_color: [0.06, 0.16, 0.5, 1.0],
        material: [0.18, 0.36, 1.0, 0.0],
    };

    Ok((vertices, indices, instance))
}

impl MeshPass {
    fn create_material_table(
        context: &VulkanContext,
    ) -> Result<(GpuImage, vk::ImageView, vk::Sampler), RenderError> {
        let device = context.device().clone();
        // --- Bindless material table (4 texture layers) -------------------
        const MATERIAL_LAYERS: u32 = 4;
        const MATERIAL_SIZE: u32 = 4;
        let material_image = context.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width: MATERIAL_SIZE,
                    height: MATERIAL_SIZE,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(MATERIAL_LAYERS)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            "HBD material table",
        )?;
        // Four simple patterns so the per-instance layer index is visible.
        let mut material_pixels =
            Vec::with_capacity((MATERIAL_SIZE * MATERIAL_SIZE * MATERIAL_LAYERS * 4) as usize);
        for layer in 0..MATERIAL_LAYERS {
            for y in 0..MATERIAL_SIZE {
                for x in 0..MATERIAL_SIZE {
                    let checker = (x + y) % 2 == 0;
                    let stripe = x < 2;
                    let dot = x == 1 && y == 1;
                    let shade = match layer {
                        0 => {
                            if checker {
                                255
                            } else {
                                170
                            }
                        }
                        1 => {
                            if stripe {
                                255
                            } else {
                                110
                            }
                        }
                        2 => {
                            if dot {
                                255
                            } else {
                                140
                            }
                        }
                        _ => 120 + x * 32 + y * 8,
                    } as u8;
                    material_pixels.extend_from_slice(&[shade, shade, shade, 255]);
                }
            }
        }
        let material_staging = context.create_buffer(
            material_pixels.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "HBD material staging",
        )?;
        material_staging
            .write(&material_pixels)
            .map_err(RenderError::Gfx)?;
        context.one_time_commands(|device, command_buffer| unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(material_image.image())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(MATERIAL_LAYERS),
                    )],
            );
            device.cmd_copy_buffer_to_image(
                command_buffer,
                material_staging.buffer(),
                material_image.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(MATERIAL_LAYERS),
                    )
                    .image_extent(vk::Extent3D {
                        width: MATERIAL_SIZE,
                        height: MATERIAL_SIZE,
                        depth: 1,
                    })],
            );
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(material_image.image())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(MATERIAL_LAYERS),
                    )],
            );
        })?;
        let material_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(material_image.image())
                    .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                    .format(vk::Format::R8G8B8A8_UNORM)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(MATERIAL_LAYERS),
                    ),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let material_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        info!(layers = MATERIAL_LAYERS, "bindless material table ready");
        Ok((material_image, material_view, material_sampler))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_obj_triangulates_and_normalizes() {
        let path = std::env::temp_dir().join("hbd-obj-loader-test.obj");
        std::fs::write(
            &path,
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 4 2\nf 2 4 3\nf 3 4 1\n",
        )
        .unwrap();

        let (vertices, indices, instance) = load_obj(&path).unwrap();
        assert_eq!(vertices.len(), 4);
        assert_eq!(indices.len(), 12);
        assert!(
            instance
                .model
                .iter()
                .all(|row| row.iter().all(|value| value.is_finite()))
        );

        std::fs::remove_file(path).unwrap();
    }
}
