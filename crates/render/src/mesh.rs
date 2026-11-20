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
use hbd_gfx::{GfxError, GpuBuffer, MemoryLocation, VulkanContext};
use tracing::info;

use crate::{Camera, DirectionalLight, RenderError, shader};

const MESH_WGSL: &str = include_str!("mesh.wgsl");
const SKY_WGSL: &str = include_str!("sky.wgsl");
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

pub(crate) struct MeshPass {
    device: ash::Device,
    pipeline: vk::Pipeline,
    sky_pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    vertex_buffer: GpuBuffer,
    index_buffer: GpuBuffer,
    // Kept alive for the lifetime of the descriptor set; not read after setup.
    _instance_buffer: GpuBuffer,
    /// GPU-readable draw parameters for `vkCmdDrawIndexedIndirect`.
    indirect_buffer: GpuBuffer,
    uniform_buffer: GpuBuffer,
    index_count: u32,
    base_count: u32,
}

impl MeshPass {
    pub(crate) fn new(
        context: &VulkanContext,
        render_pass: vk::RenderPass,
        shadow_views: &[vk::ImageView],
        shadow_sampler: vk::Sampler,
        obj_path: Option<&Path>,
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
        let instance_buffer = context.upload_buffer(
            vk::BufferUsageFlags::STORAGE_BUFFER,
            bytemuck::cast_slice(&instances),
            "HBD instance buffer",
        )?;
        let index_count = indices.len() as u32;
        let base_count = instances.len() as u32;
        let draw = [index_count, base_count, 0u32, 0u32, 0u32];
        let indirect_buffer = context.create_buffer(
            size_of::<[u32; 5]>() as u64,
            vk::BufferUsageFlags::INDIRECT_BUFFER,
            MemoryLocation::CpuToGpu,
            "HBD indirect draw",
        )?;
        indirect_buffer.write(bytemuck::cast_slice(&draw))?;
        let uniform_buffer = context.create_buffer(
            size_of::<FrameUniform>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "HBD frame uniform",
        )?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(3),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(1),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(1),
                None,
            )
        }
        .map_err(GfxError::Vulkan)?;
        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(GfxError::Vulkan)?[0];

        let instance_info = [vk::DescriptorBufferInfo::default()
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
        let uniform_info = [vk::DescriptorBufferInfo::default()
            .buffer(uniform_buffer.buffer())
            .offset(0)
            .range(uniform_buffer.size())];
        let descriptor_writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&uniform_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&instance_info),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&shadow_image_infos[0..1]),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&shadow_image_infos[1..2]),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&shadow_image_infos[2..3]),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&shadow_sampler_info),
        ];
        unsafe {
            device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        info!(
            instances = instances.len(),
            triangles = indices.len() / 3,
            source,
            "mesh pipeline ready"
        );

        Ok(Self {
            device,
            pipeline,
            sky_pipeline,
            layout: pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            vertex_buffer,
            index_buffer,
            _instance_buffer: instance_buffer,
            indirect_buffer,
            uniform_buffer,
            index_count,
            base_count,
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

        self.uniform_buffer
            .write(bytemuck::bytes_of(&uniform))
            .map_err(RenderError::Gfx)
    }

    /// Override the indirect instance count. Values above the source instance
    /// count repeat that source set; this is a GPU-throughput stress mode, not
    /// a scene of unique objects. Always called explicitly and logged by the
    /// renderer.
    pub(crate) fn set_instance_count(&self, count: u32) -> Result<(), RenderError> {
        let draw = [self.index_count, count, 0u32, 0u32, 0u32];
        self.indirect_buffer
            .write(bytemuck::cast_slice(&draw))
            .map_err(RenderError::Gfx)
    }

    pub(crate) fn record(&self, command_buffer: vk::CommandBuffer, extent: vk::Extent2D) {
        let descriptor_set = self.descriptor_set;
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
                self.indirect_buffer.buffer(),
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
                material: [metallic, roughness, 1.0, 0.0],
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
