//! Editor panels: dockspace, hierarchy, inspector, toolbar, stats, shaders.

use dear_imgui_rs::{
    Condition, DockLayout, DockLayoutApply, DockNodeFlags, DockSplit, WindowFlags, WindowKey,
    WindowKeyError,
};
use dear_imguizmo::GuizmoExt;
use dear_imguizmo::{Operation, decompose_matrix, recompose_matrix};
use hbd_engine::render::{DebugView, HbtrQuality, RenderResolution};
use tracing::error;

use crate::Editor;
use crate::gizmo::{ViewportRect, handle_viewport};

pub(crate) const WATCHED_SHADERS: &[&str] = &[
    "crates/render/src/mesh.wgsl",
    "crates/render/src/sky.wgsl",
    "crates/render/src/mesh_generate.wgsl",
    "crates/render/src/cull.wgsl",
    "crates/render/src/cluster.wgsl",
    "crates/render/src/hzb.wgsl",
    "crates/render/src/animate.wgsl",
    "crates/render/src/hbtr_resolve.wgsl",
    "crates/render/src/shadow.wgsl",
    "crates/render/src/ssao.wgsl",
    "crates/render/src/ssao_blur.wgsl",
    "crates/render/src/bloom.wgsl",
    "crates/render/src/tonemap.wgsl",
];

struct EditorKeys {
    hierarchy: WindowKey,
    inspector: WindowKey,
    stats: WindowKey,
    shaders: WindowKey,
    toolbar: WindowKey,
    error: WindowKey,
}

impl EditorKeys {
    fn new() -> Result<Self, WindowKeyError> {
        Ok(Self {
            hierarchy: WindowKey::new("hierarchy", "Hierarchy")?,
            inspector: WindowKey::new("inspector", "Inspector")?,
            stats: WindowKey::new("stats", "Stats")?,
            shaders: WindowKey::new("shaders", "Shaders")?,
            toolbar: WindowKey::new("toolbar", "Viewport Controls")?,
            error: WindowKey::new("error", "Error")?,
        })
    }
}

/// Declarative dock tree: Hierarchy left, Inspector right on top, Stats and
/// Shaders stacked right. The dockspace's central node is intentionally left
/// EMPTY: with PASSTHRU_CENTRAL_NODE that is the only configuration where
/// Dear ImGui does not paint an opaque window background over the engine's
/// swapchain render. The viewport is that empty central area.
fn dock_layout(keys: &EditorKeys) -> DockLayout {
    DockLayout::split(
        DockSplit::Left,
        0.2,
        DockLayout::tabs([keys.hierarchy.clone()]),
        DockLayout::split(
            DockSplit::Right,
            0.32,
            DockLayout::split(
                DockSplit::Up,
                0.45,
                DockLayout::tabs([keys.inspector.clone()]),
                DockLayout::tabs([keys.stats.clone(), keys.shaders.clone()]),
            ),
            DockLayout::tabs(Vec::<WindowKey>::new()),
        ),
    )
}

pub fn build_ui(editor: &mut Editor) {
    let ui = editor.imgui.frame();
    let keys = match EditorKeys::new() {
        Ok(keys) => keys,
        Err(error) => {
            error!(%error, "editor window keys unavailable; skipping panels");
            return;
        }
    };
    if let Err(error) = ui
        .dockspace()
        .flags(DockNodeFlags::PASSTHRU_CENTRAL_NODE)
        .layout(&dock_layout(&keys), DockLayoutApply::IfMissing)
        .build()
    {
        error!(%error, "dockspace submission failed");
    }

    // --- Gizmo context: one per frame --------------------------------------
    let giz = ui.guizmo();
    let display_size = ui.io().display_size();

    // --- Transparent floating viewport controls ----------------------------
    ui.window(keys.toolbar.label("Viewport Controls"))
        .flags(
            WindowFlags::NO_BACKGROUND
                | WindowFlags::NO_SCROLLBAR
                | WindowFlags::ALWAYS_AUTO_RESIZE,
        )
        .position(
            [display_size[0] * 0.2 + 12.0, 36.0],
            Condition::FirstUseEver,
        )
        .build(|| {
            ui.menu_bar(|| {
                ui.menu("File", || {
                    if ui.menu_item("Reset Camera") {
                        editor.scene.reset_camera();
                    }
                    if ui.menu_item("Reload Shaders")
                        && let Some(renderer) = editor.renderer.as_mut()
                        && let Err(error) = renderer.rebuild_passes()
                    {
                        editor.error_message = Some(error.to_string());
                    }
                });
                ui.menu("Edit", || {
                    if ui.menu_item_with_shortcut("Translate", "W") {
                        editor.scene.set_gizmo_mode(Operation::TRANSLATE);
                    }
                    if ui.menu_item_with_shortcut("Rotate", "E") {
                        editor.scene.set_gizmo_mode(Operation::ROTATE);
                    }
                    if ui.menu_item_with_shortcut("Scale", "R") {
                        editor.scene.set_gizmo_mode(Operation::SCALE);
                    }
                    if ui.menu_item_with_shortcut("Toggle World/Local", "L") {
                        editor.scene.set_world_mode(!editor.scene.world_mode);
                    }
                    if ui.menu_item_with_shortcut("Toggle Snap", "F") {
                        editor.scene.set_snap_enabled(!editor.scene.snap_enabled);
                    }
                });
            });
            ui.new_line();
            if ui.button("Translate") {
                editor.scene.set_gizmo_mode(Operation::TRANSLATE);
            }
            ui.same_line();
            if ui.button("Rotate") {
                editor.scene.set_gizmo_mode(Operation::ROTATE);
            }
            ui.same_line();
            if ui.button("Scale") {
                editor.scene.set_gizmo_mode(Operation::SCALE);
            }
            ui.same_line();
            if ui.checkbox("World", &mut editor.scene.world_mode) {
                editor.scene.set_world_mode(editor.scene.world_mode);
            }
            ui.same_line();
            if ui.checkbox("Snap", &mut editor.scene.snap_enabled) {
                editor.scene.set_snap_enabled(editor.scene.snap_enabled);
            }
            ui.same_line();
            if ui.button("Reset Camera") {
                editor.scene.reset_camera();
            }
            ui.same_line();
            let mut running = editor.scene.scripts_running;
            if ui.checkbox("Scripts", &mut running) {
                editor.scene.set_scripts_running(running);
            }
        });

    // --- Hierarchy (left) --------------------------------------------------
    ui.window(keys.hierarchy.label("Hierarchy")).build(|| {
        let entities: Vec<_> = editor.scene.object_entities().to_vec();
        for entity in entities {
            let name = editor.scene.name(entity);
            if ui
                .selectable_config(name.as_str())
                .selected(editor.scene.selected() == Some(entity))
                .build()
            {
                editor.scene.select(Some(entity));
            }
        }
    });

    // --- Inspector (right, top) ---------------------------------------------
    ui.window(keys.inspector.label("Inspector")).build(|| {
        let Some(entity) = editor.scene.selected() else {
            ui.text("Select an object in Hierarchy.");
            return;
        };
        let Some(object) = editor.scene.object(entity) else {
            return;
        };
        ui.text(format!("{}  ({:?})", editor.scene.name(entity), entity));
        let (translation, rotation, scale) = decompose_matrix(&object.model);
        let mut translation = translation;
        let mut rotation = rotation;
        let mut scale = scale;
        let mut dirty = false;
        if ui.drag_float3("Pos", &mut translation) {
            dirty = true;
        }
        if ui.drag_float3("Rot", &mut rotation) {
            dirty = true;
        }
        if ui.drag_float3("Scale", &mut scale) {
            dirty = true;
        }
        if dirty {
            editor
                .scene
                .set_transform(entity, &recompose_matrix(&translation, &rotation, &scale));
            if let Some(renderer) = editor.renderer.as_mut()
                && let Err(error) = renderer.update_scene_instances(&editor.scene.instances())
            {
                editor.error_message = Some(error.to_string());
            }
        }
        let mut color = [
            object.base_color.x,
            object.base_color.y,
            object.base_color.z,
        ];
        if ui.color_edit3("Color", &mut color) {
            editor.scene.set_color(entity, color);
            if let Some(renderer) = editor.renderer.as_mut()
                && let Err(error) = renderer.update_scene_instances(&editor.scene.instances())
            {
                editor.error_message = Some(error.to_string());
            }
        }
        let mut metallic = object.material.x;
        let mut roughness = object.material.y;
        let mut ao = object.material.z;
        let mut material_dirty = false;
        if ui.drag_float("Metallic", &mut metallic) {
            material_dirty = true;
        }
        if ui.drag_float("Roughness", &mut roughness) {
            material_dirty = true;
        }
        if ui.drag_float("AO", &mut ao) {
            material_dirty = true;
        }
        if material_dirty {
            editor.scene.set_material(entity, metallic, roughness, ao);
            if let Some(renderer) = editor.renderer.as_mut()
                && let Err(error) = renderer.update_scene_instances(&editor.scene.instances())
            {
                editor.error_message = Some(error.to_string());
            }
        }

        ui.separator();
        ui.text("Lua Script");
        if let Some(path) = editor.scene.script_path(entity) {
            ui.text(path.display().to_string());
            let mut enabled = editor.scene.script_enabled(entity);
            if ui.checkbox("Enabled", &mut enabled) {
                editor.scene.toggle_script(entity, enabled);
            }
            if ui.button("Detach Script") {
                editor.scene.detach_script(entity);
            }
        } else {
            ui.input_text("Script file", &mut editor.script_path_buf)
                .build();
            if ui.button("Attach Script") {
                let path = std::path::PathBuf::from(&editor.script_path_buf);
                editor.scene.attach_script(entity, path);
            }
        }
    });

    // --- Stats (right, bottom) ---------------------------------------------
    ui.window(keys.stats.label("Stats")).build(|| {
        ui.text(format!("Frame {}", editor.frame));
        ui.text(format!(
            "Resolution {:?} / {:?}",
            editor.resolution.label(),
            editor.quality
        ));
        ui.text(format!(
            "Visible {}/{}  ({} tris/instance)",
            editor.last_counts.0, editor.last_counts.1, editor.last_counts.2
        ));
        let mut intensity = editor.scene.light_intensity();
        if ui.drag_float("Light intensity", &mut intensity) {
            editor.scene.set_light_intensity(intensity);
        }
        for (label, view) in [
            ("Final", DebugView::Final),
            ("HDR", DebugView::Hdr),
            ("AO", DebugView::AmbientOcclusion),
            ("History", DebugView::TemporalHistory),
        ] {
            if ui.button(label)
                && let Some(renderer) = editor.renderer.as_mut()
                && let Err(error) = renderer.set_debug_view(view)
            {
                editor.error_message = Some(error.to_string());
            }
        }
    });

    // --- Shaders (right, bottom) -------------------------------------------
    ui.window(keys.shaders.label("Shaders")).build(|| {
        ui.text("Hot reload: crates/render/src/");
        for (path, changed) in WATCHED_SHADERS.iter().zip(editor.shader_mtimes.iter()) {
            let status = if changed.is_some() { "ok" } else { "missing" };
            let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
            ui.text(format!("{name}: {status}"));
        }
        if ui.button("Reload shaders now")
            && let Some(renderer) = editor.renderer.as_mut()
            && let Err(error) = renderer.rebuild_passes()
        {
            editor.error_message = Some(error.to_string());
        }
        for (label, resolution) in [
            ("Native", RenderResolution::Native),
            ("1080p", RenderResolution::P1080),
            ("Auto 50%", RenderResolution::Auto(50)),
            ("Auto 85%", RenderResolution::Auto(85)),
        ] {
            if ui.button(label) {
                let mut applied = true;
                if let Some(renderer) = editor.renderer.as_mut()
                    && let Err(error) = renderer.set_render_resolution(resolution)
                {
                    editor.error_message = Some(error.to_string());
                    applied = false;
                }
                if applied {
                    editor.resolution = resolution;
                }
            }
            ui.same_line();
        }
        ui.new_line();
        for (label, quality) in [
            ("Low", HbtrQuality::Low),
            ("Medium", HbtrQuality::Medium),
            ("High", HbtrQuality::High),
            ("Epic", HbtrQuality::Epic),
        ] {
            if ui.button(label) {
                if let Some(renderer) = editor.renderer.as_mut() {
                    renderer.set_hbtr_quality(quality);
                }
                editor.quality = quality;
            }
            ui.same_line();
        }
        ui.new_line();
    });

    // --- Error overlay ------------------------------------------------------
    if let Some(error) = &editor.error_message {
        let text = error.clone();
        ui.window(keys.error.label("Error")).build(|| {
            ui.text_colored([1.0, 0.3, 0.3, 1.0], text.as_str());
        });
    }

    // --- Viewport interaction over the empty passthrough central node -------
    // The gizmo rect matches the engine projection (full display); interact
    // only when the mouse is not captured by a panel, and never drop an active
    // drag mid-gesture if the pointer crosses a panel.
    giz.enable(!ui.io().want_capture_mouse() || giz.is_using());
    let rect = ViewportRect {
        pos: [0.0, 0.0],
        size: display_size,
    };
    let aspect = editor.renderer.as_ref().map_or(1.0, |renderer| {
        let extent = renderer.render_extent();
        extent.width as f32 / extent.height.max(1) as f32
    });
    let camera = editor.scene.camera().unwrap_or_default();
    if handle_viewport(ui, &giz, &mut editor.scene, &camera, aspect, rect)
        && let Some(renderer) = editor.renderer.as_mut()
        && let Err(error) = renderer.update_scene_instances(&editor.scene.instances())
    {
        editor.error_message = Some(error.to_string());
    }
}
