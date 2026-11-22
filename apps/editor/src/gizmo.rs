//! Viewport picking and the ImGuizmo transform gizmo over the engine pass.

use std::cmp::Ordering;

use crate::scene::{EditorObject, EditorScene};
use dear_imgui_rs::{Key, MouseButton, Ui};
use dear_imguizmo::{DrawListTarget, GizmoUi, Mode, MouseRay, Operation, compute_mouse_ray};
use hbd_engine::core::math::{Mat4, Vec3};
use hbd_engine::render::Camera;

#[derive(Clone, Copy)]
pub struct ViewportRect {
    pub pos: [f32; 2],
    pub size: [f32; 2],
}

impl ViewportRect {
    pub fn contains(&self, point: [f32; 2]) -> bool {
        point[0] >= self.pos[0]
            && point[0] <= self.pos[0] + self.size[0]
            && point[1] >= self.pos[1]
            && point[1] <= self.pos[1] + self.size[1]
    }
}

/// Axis-aligned test box around the transformed cube corners so rotated and
/// scaled instances still pick correctly.
fn object_aabb(model: &Mat4) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for x in [0.0f32, 1.0] {
        for y in [0.0f32, 1.0] {
            for z in [0.0f32, 1.0] {
                let corner = model.transform_point3(Vec3::new(x - 0.5, y - 0.5, z - 0.5));
                for axis in 0..3 {
                    min[axis] = min[axis].min(corner[axis]);
                    max[axis] = max[axis].max(corner[axis]);
                }
            }
        }
    }
    (min, max)
}

fn ray_hits_aabb(
    origin: [f32; 3],
    direction: [f32; 3],
    min: [f32; 3],
    max: [f32; 3],
) -> Option<f32> {
    let mut t_min = 0.0f32;
    let mut t_max = f32::MAX;
    for axis in 0..3 {
        if direction[axis].abs() < f32::EPSILON {
            if origin[axis] < min[axis] || origin[axis] > max[axis] {
                return None;
            }
        } else {
            let mut t1 = (min[axis] - origin[axis]) / direction[axis];
            let mut t2 = (max[axis] - origin[axis]) / direction[axis];
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return None;
            }
        }
    }
    Some(t_min)
}

/// Nearest object whose instance AABB the world-space ray hits, or `None`.
pub fn pick_object(objects: &[EditorObject], ray: &MouseRay) -> Option<usize> {
    objects
        .iter()
        .enumerate()
        .filter_map(|(index, object)| {
            let (min, max) = object_aabb(&Mat4::from_cols_array_2d(&object.instance.model));
            ray_hits_aabb(ray.origin, ray.direction, min, max).map(|distance| (index, distance))
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
}

/// Handles viewport picking and the ImGuizmo transform gizmo. Draws the
/// gizmo on the foreground draw list over `rect`. Returns true when the scene
/// model changed so the caller can push the scene to the renderer.
#[allow(clippy::too_many_arguments)]
pub fn handle_viewport(
    ui: &Ui,
    giz: &GizmoUi<'_>,
    scene: &mut EditorScene,
    camera: &Camera,
    aspect: f32,
    rect: ViewportRect,
) -> bool {
    let mut changed = false;
    let mouse_pos = ui.io().mouse_pos();

    if rect.contains(mouse_pos) {
        if ui.is_key_pressed(Key::W) {
            scene.set_gizmo_mode(Operation::TRANSLATE);
            changed = true;
        }
        if ui.is_key_pressed(Key::E) {
            scene.set_gizmo_mode(Operation::ROTATE);
            changed = true;
        }
        if ui.is_key_pressed(Key::R) {
            scene.set_gizmo_mode(Operation::SCALE);
            changed = true;
        }
        if ui.is_key_pressed(Key::F) {
            scene.set_snap_enabled(!scene.snap_enabled);
            changed = true;
        }
        if ui.is_key_pressed(Key::L) {
            scene.set_world_mode(!scene.world_mode);
            changed = true;
        }
    }

    if ui.is_mouse_clicked(MouseButton::Left)
        && !ui.io().want_capture_mouse()
        && rect.contains(mouse_pos)
    {
        let view = camera.view();
        let projection = camera.projection(aspect);
        let hit = compute_mouse_ray(&view, &projection, mouse_pos, rect.pos, rect.size)
            .and_then(|ray| pick_object(&scene.objects, &ray));
        let previous = scene.selected;
        scene.select(hit);
        changed |= previous != scene.selected;
    }

    let Some(index) = scene.selected else {
        return changed;
    };
    let mut model = Mat4::from_cols_array_2d(&scene.objects[index].instance.model);
    let view = camera.view();
    let projection = camera.projection(aspect);
    let mut builder = giz
        .manipulate_config(&view, &projection, &mut model)
        .operation(scene.gizmo_mode)
        .mode(if scene.world_mode {
            Mode::World
        } else {
            Mode::Local
        })
        .drawlist(DrawListTarget::Foreground)
        .rect(rect.pos[0], rect.pos[1], rect.size[0], rect.size[1]);
    if scene.snap_enabled {
        builder = builder.snap(scene.snap_values());
    }
    if builder.build() {
        scene.set_transform(index, &model);
        changed = true;
    }

    changed
}
