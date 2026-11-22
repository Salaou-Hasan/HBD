//! Editable scene model for the HBD editor.
//!
//! Seeds from the engine's default scene (`hbd_engine::render::build_scene`)
//! and pushes edits to the live renderer scene through
//! `Renderer::update_scene_instances`. Every state transition is logged.

use dear_imguizmo::{Operation, decompose_matrix};
use hbd_engine::core::math::Mat4;
use hbd_engine::render::{Instance, build_scene};
use tracing::info;

pub struct EditorObject {
    pub name: String,
    pub instance: Instance,
}

pub struct EditorScene {
    pub objects: Vec<EditorObject>,
    pub selected: Option<usize>,
    pub gizmo_mode: Operation,
    pub world_mode: bool,
    pub snap_enabled: bool,
}

impl EditorScene {
    pub fn new() -> Self {
        let objects = build_scene()
            .into_iter()
            .enumerate()
            .map(|(index, instance)| EditorObject {
                name: format!("Object {index:02}"),
                instance,
            })
            .collect();
        Self {
            objects,
            selected: None,
            gizmo_mode: Operation::TRANSLATE,
            world_mode: false,
            snap_enabled: false,
        }
    }

    pub fn instances(&self) -> Vec<Instance> {
        self.objects.iter().map(|object| object.instance).collect()
    }

    pub fn select(&mut self, index: Option<usize>) {
        if self.selected != index {
            self.selected = index;
            info!(selection = ?index, count = self.objects.len(), "editor selection changed");
        }
    }

    pub fn set_gizmo_mode(&mut self, mode: Operation) {
        if self.gizmo_mode != mode {
            self.gizmo_mode = mode;
            info!(mode = ?mode, "editor gizmo mode changed");
        }
    }

    pub fn set_world_mode(&mut self, world: bool) {
        if self.world_mode != world {
            self.world_mode = world;
            info!(world, "editor gizmo mode (world/local) changed");
        }
    }

    pub fn set_snap_enabled(&mut self, enabled: bool) {
        if self.snap_enabled != enabled {
            self.snap_enabled = enabled;
            info!(enabled, "editor gizmo snap toggled");
        }
    }

    pub fn snap_values(&self) -> [f32; 3] {
        match self.gizmo_mode {
            Operation::ROTATE => [15.0, 0.0, 0.0],
            Operation::SCALE => [0.1, 0.1, 0.1],
            _ => [0.5, 0.5, 0.5],
        }
    }

    pub fn set_transform(&mut self, index: usize, model: &Mat4) {
        let object = &mut self.objects[index];
        object.instance.model = model.to_cols_array_2d();
        object.instance.previous_model = object.instance.model;
        let (translation, rotation, scale) =
            decompose_matrix(&Mat4::from_cols_array_2d(&object.instance.model));
        info!(
            object = index,
            translation = ?translation,
            rotation = ?rotation,
            scale = ?scale,
            "scene object transform changed"
        );
    }

    pub fn set_color(&mut self, index: usize, color: [f32; 3]) {
        let object = &mut self.objects[index];
        object.instance.base_color[0..3].copy_from_slice(&color);
        info!(object = index, color = ?color, "scene object color changed");
    }

    pub fn set_material(&mut self, index: usize, metallic: f32, roughness: f32, ao: f32) {
        let object = &mut self.objects[index];
        object.instance.material[0] = metallic;
        object.instance.material[1] = roughness;
        object.instance.material[2] = ao;
        info!(
            object = index,
            metallic, roughness, ao, "scene object material changed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hbd_engine::core::math::Vec3;

    #[test]
    fn seeds_from_the_engine_default_scene() {
        let scene = EditorScene::new();
        assert!(!scene.objects.is_empty());
        assert!(scene.selected.is_none());
        assert_eq!(scene.objects.len(), build_scene().len());
    }

    #[test]
    fn selection_round_trips_through_the_objects() {
        let mut scene = EditorScene::new();
        scene.select(Some(3));
        assert_eq!(scene.selected, Some(3));
        scene.select(None);
        assert_eq!(scene.selected, None);
    }

    #[test]
    fn transform_round_trips_through_decompose_and_recompose() {
        let mut scene = EditorScene::new();
        let translation = [2.0, -1.5, 0.25];
        let model = Mat4::from_translation(Vec3::from_array(translation));
        scene.set_transform(0, &model);
        let instance = scene.objects[0].instance;
        let (position, _, scale) = decompose_matrix(&Mat4::from_cols_array_2d(&instance.model));
        for axis in 0..3 {
            assert!((position[axis] - translation[axis]).abs() < 1.0e-5);
            assert!((scale[axis] - 1.0).abs() < 1.0e-5);
        }
        assert_eq!(scene.objects[0].instance.previous_model, instance.model);
    }

    #[test]
    fn snap_values_follow_the_gizmo_mode() {
        let mut scene = EditorScene::new();
        assert_eq!(scene.snap_values(), [0.5, 0.5, 0.5]);
        scene.set_gizmo_mode(Operation::ROTATE);
        assert_eq!(scene.snap_values(), [15.0, 0.0, 0.0]);
        scene.set_snap_enabled(true);
        scene.set_snap_enabled(false);
        scene.set_world_mode(true);
        scene.set_world_mode(false);
    }
}
