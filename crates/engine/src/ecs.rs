//! ECS scene bridge.
//!
//! An [`EcsScene`] owns an ECS world and runs an explicit schedule that extracts
//! this frame's render inputs. The renderer stays backend-agnostic: it receives
//! the same public `Camera`/`DirectionalLight` values whether they came from an
//! ECS entity, a script, or a tool. Every extraction is driven by [`EcsScene::run`]
//! and is therefore visible to the caller — no hidden frame work.

use bevy_ecs::prelude::*;

use crate::render::{Camera, DirectionalLight, Instance};
use crate::scene::{Camera as EcsCamera, Light, Object};

/// Per-frame render inputs extracted from the ECS world.
#[derive(Debug, Clone, Default)]
pub struct SceneFrame {
    pub camera: Option<Camera>,
    pub light: Option<DirectionalLight>,
    pub objects: Vec<Object>,
}

/// World-side resource the extraction system writes into. Kept in the world so
/// the schedule owns the write; `EcsScene::run` copies it into a `SceneFrame`.
#[derive(Resource, Debug, Clone, Default)]
pub struct SceneState {
    pub camera: Option<Camera>,
    pub light: Option<DirectionalLight>,
    pub objects: Vec<Object>,
}

/// Maps scene entities to the render instance layout. `Object` and
/// `Instance` share the same field layout, so this is a lossless copy of the
/// poses the renderer draws.
pub fn instances_from_objects(objects: &[Object]) -> Vec<Instance> {
    objects
        .iter()
        .map(|object| Instance {
            model: object.model.to_cols_array_2d(),
            previous_model: object.previous_model.to_cols_array_2d(),
            base_color: object.base_color.to_array(),
            material: object.material.to_array(),
        })
        .collect()
}

/// Owns the ECS scene and the schedule that extracts render inputs from it.
pub struct EcsScene {
    world: World,
    schedule: Schedule,
}

impl EcsScene {
    pub fn new() -> Self {
        let mut schedule = Schedule::default();
        schedule.add_systems(extract_scene);
        let mut world = World::new();
        world.insert_resource(SceneState::default());
        Self { world, schedule }
    }

    /// Direct ECS access for spawning and mutating scene entities.
    pub fn world(&mut self) -> &mut World {
        &mut self.world
    }

    /// Read-only ECS access for tools that only inspect entities.
    pub fn world_ref(&self) -> &World {
        &self.world
    }

    /// Runs the scene schedule explicitly and returns the extracted inputs.
    pub fn run(&mut self) -> SceneFrame {
        self.schedule.run(&mut self.world);
        let state = self.world.resource::<SceneState>();
        SceneFrame {
            camera: state.camera,
            light: state.light,
            objects: state.objects.clone(),
        }
    }
}

impl Default for EcsScene {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneFrame {
    /// Scene instances in render form; used by the shadow-pass bridge.
    pub fn instances(&self) -> Vec<Instance> {
        instances_from_objects(&self.objects)
    }
}

/// The one scene-extraction system: first camera, first light, all objects.
/// Multiple cameras or lights are not merged; spawn exactly one of each.
fn extract_scene(
    cameras: Query<&EcsCamera>,
    lights: Query<&Light>,
    objects: Query<&Object>,
    mut state: ResMut<SceneState>,
) {
    state.camera = cameras.iter().next().map(|camera| Camera {
        position: camera.position,
        target: camera.target,
        up: camera.up,
        fov_y_radians: camera.fov_y_radians,
        z_near: camera.z_near,
        z_far: camera.z_far,
    });
    state.light = lights.iter().next().map(|light| DirectionalLight {
        direction: light.direction,
        color: light.color,
        intensity: light.intensity,
    });
    state.objects = objects.iter().copied().collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_camera_light_and_objects() {
        let mut scene = EcsScene::new();
        scene.world().spawn((EcsCamera::default(),));
        scene.world().spawn((Light::default(),));
        scene.world().spawn((Object::default(),));
        scene.world().spawn((Object::default(),));

        let frame = scene.run();

        assert!(frame.camera.is_some());
        assert!(frame.light.is_some());
        assert_eq!(frame.objects.len(), 2);
    }

    #[test]
    fn empty_world_extracts_nothing() {
        let frame = EcsScene::new().run();
        assert!(frame.camera.is_none());
        assert!(frame.light.is_none());
        assert!(frame.objects.is_empty());
    }

    #[test]
    fn instances_from_objects_maps_pose_color_and_material() {
        use crate::core::math::{Mat4, Vec3, Vec4};
        let object = Object {
            model: Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            previous_model: Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0)),
            base_color: Vec4::new(0.1, 0.2, 0.3, 1.0),
            material: Vec4::new(0.5, 0.6, 0.7, 0.8),
        };
        let instances = instances_from_objects(&[object]);
        assert_eq!(instances.len(), 1);
        let instance = &instances[0];
        assert_eq!(instance.model[3][0], 1.0);
        assert_eq!(instance.model[3][1], 2.0);
        assert_eq!(instance.model[3][2], 3.0);
        assert_eq!(instance.previous_model[3][0], 4.0);
        assert_eq!(instance.previous_model[3][1], 5.0);
        assert_eq!(instance.base_color, [0.1, 0.2, 0.3, 1.0]);
        assert_eq!(instance.material, [0.5, 0.6, 0.7, 0.8]);

        let mut scene = EcsScene::new();
        scene.world().spawn((object,));
        let frame = scene.run();
        assert_eq!(frame.instances().len(), 1);
    }
}
