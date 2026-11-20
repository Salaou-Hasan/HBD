//! ECS scene bridge.
//!
//! An [`EcsScene`] owns an ECS world and runs an explicit schedule that extracts
//! this frame's render inputs. The renderer stays backend-agnostic: it receives
//! the same public `Camera`/`DirectionalLight` values whether they came from an
//! ECS entity, a script, or a tool. Every extraction is driven by [`EcsScene::run`]
//! and is therefore visible to the caller — no hidden frame work.

use bevy_ecs::prelude::*;

use crate::render::{Camera, DirectionalLight};
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
}
