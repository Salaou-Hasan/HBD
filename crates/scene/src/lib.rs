//! ECS world and scene representation.
//!
//! `bevy_ecs` is used standalone: no Bevy renderer, no Bevy app. The engine
//! re-exports the parts gameplay code needs and keeps the option to replace the
//! backend behind its own facade.
//!
//! The components here are the scene-side inputs the engine extracts into the
//! renderer each frame. They deliberately carry no Vulkan or ash types: the
//! engine maps them onto the renderer's public inputs.

use glam::{Mat4, Vec3, Vec4};

pub use bevy_ecs::prelude::*;

/// Directional light in the scene. The engine maps this onto the renderer's
/// shadow and geometry uniforms.
#[derive(Component, Clone, Copy, Debug)]
pub struct Light {
    /// Direction from the surface toward the light.
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.35, 0.85, 0.4).normalize(),
            color: Vec3::new(1.0, 0.96, 0.9),
            intensity: 1.0,
        }
    }
}

/// Camera pose in the scene. The engine extracts the first entity with this
/// component into the renderer's camera for the frame.
#[derive(Component, Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: 60.0_f32.to_radians(),
            z_near: 0.1,
            z_far: 1000.0,
        }
    }
}

/// A renderable instance in the scene. Mirrors the GPU instance layout so the
/// engine can map entities onto the renderer's instance data without a second
/// translation layer.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Object {
    pub model: Mat4,
    pub previous_model: Mat4,
    pub base_color: Vec4,
    pub material: Vec4,
}

pub fn new_world() -> World {
    World::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawns_and_despawns_entities() {
        let mut world = new_world();
        let entity = world.spawn_empty().id();

        assert!(world.get_entity(entity).is_ok());

        world.despawn(entity);

        assert!(world.get_entity(entity).is_err());
    }

    #[test]
    fn scene_components_are_queryable() {
        let mut world = new_world();
        let light = world
            .spawn((Light {
                intensity: 1.0,
                ..Default::default()
            },))
            .id();
        let camera = world
            .spawn((Camera {
                position: Vec3::new(0.0, 4.5, 9.0),
                ..Default::default()
            },))
            .id();
        let object = world.spawn((Object::default(),)).id();

        assert_eq!(world.entity(light).get::<Light>().unwrap().intensity, 1.0);
        assert_eq!(
            world.entity(camera).get::<Camera>().unwrap().position,
            Vec3::new(0.0, 4.5, 9.0)
        );
        assert!(world.entity(object).contains::<Object>());
    }
}
