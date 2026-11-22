//! ECS-backed editable scene model for the HBD editor.
//!
//! The authoritative scene is the engine's `EcsScene` world: objects, camera
//! and light are entities with `Object`/`Camera`/`Light` components. The editor
//! reads and writes those components, extracts one frame explicitly, and
//! pushes the resulting instances to the live renderer path. Lua scripts run
//! through the engine's sandboxed `ScriptHost` before extraction.
//! Every state change is logged.

use dear_imguizmo::{Operation, decompose_matrix, recompose_matrix};
use hbd_engine::core::math::{Mat4, Vec3, Vec4};
use hbd_engine::ecs::{EcsScene, SceneFrame};
use hbd_engine::render::{Instance, build_scene};
use hbd_engine::scene::{Camera as EcsCamera, Entity, Light as EcsLight, Name, Object, With};
use hbd_engine::script::{ScriptComponent, ScriptEntityState, ScriptError, ScriptHost};
use tracing::{info, warn};

pub struct EditorScene {
    ecs: EcsScene,
    frame: SceneFrame,
    object_entities: Vec<Entity>,
    selected: Option<Entity>,
    camera_entity: Entity,
    light_entity: Entity,
    pub gizmo_mode: Operation,
    pub world_mode: bool,
    pub snap_enabled: bool,
    pub scripts_running: bool,
    host: ScriptHost,
    elapsed: f32,
}

impl EditorScene {
    pub fn new() -> Result<Self, ScriptError> {
        let mut ecs = EcsScene::new();
        let camera_entity = ecs
            .world()
            .spawn((
                EcsCamera {
                    position: Vec3::new(9.0, 4.5, 0.0),
                    target: Vec3::ZERO,
                    ..Default::default()
                },
                Name::new("Camera"),
            ))
            .id();
        let light_entity = ecs
            .world()
            .spawn((EcsLight::default(), Name::new("Directional Light")))
            .id();
        let mut object_entities = Vec::new();
        for (index, instance) in build_scene().into_iter().enumerate() {
            let object = Object {
                model: Mat4::from_cols_array_2d(&instance.model),
                previous_model: Mat4::from_cols_array_2d(&instance.previous_model),
                base_color: Vec4::from_array(instance.base_color),
                material: Vec4::from_array(instance.material),
            };
            let entity = ecs
                .world()
                .spawn((object, Name::new(format!("Object {index:02}"))))
                .id();
            object_entities.push(entity);
        }
        info!(
            objects = object_entities.len(),
            ?camera_entity,
            ?light_entity,
            "ECS editor scene seeded from build_scene()"
        );
        Ok(Self {
            frame: SceneFrame::default(),
            ecs,
            object_entities,
            selected: None,
            camera_entity,
            light_entity,
            gizmo_mode: Operation::TRANSLATE,
            world_mode: false,
            snap_enabled: false,
            scripts_running: false,
            host: ScriptHost::new()?,
            elapsed: 0.0,
        })
    }

    /// Runs attached scripts, then the engine's explicit ECS extraction
    /// schedule. Call once per frame before reading camera/light/instances.
    pub fn tick(&mut self, dt: f32) {
        self.elapsed += dt;
        self.run_scripts(dt);
        self.frame = self.ecs.run();
    }

    pub fn camera(&self) -> Option<hbd_engine::render::Camera> {
        self.frame.camera
    }

    pub fn light(&self) -> Option<hbd_engine::render::DirectionalLight> {
        self.frame.light
    }

    /// Re-extracts the ECS frame and returns render instances. UI edits happen
    /// after the previous tick, so the extraction must be refreshed before
    /// pushing instances to the renderer.
    pub fn instances(&mut self) -> Vec<Instance> {
        self.frame = self.ecs.run();
        self.frame.instances()
    }

    pub fn object_entities(&self) -> &[Entity] {
        &self.object_entities
    }

    pub fn name(&self, entity: Entity) -> String {
        self.ecs
            .world_ref()
            .entity(entity)
            .get::<Name>()
            .map_or_else(|| "unnamed".to_owned(), |name| name.0.clone())
    }

    pub fn script_path(&self, entity: Entity) -> Option<std::path::PathBuf> {
        self.ecs
            .world_ref()
            .entity(entity)
            .get::<ScriptComponent>()
            .map(|script| script.path.clone())
    }

    pub fn script_enabled(&self, entity: Entity) -> bool {
        self.ecs
            .world_ref()
            .entity(entity)
            .get::<ScriptComponent>()
            .is_some_and(|script| script.enabled)
    }

    pub fn selected(&self) -> Option<Entity> {
        self.selected
    }

    pub fn select(&mut self, entity: Option<Entity>) {
        if self.selected != entity {
            self.selected = entity;
            info!(
                ?entity,
                count = self.object_entities.len(),
                "editor selection changed"
            );
        }
    }

    pub fn set_gizmo_mode(&mut self, mode: Operation) {
        if self.gizmo_mode != mode {
            self.gizmo_mode = mode;
            info!(?mode, "editor gizmo mode changed");
        }
    }

    pub fn set_world_mode(&mut self, world: bool) {
        if self.world_mode != world {
            self.world_mode = world;
            info!(world, "editor gizmo space changed");
        }
    }

    pub fn set_snap_enabled(&mut self, enabled: bool) {
        if self.snap_enabled != enabled {
            self.snap_enabled = enabled;
            info!(enabled, "editor gizmo snap toggled");
        }
    }

    pub fn set_scripts_running(&mut self, running: bool) {
        if self.scripts_running != running {
            self.scripts_running = running;
            info!(running, "editor script execution toggled");
        }
    }

    pub fn snap_values(&self) -> [f32; 3] {
        match self.gizmo_mode {
            Operation::ROTATE => [15.0, 0.0, 0.0],
            Operation::SCALE => [0.1, 0.1, 0.1],
            _ => [0.5, 0.5, 0.5],
        }
    }

    /// Instance data for viewport picking, keyed by ECS entity.
    pub fn pick_candidates(&self) -> Vec<(Entity, Instance)> {
        self.object_entities
            .iter()
            .filter_map(|entity| {
                self.ecs
                    .world_ref()
                    .entity(*entity)
                    .get::<Object>()
                    .map(|object| {
                        (
                            *entity,
                            Instance {
                                model: object.model.to_cols_array_2d(),
                                previous_model: object.previous_model.to_cols_array_2d(),
                                base_color: object.base_color.to_array(),
                                material: object.material.to_array(),
                            },
                        )
                    })
            })
            .collect()
    }

    pub fn object(&self, entity: Entity) -> Option<Object> {
        self.ecs.world_ref().entity(entity).get::<Object>().copied()
    }

    pub fn set_transform(&mut self, entity: Entity, model: &Mat4) {
        if let Some(mut object) = self.ecs.world().get_mut::<Object>(entity) {
            object.model = *model;
            object.previous_model = *model;
            let (translation, rotation, scale) = decompose_matrix(model);
            info!(
                ?entity,
                translation = ?translation,
                rotation = ?rotation,
                scale = ?scale,
                "ECS object transform changed"
            );
        }
    }

    pub fn set_color(&mut self, entity: Entity, color: [f32; 3]) {
        if let Some(mut object) = self.ecs.world().get_mut::<Object>(entity) {
            object.base_color.x = color[0];
            object.base_color.y = color[1];
            object.base_color.z = color[2];
            info!(?entity, ?color, "ECS object color changed");
        }
    }

    pub fn set_material(&mut self, entity: Entity, metallic: f32, roughness: f32, ao: f32) {
        if let Some(mut object) = self.ecs.world().get_mut::<Object>(entity) {
            object.material.x = metallic;
            object.material.y = roughness;
            object.material.z = ao;
            info!(
                ?entity,
                metallic, roughness, ao, "ECS object material changed"
            );
        }
    }

    pub fn attach_script(&mut self, entity: Entity, path: std::path::PathBuf) {
        if let Some(mut script) = self.ecs.world().get_mut::<ScriptComponent>(entity) {
            script.path = path.clone();
            script.enabled = true;
        } else {
            self.ecs
                .world()
                .entity_mut(entity)
                .insert(ScriptComponent::new(path.clone()));
        }
        info!(?entity, path = %path.display(), "Lua script attached to ECS entity");
    }

    pub fn detach_script(&mut self, entity: Entity) {
        if let Some(path) = self.script_path(entity) {
            self.host.forget(&path);
        }
        self.ecs
            .world()
            .entity_mut(entity)
            .remove::<ScriptComponent>();
        info!(?entity, "Lua script detached from ECS entity");
    }

    pub fn toggle_script(&mut self, entity: Entity, enabled: bool) {
        if let Some(mut script) = self.ecs.world().get_mut::<ScriptComponent>(entity) {
            script.enabled = enabled;
            info!(?entity, enabled, "Lua script enabled state changed");
        }
    }

    pub fn light_intensity(&self) -> f32 {
        self.ecs
            .world_ref()
            .entity(self.light_entity)
            .get::<EcsLight>()
            .map_or(1.0, |light| light.intensity)
    }

    pub fn set_light_intensity(&mut self, intensity: f32) {
        if let Some(mut light) = self.ecs.world().get_mut::<EcsLight>(self.light_entity) {
            light.intensity = intensity;
        }
        info!(intensity, "ECS light intensity changed");
    }

    /// Moves the scene camera entity (used by Reset Camera).
    pub fn reset_camera(&mut self) {
        if let Some(mut camera) = self.ecs.world().get_mut::<EcsCamera>(self.camera_entity) {
            *camera = EcsCamera {
                position: Vec3::new(9.0, 4.5, 0.0),
                target: Vec3::ZERO,
                ..Default::default()
            };
        }
        info!(?self.camera_entity, "editor camera reset to default");
    }

    fn run_scripts(&mut self, dt: f32) {
        if !self.scripts_running {
            return;
        }
        let entities: Vec<Entity> = {
            let mut query = self
                .ecs
                .world()
                .query_filtered::<Entity, With<ScriptComponent>>();
            query.iter(self.ecs.world_ref()).collect()
        };
        for entity in entities {
            let Some(path) = self.script_path(entity) else {
                continue;
            };
            if !self.script_enabled(entity) {
                continue;
            }
            let state = {
                let world = self.ecs.world_ref();
                let Some(object) = world.entity(entity).get::<Object>() else {
                    continue;
                };
                state_from_object(object)
            };
            match self.host.run(&path, state, dt, self.elapsed) {
                Ok(updated) => {
                    if let Some(mut object) = self.ecs.world().get_mut::<Object>(entity) {
                        apply_state(&mut object, &updated);
                    }
                }
                Err(error) => {
                    warn!(?entity, path = %path.display(), %error, "Lua script tick failed")
                }
            }
        }
    }
}

fn state_from_object(object: &Object) -> ScriptEntityState {
    let (position, rotation_degrees, scale) = decompose_matrix(&object.model);
    ScriptEntityState {
        position,
        rotation_degrees,
        scale,
        color: [
            object.base_color.x,
            object.base_color.y,
            object.base_color.z,
        ],
    }
}

fn apply_state(object: &mut Object, state: &ScriptEntityState) {
    let model = recompose_matrix(&state.position, &state.rotation_degrees, &state.scale);
    object.model = model;
    object.previous_model = model;
    object.base_color.x = state.color[0];
    object.base_color.y = state.color[1];
    object.base_color.z = state.color[2];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_ecs_world_with_camera_light_and_objects() {
        let mut scene = EditorScene::new().unwrap();
        assert_eq!(scene.object_entities().len(), build_scene().len());
        assert_eq!(scene.camera_entity, scene.camera_entity);
        scene.tick(0.0);
        assert!(scene.camera().is_some());
        assert!(scene.light().is_some());
        assert_eq!(scene.instances().len(), build_scene().len());
    }

    #[test]
    fn editor_edits_flow_through_ecs_components() {
        let mut scene = EditorScene::new().unwrap();
        let entity = scene.object_entities()[0];
        scene.select(Some(entity));
        assert_eq!(scene.selected(), Some(entity));
        let model = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        scene.set_transform(entity, &model);
        scene.set_color(entity, [0.1, 0.2, 0.3]);
        scene.set_material(entity, 0.5, 0.6, 0.7);
        let object = scene.object(entity).unwrap();
        assert_eq!(object.model, model);
        assert_eq!(object.base_color.x, 0.1);
        assert_eq!(object.material.y, 0.6);
    }

    #[test]
    fn lua_script_tick_moves_ecs_object() {
        let mut scene = EditorScene::new().unwrap();
        let entity = scene.object_entities()[0];
        let dir = std::env::temp_dir().join(format!("hbd-editor-script-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rotate.lua");
        std::fs::write(
            &path,
            "function update(dt) hbd.rotation.y = hbd.rotation.y + dt * 90 end",
        )
        .unwrap();
        scene.attach_script(entity, path.clone());
        scene.set_scripts_running(true);

        let before = scene.object(entity).unwrap().model;
        scene.tick(1.0);
        let after = scene.object(entity).unwrap().model;
        assert_ne!(before, after);

        scene.set_scripts_running(false);
        scene.detach_script(entity);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn script_attachment_and_toggle_are_component_state() {
        let mut scene = EditorScene::new().unwrap();
        let entity = scene.object_entities()[0];
        let path = std::path::PathBuf::from("assets/scripts/spin.lua");
        scene.attach_script(entity, path.clone());
        assert_eq!(scene.script_path(entity), Some(path));
        assert!(scene.script_enabled(entity));
        scene.toggle_script(entity, false);
        assert!(!scene.script_enabled(entity));
        scene.detach_script(entity);
        assert_eq!(scene.script_path(entity), None);
    }
}
