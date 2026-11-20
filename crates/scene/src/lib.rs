//! ECS world and scene representation.
//!
//! `bevy_ecs` is used standalone: no Bevy renderer, no Bevy app. The engine
//! re-exports the parts gameplay code needs and keeps the option to replace the
//! backend behind its own facade.

pub use bevy_ecs::prelude::*;

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
}
