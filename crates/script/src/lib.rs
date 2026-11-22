//! Lua scripting with a sandboxed, curated API.
//!
//! Scripts never see raw ECS internals or the filesystem. Each script runs in
//! its own Lua environment with only `math`, `string`, `table` and a single
//! `hbd` table holding this entity's pose plus `dt`/`time`, and an optional
//! `print` for logging. A script defines `function update(dt)` and mutates the
//! `hbd` tables; the host reads those values back into the entity state.
//!
//! Everything is explicit: loading, hot reload and each script tick are
//! driven by the caller and logged through `tracing`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use hbd_scene::Component;
use mlua::prelude::*;
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum ScriptError {
    #[error("Lua error: {0}")]
    Lua(#[from] mlua::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// ECS component attaching a Lua script file to an entity.
#[derive(Component, Clone, Debug)]
pub struct ScriptComponent {
    pub path: PathBuf,
    pub enabled: bool,
}

impl ScriptComponent {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            enabled: true,
        }
    }
}

/// Pose and appearance exposed to a script. Values are copied in before the
/// tick and read back afterwards, so a script can never mutate the ECS world
/// directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScriptEntityState {
    pub position: [f32; 3],
    pub rotation_degrees: [f32; 3],
    pub scale: [f32; 3],
    pub color: [f32; 3],
}

impl Default for ScriptEntityState {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            rotation_degrees: [0.0; 3],
            scale: [1.0; 3],
            color: [1.0; 3],
        }
    }
}

struct LoadedScript {
    update: mlua::RegistryKey,
    env: mlua::RegistryKey,
    modified: Option<SystemTime>,
}

/// Sandboxed Lua host with a per-script environment and file hot reload.
pub struct ScriptHost {
    lua: Lua,
    scripts: HashMap<PathBuf, LoadedScript>,
}

impl ScriptHost {
    pub fn new() -> Result<Self, ScriptError> {
        let lua = Lua::new();
        {
            let globals = lua.globals();
            // No filesystem, process, debug or module loading from scripts.
            for unsafe_lib in ["io", "os", "debug", "package"] {
                globals.set(unsafe_lib, LuaNil)?;
            }
        }
        info!("Lua script host created (sandboxed: io/os/debug/package removed)");
        Ok(Self {
            lua,
            scripts: HashMap::new(),
        })
    }

    /// Loads or hot-reloads a script if its file changed. Returns true when a
    /// (re)load happened.
    pub fn load_if_changed(&mut self, path: &Path) -> Result<bool, ScriptError> {
        let modified = fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok();
        if let Some(existing) = self.scripts.get(path)
            && existing.modified == modified
        {
            return Ok(false);
        }
        let source = fs::read_to_string(path)?;
        let env = self.create_environment()?;
        let chunk = self.lua.load(&source).set_name(path.to_string_lossy());
        let function: LuaFunction = chunk.into_function()?;
        function.set_environment(env.clone())?;
        function.call::<()>(())?;
        let update = if env.get::<Option<LuaFunction>>("update")?.is_some() {
            self.lua
                .create_registry_value(env.get::<LuaFunction>("update")?)?
        } else {
            self.lua.create_registry_value(LuaNil)?
        };
        let previous = self.scripts.insert(
            path.to_owned(),
            LoadedScript {
                update,
                env: self.lua.create_registry_value(env)?,
                modified,
            },
        );
        info!(
            path = %path.display(),
            reloaded = previous.is_some(),
            "Lua script loaded"
        );
        Ok(true)
    }

    /// Runs one script tick for one entity and returns the mutated state.
    pub fn run(
        &mut self,
        path: &Path,
        state: ScriptEntityState,
        dt: f32,
        time: f32,
    ) -> Result<ScriptEntityState, ScriptError> {
        self.load_if_changed(path)?;
        let Some(entry) = self.scripts.get(path) else {
            return Ok(state);
        };
        let env: LuaTable = self.lua.registry_value(&entry.env)?;
        let hbd: LuaTable = env.get("hbd")?;
        set_vec3(&self.lua, &hbd, "position", state.position)?;
        set_vec3(&self.lua, &hbd, "rotation", state.rotation_degrees)?;
        set_vec3(&self.lua, &hbd, "scale", state.scale)?;
        set_vec3(&self.lua, &hbd, "color", state.color)?;
        hbd.set("dt", dt)?;
        hbd.set("time", time)?;

        let update: LuaValue = self.lua.registry_value(&entry.update)?;
        if let LuaValue::Function(update) = update {
            update.call::<()>(dt)?;
        }

        Ok(ScriptEntityState {
            position: get_vec3(&hbd, "position")?,
            rotation_degrees: get_vec3(&hbd, "rotation")?,
            scale: get_vec3(&hbd, "scale")?,
            color: get_vec3(&hbd, "color")?,
        })
    }

    /// Drops cached chunks for a script path (e.g. after deleting the file).
    pub fn forget(&mut self, path: &Path) {
        if self.scripts.remove(path).is_some() {
            info!(path = %path.display(), "Lua script cache entry removed");
        }
    }

    fn create_environment(&self) -> Result<LuaTable, ScriptError> {
        let env = self.lua.create_table()?;
        env.set("_G", env.clone())?;
        let globals = self.lua.globals();
        for safe_lib in ["math", "string", "table"] {
            env.set(safe_lib, globals.get::<LuaTable>(safe_lib)?)?;
        }
        env.set(
            "print",
            self.lua.create_function(|_, message: String| {
                info!(message, "lua script");
                Ok(())
            })?,
        )?;
        let hbd = self.lua.create_table()?;
        set_vec3(&self.lua, &hbd, "position", [0.0; 3])?;
        set_vec3(&self.lua, &hbd, "rotation", [0.0; 3])?;
        set_vec3(&self.lua, &hbd, "scale", [1.0; 3])?;
        set_vec3(&self.lua, &hbd, "color", [1.0; 3])?;
        hbd.set("dt", 0.0f32)?;
        hbd.set("time", 0.0f32)?;
        env.set("hbd", hbd)?;
        Ok(env)
    }
}

fn set_vec3(lua: &Lua, parent: &LuaTable, key: &str, value: [f32; 3]) -> Result<(), ScriptError> {
    let table = match parent.get::<Option<LuaTable>>(key)? {
        Some(table) => table,
        None => {
            let table = lua.create_table()?;
            parent.set(key, table.clone())?;
            table
        }
    };
    table.set("x", value[0])?;
    table.set("y", value[1])?;
    table.set("z", value[2])?;
    Ok(())
}

fn get_vec3(parent: &LuaTable, key: &str) -> Result<[f32; 3], ScriptError> {
    let table = parent.get::<LuaTable>(key)?;
    Ok([table.get("x")?, table.get("y")?, table.get("z")?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_mutates_entity_state() {
        let dir = std::env::temp_dir().join(format!("hbd-script-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("move.lua");
        fs::write(
            &path,
            "function update(dt) hbd.position.x = hbd.position.x + dt; hbd.rotation.y = 45 end",
        )
        .unwrap();

        let mut host = ScriptHost::new().unwrap();
        let state = host
            .run(&path, ScriptEntityState::default(), 2.0, 0.0)
            .unwrap();

        assert_eq!(state.position[0], 2.0);
        assert_eq!(state.rotation_degrees[1], 45.0);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn unsafe_libraries_are_unavailable() {
        let dir = std::env::temp_dir().join(format!("hbd-script-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unsafe.lua");
        fs::write(&path, "function update(dt) os.execute('echo nope') end").unwrap();

        let mut host = ScriptHost::new().unwrap();
        assert!(
            host.run(&path, ScriptEntityState::default(), 0.0, 0.0)
                .is_err()
        );
        fs::remove_file(&path).unwrap();
    }
}
