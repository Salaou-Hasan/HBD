//! Lua scripting.
//!
//! Scripts receive a curated engine API, not raw ECS internals. Mutations are
//! queued and applied at the end of the script schedule.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScriptLanguage {
    Lua54,
    Luau,
    LuaJit,
}

#[derive(Debug, Clone)]
pub struct ScriptConfig {
    pub language: ScriptLanguage,
    pub sandboxed: bool,
    pub hot_reload: bool,
}

impl Default for ScriptConfig {
    fn default() -> Self {
        Self {
            language: ScriptLanguage::Lua54,
            sandboxed: true,
            hot_reload: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipping_defaults_are_sandboxed() {
        let config = ScriptConfig::default();
        assert_eq!(config.language, ScriptLanguage::Lua54);
        assert!(config.sandboxed);
        assert!(!config.hot_reload);
    }
}
