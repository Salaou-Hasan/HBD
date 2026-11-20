//! Re-export of the engine math types.
//!
//! Gameplay and renderer code should import from here, not from `glam`
//! directly. That keeps the math backend replaceable at the facade level.

pub use glam::*;
