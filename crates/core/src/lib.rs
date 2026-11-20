//! Foundation types shared by every HBD crate.
//!
//! This crate deliberately has no dependency on graphics, physics, or scripting.

pub mod error;
pub mod handle;
pub mod math;
pub mod time;

pub use error::{HbdError, Result};
pub use handle::{Handle, HandleAllocator};
pub use time::Time;
