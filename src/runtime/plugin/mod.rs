//! The particle [`ItemTypePlugin`] and its per-frame item collection.
//!
//! Split across submodules: [`items`] (the public `ParticleItem` / `ParticleItems`
//! API), [`types`] (`ParticlePlugin` and the internal GPU / state structs),
//! [`lifecycle`] (registration, hot-reload, and GPU-state construction),
//! [`prepare`] (the per-frame simulate orchestration and the draw passes), and
//! [`build`] (GPU-resource and uniform builders).

mod build;
mod items;
mod lifecycle;
mod prepare;
mod types;

pub use items::{ParticleItem, ParticleItems};
pub use types::ParticlePlugin;

// Crate-internal re-exports so the submodules reach each other's items through
// `use super::*`.
pub(crate) use build::*;
pub(crate) use items::EffectCommand;
pub(crate) use types::*;

/// MSAA sample count the plugin builds its pipelines for. A plugin cannot read
/// the renderer's configured sample count from `SharedBindings` today, so this
/// assumes the default single-sampled HDR pass (see the plan).
pub(crate) const SAMPLE_COUNT: u32 = 1;

/// High bit of an `InstanceKey`'s `key`, reserved for internal transient (burst)
/// instances so they never collide with host-supplied keys.
pub(crate) const TRANSIENT_KEY_BIT: u64 = 1 << 63;

/// Default cap on concurrent transient (burst) instances per effect.
pub(crate) const DEFAULT_MAX_TRANSIENTS: usize = 64;

/// Max forces per effect, matching the fixed array in `particle_sim.wgsl`.
pub(crate) const MAX_FORCES: usize = 8;

/// Trail history samples kept per particle. Matches `HISTORY_LEN` usage in the
/// trail shaders.
pub(crate) const HISTORY_LEN: u32 = 24;

/// Bytes of the event buffer header (an atomic count plus padding to the 16-byte
/// alignment the `SpawnEvent` array needs).
pub(crate) const EVENT_HEADER_BYTES: u64 = 16;

/// Minimum dynamic-uniform-offset alignment (wgpu default), the stride between
/// the per-stage `SortStageParams` in the shared stage buffer.
pub(crate) const SORT_STAGE_STRIDE: u64 = 256;

/// Width of the lifetime-ramp LUT texture in texels.
pub(crate) const RAMP_WIDTH: u32 = 64;
