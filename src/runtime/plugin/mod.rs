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
mod shade;
mod types;

pub use items::{ParticleItem, ParticleItems};
pub use types::{ParticlePlugin, ParticleStats, ParticleStatsHandle};

// Crate-internal re-exports so the submodules reach each other's items through
// `use super::*`.
pub(crate) use build::*;
pub(crate) use items::EffectCommand;
pub(crate) use shade::*;
pub(crate) use types::*;

/// Sample count for the plugin's single-sampled interaction passes (shadow,
/// pick, outline-mask). The library pegs those pass targets to 1 regardless of
/// the scene MSAA level (see [`SharedBindings::sample_count`]), so the plugin uses
/// 1 there. The HDR colour passes instead build against `self.sample_count`, read
/// from `SharedBindings.sample_count` in `build_gpu`.
pub(crate) const INTERACTION_SAMPLE_COUNT: u32 = 1;

/// High bit of an `InstanceKey`'s `key`, reserved for internal transient (burst)
/// instances so they never collide with host-supplied keys.
pub(crate) const TRANSIENT_KEY_BIT: u64 = 1 << 63;

/// Default cap on concurrent transient (burst) instances per effect.
pub(crate) const DEFAULT_MAX_TRANSIENTS: usize = 64;

/// Max modifiers per effect, matching the fixed array in the simulate shaders.
/// Bump this and the array size in `MODIFIERS_WGSL` together to allow more.
pub(crate) const MAX_MODIFIERS: usize = 8;

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

/// Number of samples baked into a `MeshSurface` / `PointCloud` spawn pool
/// (Phase 22). The emit kernel picks a random index into this pool per spawn.
pub(crate) const MESH_SAMPLE_POOL: u32 = 8192;

/// Mappable readback buffers per instance for the live-count stats ring. Enough
/// to pipeline the copy-this-frame, map-next-frame cycle without a copy ever
/// aliasing an in-flight map.
pub(crate) const STATS_RING: usize = 3;

/// Idle-GC threshold (issue 013): a steady instance whose `(effect, key)` has
/// not been submitted for this many frames is reclaimed from `self.instances`,
/// so the instance map stays bounded to the live working set across long
/// sessions with many scene reloads. Submission (not simulation) resets the
/// clock, so a paused, culled, or budget-frozen instance the host still submits
/// is never reaped. Transient (pooled) and sub-emitter instances are exempt.
pub(crate) const IDLE_RETIRE_FRAMES: u64 = 120;

/// Upper bound on the number of sub-steps a prewarm (issue 015) runs on an
/// instance's first activation. Warmup time is divided into at most this many
/// steps (each its own tiny command buffer), so a large warmup stays a bounded
/// one-time cost even when many instances activate on the same frame.
pub(crate) const MAX_PREWARM_STEPS: u32 = 120;

/// Fixed number of texture slots a custom fragment hook (Phase 33 / issue 018)
/// can bind. The group-3 hook layout always declares this many (unused slots bind
/// the shared 1x1 white default), so every hooked billboard pipeline shares one
/// layout and pipelines cache purely by composed-source hash.
pub(crate) const HOOK_TEXTURES: usize = 4;

/// Number of `vec4` lanes in a fragment hook's params uniform. Lane 0 is engine-
/// reserved (`.w` carries `time`); lanes 1.. are the effect's own params.
pub(crate) const HOOK_PARAM_LANES: usize = 8;
