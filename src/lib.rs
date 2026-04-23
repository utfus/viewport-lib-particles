//! GPU particle and VFX system for `viewport-lib`.
//!
//! This crate implements a particle system as an
//! [`ItemTypePlugin`](viewport_lib::plugin_api::ItemTypePlugin): particles are
//! a scene item category that owns its own compute pipelines (emit + simulate)
//! and draws inside the lib's HDR scene pass, so they pick up scene lighting,
//! shadows, clip planes, and the outline/pick passes through the shared group-0
//! bindings.
//!
//! Today an [`EffectAsset`] is fixed-function: a concrete [`Emitter`] (spawn
//! rate, shape, velocity, lifetime, colour, size) plus a list of
//! [`ForceModifier`]s. The plugin runs it as an emit compute pass (recycle dead
//! slots into new particles), a simulate compute pass (integrate forces, age
//! particles), and a billboard draw, all sourcing from a persistent per-effect
//! GPU buffer. No per-particle work happens on the CPU.
//!
//! The design target is the authoring power of a modern GPU particle system:
//! per-attribute logic built from an expression graph ([`Module`], [`Expr`])
//! that compiles to the emit and simulate kernels. That codegen path is the next
//! phase; the [`Module`] / [`Expr`] types are present but not yet wired into the
//! shaders.
//!
//! Status: early. Emit + simulate + draw work (verified by
//! `tests/simulates.rs`); the expression/modifier codegen does not exist yet.
//! See `docs/plans/particle-system-plan.md` for the phased build-out.
//!
//! Typical setup:
//!
//! ```ignore
//! use viewport_lib_particles::{EffectAsset, ParticleItem, ParticleItems, ParticlePlugin};
//!
//! // Register the plugin and its effects once, at startup.
//! let mut plugin = ParticlePlugin::new();
//! let effect = plugin.add_effect(&device, EffectAsset::default());
//! renderer.with_item_type_plugin(&device, Box::new(plugin));
//!
//! // Each frame, submit live effect instances under the plugin's type name.
//! let mut items = ParticleItems::new().with_dt(dt);
//! items.push(ParticleItem::new(effect).at([0.0, 0.0, 0.0]));
//! frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
//! ```

mod codegen;
mod effect;
mod expr;
mod plugin;

pub use effect::{
    EffectAsset, EffectId, Emitter, ForceModifier, ParticleBlend, SpawnRate, SpawnShape,
    VelocityDist,
};
pub use expr::{Expr, ExprHandle, Module};
pub use plugin::{ParticleItem, ParticleItems, ParticlePlugin};

/// Catalogue version of `viewport-lib` this crate was built against.
///
/// Bump in lockstep with `viewport_lib::plugin_api::shared_wgsl::WGSL_VERSION`
/// whenever the shared-WGSL contract changes. The compile-time assertion below
/// catches accidental drift so a stale shared-binding layout is a build error,
/// not a runtime mismatch.
pub const VIEWPORT_LIB_WGSL_VERSION: u32 = 6;

const _: () = assert!(
    viewport_lib::plugin_api::shared_wgsl::WGSL_VERSION == VIEWPORT_LIB_WGSL_VERSION,
    "viewport-lib catalogue version drifted; update VIEWPORT_LIB_WGSL_VERSION and review the particle shaders",
);
