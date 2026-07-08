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
//! For richer effects, an [`EffectAsset`] can instead carry an
//! [`EffectProgram`]: per-attribute logic built from an expression graph
//! ([`Module`], [`Expr`]) plus init and update modifiers, which
//! [`crate::codegen`] lowers to the effect's own emit and simulate WGSL kernels.
//! Compiled pipelines are cached by the hash of their generated source.
//!
//! Colour and size ramps over lifetime attach to either path via
//! [`EffectAsset::with_gradient`] (a LUT sampled by particle age).
//!
//! Particles draw as camera-facing billboards (optionally velocity-stretched),
//! as instances of an uploaded mesh, or as ribbon trails swept through each
//! particle's recorded path, chosen by [`EffectAsset::with_render`].
//!
//! Screen-space effects that bend the scene behind particles (shockwaves, heat
//! haze) cannot be done in the particle draw, which has no readable copy of the
//! scene colour. Those live on the other seam `viewport-lib` exposes: see
//! [`RefractionPlugin`], a companion `GpuPlugin` that distorts the rendered
//! colour in a `post_paint` pass.
//!
//! Status: early. Both authoring paths work -- fixed-function (verified by
//! `tests/simulates.rs`) and codegen (verified by `tests/expression.rs`) -- with
//! lifetime gradients (`tests/gradient.rs`) and the billboard, mesh, and trail
//! render routes (`tests/render_routes.rs`, `tests/trails.rs`). Non-additive
//! effects depth-sort back-to-front (`tests/sort.rs`), and the systems are
//! GPU-pickable and cast shadows (`tests/pick.rs`, `tests/shadow.rs`). Soft
//! particles are the main feature still ahead; see
//! `docs/plans/particle-system-plan.md` for the phased build-out.
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
mod refraction;

pub use effect::{
    Attribute, EffectAsset, EffectId, EffectProgram, Emitter, Flipbook, ForceModifier, Gradient,
    MeshAlign, ParticleBlend, ParticleMeshId, ParticleRender, ParticleTextureId, PropertyDecl,
    PropertyValue, SetAttribute, SpawnRate, SpawnShape, TextureMode, UpdateOp, VelocityDist,
};
pub use expr::{Expr, ExprHandle, Module};
pub use plugin::{ParticleItem, ParticleItems, ParticlePlugin};
pub use refraction::RefractionPlugin;

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
