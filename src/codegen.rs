//! WGSL codegen for an effect's emit and simulate kernels.
//!
//! Each [`EffectAsset`](crate::EffectAsset) compiles to two compute shaders: an
//! emit kernel that recycles dead slots and runs the init modifiers, and a
//! simulate kernel that sums the force modifiers, integrates, and decrements
//! lifetime. The reachable part of the effect's expression
//! [`Module`](crate::Module) lowers into WGSL that both kernels share.
//!
//! The generated bodies splice into a fixed harness (bindings, workgroup setup,
//! the particle struct) so only the modifier logic differs per effect. Compiled
//! pipelines are cached by the effect's structural signature: two effects whose
//! modifier stacks lower to identical WGSL reuse one pipeline.
//!
//! Status: scaffolding for the codegen phase. Nothing calls [`generate`] yet;
//! Phase 2 runs a fixed-function emit/simulate pair instead. Kept here as the
//! shape the modifier lowering will grow into.
#![allow(dead_code)]

use crate::effect::EffectAsset;

/// The WGSL source for one compiled effect: the emit and simulate kernels plus
/// any shared helpers.
#[derive(Clone, Debug)]
#[allow(dead_code)] // consumed once the emit/simulate pipelines are built
pub(crate) struct GeneratedShaders {
    /// Compute shader source for the emit pass.
    pub emit: String,
    /// Compute shader source for the simulate pass.
    pub simulate: String,
}

/// The fixed harness spliced around every effect's generated bodies. The
/// `// {{init}}` and `// {{update}}` markers are where lowered modifier code is
/// inserted once codegen is implemented.
const HARNESS: &str = include_str!("shaders/particle_harness.wgsl");

/// Lower an effect to its emit and simulate WGSL.
///
/// Currently returns the harness with empty modifier hooks. Threading the real
/// modifier and expression lowering through here is the codegen work.
pub(crate) fn generate(_effect: &EffectAsset) -> GeneratedShaders {
    // Placeholder: both kernels share the harness until per-pass lowering
    // splits them. The plan's codegen phase replaces this with real emit /
    // simulate bodies derived from the modifier stacks.
    GeneratedShaders {
        emit: HARNESS.to_string(),
        simulate: HARNESS.to_string(),
    }
}
