//! The CPU-side authoring types: the effect asset model, the expression IR,
//! the expression-to-WGSL compiler, and the modifier registry. No GPU state.

#[cfg(feature = "gpu")]
pub(crate) mod codegen;
pub(crate) mod effect;
pub(crate) mod expr;
pub(crate) mod registry;
