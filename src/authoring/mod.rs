//! The CPU-side authoring surface: the effect asset model, the expression IR,
//! the expression→WGSL compiler, and the modifier registry. No GPU state.

pub(crate) mod codegen;
pub(crate) mod effect;
pub(crate) mod expr;
pub(crate) mod registry;
