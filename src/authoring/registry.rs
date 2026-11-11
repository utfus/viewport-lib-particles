//! A name -> factory registry for reconstructing modifiers and effect templates
//! by string.
//!
//! This is the non-reflection analogue of Hanabi's reflection-based modifier
//! registry. The built-in modifier set ([`ForceModifier`], the spawn/velocity
//! shapes, the expression [`Module`](crate::Module)) is a closed enum and
//! round-trips through serde directly, so it needs no registry. The registry
//! exists for the open-ended case: host code or an authoring tool
//! reconstructing *named* modifiers or whole effect templates on load, without
//! hard-coding the concrete Rust constructor at every call site.
//!
//! Register factories once at startup, then look them up by the name that a
//! loaded `.effect` file (or UI) refers to.

use std::collections::HashMap;

use crate::effect::{EffectAsset, ForceModifier};

/// Builds a [`ForceModifier`] on demand.
type ForceFactory = Box<dyn Fn() -> ForceModifier + Send + Sync>;
/// Builds an [`EffectAsset`] on demand.
type EffectFactory = Box<dyn Fn() -> EffectAsset + Send + Sync>;

/// A registry of named force and effect factories.
///
/// ```
/// use viewport_lib_particles::{EffectAsset, ForceModifier, ModifierRegistry};
///
/// let mut reg = ModifierRegistry::new();
/// reg.register_force("gravity", || ForceModifier::Accel([0.0, 0.0, -9.81]));
/// reg.register_effect("spark", || EffectAsset::new("spark"));
///
/// let g = reg.force("gravity").unwrap();
/// assert!(matches!(g, ForceModifier::Accel(_)));
/// assert!(reg.effect("spark").is_some());
/// assert!(reg.force("missing").is_none());
/// ```
#[derive(Default)]
pub struct ModifierRegistry {
    forces: HashMap<String, ForceFactory>,
    effects: HashMap<String, EffectFactory>,
}

impl ModifierRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a named force factory (replacing any existing one of that name).
    pub fn register_force(
        &mut self,
        name: impl Into<String>,
        factory: impl Fn() -> ForceModifier + Send + Sync + 'static,
    ) {
        self.forces.insert(name.into(), Box::new(factory));
    }

    /// Build the force registered under `name`, if any.
    pub fn force(&self, name: &str) -> Option<ForceModifier> {
        self.forces.get(name).map(|f| f())
    }

    /// Register a named effect-template factory (replacing any existing one).
    pub fn register_effect(
        &mut self,
        name: impl Into<String>,
        factory: impl Fn() -> EffectAsset + Send + Sync + 'static,
    ) {
        self.effects.insert(name.into(), Box::new(factory));
    }

    /// Build the effect registered under `name`, if any.
    pub fn effect(&self, name: &str) -> Option<EffectAsset> {
        self.effects.get(name).map(|f| f())
    }

    /// Names of all registered forces.
    pub fn force_names(&self) -> impl Iterator<Item = &str> {
        self.forces.keys().map(String::as_str)
    }

    /// Names of all registered effect templates.
    pub fn effect_names(&self) -> impl Iterator<Item = &str> {
        self.effects.keys().map(String::as_str)
    }
}
