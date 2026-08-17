use crate::expr::{ExprHandle, Module};
use super::*;

/// A particle attribute an init modifier can set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Attribute {
    /// World-space position (`vec3`).
    Position,
    /// Velocity (`vec3`).
    Velocity,
    /// Lifetime in seconds (`f32`); also seeds `max_lifetime`.
    Lifetime,
    /// RGB colour (`vec3`); alpha comes from the lifetime fade at draw.
    Colour,
    /// Billboard size (`f32`).
    Size,
}

/// An init modifier: set an attribute from an expression at spawn.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SetAttribute {
    /// Which attribute to write.
    pub attribute: Attribute,
    /// Expression evaluated per particle. Its type must match the attribute
    /// (`vec3` for position/velocity/colour, `f32` for lifetime/size).
    pub value: ExprHandle,
}

/// An update modifier: contributes to the per-step motion.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UpdateOp {
    /// Add a `vec3` acceleration (integrated into velocity each step). Force
    /// modifiers are expressed this way; the expression may read attributes such
    /// as `position` to build attractors.
    Accelerate(ExprHandle),
}

/// A CPU-updatable value the host sets per frame, read in expressions via
/// [`Expr::Property`](crate::Expr::Property). The variant fixes the property's
/// type; a read yields `f32`, `vec3`, or `vec4` accordingly.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PropertyValue {
    /// A scalar property.
    F32(f32),
    /// A 3-component vector property.
    Vec3([f32; 3]),
    /// A 4-component vector property.
    Vec4([f32; 4]),
}

impl PropertyValue {
    /// Pack the value into a `vec4` lane for the property uniform (the GPU stores
    /// every property as a `vec4<f32>`, swizzled on read to its declared type).
    pub(crate) fn to_vec4(self) -> [f32; 4] {
        match self {
            PropertyValue::F32(v) => [v, 0.0, 0.0, 0.0],
            PropertyValue::Vec3([x, y, z]) => [x, y, z, 0.0],
            PropertyValue::Vec4(v) => v,
        }
    }
}

/// A named, typed property declared on an [`EffectProgram`]. Its
/// [`default`](Self::default) supplies the value until the host overrides it per
/// frame via [`ParticleItem`](crate::ParticleItem).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PropertyDecl {
    /// Property name, referenced by [`Expr::Property`](crate::Expr::Property) and
    /// used as the uniform field name; must be a valid WGSL identifier.
    pub name: String,
    /// Default value (and, via its variant, the property's type).
    pub default: PropertyValue,
}

/// A codegen program: an expression [`Module`] plus the init and update
/// modifiers that reference it. An [`EffectAsset`] carrying a program compiles
/// to per-effect emit and simulate WGSL kernels instead of using the
/// fixed-function [`Emitter`] path.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EffectProgram {
    /// The expression graph the modifiers reference.
    pub module: Module,
    /// Emission schedule.
    pub rate: SpawnRate,
    /// Spawn-time attribute writers, applied in order.
    pub init: Vec<SetAttribute>,
    /// Per-step contributions, applied in order.
    pub update: Vec<UpdateOp>,
    /// Per-particle lifetime range, sampled uniformly at spawn unless an init
    /// modifier writes `Lifetime` explicitly.
    pub lifetime: (f32, f32),
    /// CPU-updatable properties readable in the expression graph. Packed into a
    /// per-effect uniform in declaration order and updated from the per-frame
    /// submission.
    pub properties: Vec<PropertyDecl>,
}

impl Default for EffectProgram {
    fn default() -> Self {
        Self {
            module: Module::new(),
            rate: SpawnRate::default(),
            init: Vec::new(),
            update: Vec::new(),
            lifetime: (1.0, 2.0),
            properties: Vec::new(),
        }
    }
}

impl EffectProgram {
    /// A program with an empty module and the default rate and lifetime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the emission schedule.
    pub fn with_rate(mut self, rate: SpawnRate) -> Self {
        self.rate = rate;
        self
    }

    /// Append an init modifier.
    pub fn set(mut self, attribute: Attribute, value: ExprHandle) -> Self {
        self.init.push(SetAttribute { attribute, value });
        self
    }

    /// Append an update modifier.
    pub fn update(mut self, op: UpdateOp) -> Self {
        self.update.push(op);
        self
    }

    /// Set the spawn lifetime range.
    pub fn with_lifetime(mut self, min: f32, max: f32) -> Self {
        self.lifetime = (min, max);
        self
    }

    /// Declare a CPU-updatable property with a default value. Reference it in the
    /// expression graph with [`Module::property`](crate::Module::property) and
    /// override it per frame via
    /// [`ParticleItem::with_property`](crate::ParticleItem::with_property).
    pub fn property(mut self, name: impl Into<String>, default: PropertyValue) -> Self {
        self.properties.push(PropertyDecl {
            name: name.into(),
            default,
        });
        self
    }
}
