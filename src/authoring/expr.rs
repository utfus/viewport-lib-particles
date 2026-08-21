//! Expression graph.
//!
//! An effect's per-attribute logic is built as a small expression graph rather
//! than fixed config fields. A [`Module`] owns the nodes; an [`ExprHandle`] is
//! an index into it. Init and update modifiers reference expressions by handle,
//! and the codegen pass ([`crate::codegen`]) lowers the reachable graph into
//! WGSL that runs per particle in the emit and simulate kernels.
//!
//! Nodes are typed (`f32` / `vec2` / `vec3` / `vec4` / `bool`); the type is
//! inferred from the node and its children ([`Module::infer_types`]) so lowering
//! can emit correct WGSL and assign to the right particle attribute, and so
//! [`crate::EffectProgram::validate`] can reject an ill-typed graph before the
//! generated WGSL reaches the shader compiler.
//!
//! The op set is table-driven ([`UnaryOp`] / [`BinaryOp`] / [`TernaryOp`]): a new
//! intrinsic is usually one enum variant plus its lowering/type row, with no new
//! `Expr` variant or reachability arm — the general `Unary`/`Binary`/`Ternary`
//! nodes carry the op.

/// Index into a [`Module`]'s expression arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExprHandle(pub(crate) u32);

/// The value type of an expression node. Used by [`Module::infer_types`] /
/// [`crate::EffectProgram::validate`] to catch arity and type errors at build
/// time, before the generated WGSL is compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ValueType {
    /// Scalar `f32`.
    F32,
    /// `vec2<f32>`.
    Vec2,
    /// `vec3<f32>`.
    Vec3,
    /// `vec4<f32>`.
    Vec4,
    /// `bool` (from comparisons / `is_alive`; usable as a `select` condition).
    Bool,
}

impl ValueType {
    /// Component count for a float type (`1` for `F32`), or `None` for `Bool`.
    pub fn components(self) -> Option<u32> {
        match self {
            ValueType::F32 => Some(1),
            ValueType::Vec2 => Some(2),
            ValueType::Vec3 => Some(3),
            ValueType::Vec4 => Some(4),
            ValueType::Bool => None,
        }
    }

    /// Whether this is a float scalar or vector (not `Bool`).
    pub fn is_float(self) -> bool {
        !matches!(self, ValueType::Bool)
    }

    /// Whether this is a vector (`Vec2`/`Vec3`/`Vec4`).
    pub fn is_vec(self) -> bool {
        matches!(self, ValueType::Vec2 | ValueType::Vec3 | ValueType::Vec4)
    }

    fn label(self) -> &'static str {
        match self {
            ValueType::F32 => "f32",
            ValueType::Vec2 => "vec2",
            ValueType::Vec3 => "vec3",
            ValueType::Vec4 => "vec4",
            ValueType::Bool => "bool",
        }
    }
}

/// A componentwise unary intrinsic (`fn(x) -> same-shape`). Every variant lowers
/// to the like-named WGSL builtin and preserves its operand's float type, so
/// adding one is a single enum variant plus its [`wgsl_fn`](UnaryOp::wgsl_fn)
/// name — no separate lowering, type rule, or reachability arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnaryOp {
    Abs,
    Floor,
    Fract,
    Ceil,
    Round,
    Sign,
    Saturate,
    Sqrt,
    InverseSqrt,
    Exp,
    Exp2,
    Log,
    Log2,
    Tan,
    Asin,
    Acos,
    Atan,
}

impl UnaryOp {
    /// The WGSL builtin this op lowers to.
    pub fn wgsl_fn(self) -> &'static str {
        match self {
            UnaryOp::Abs => "abs",
            UnaryOp::Floor => "floor",
            UnaryOp::Fract => "fract",
            UnaryOp::Ceil => "ceil",
            UnaryOp::Round => "round",
            UnaryOp::Sign => "sign",
            UnaryOp::Saturate => "saturate",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::InverseSqrt => "inverseSqrt",
            UnaryOp::Exp => "exp",
            UnaryOp::Exp2 => "exp2",
            UnaryOp::Log => "log",
            UnaryOp::Log2 => "log2",
            UnaryOp::Tan => "tan",
            UnaryOp::Asin => "asin",
            UnaryOp::Acos => "acos",
            UnaryOp::Atan => "atan",
        }
    }
}

/// A binary op. Each variant carries both its lowering shape and its type rule
/// (via [`lower`](BinaryOp::lower) / the `infer` match in [`Module::infer_types`]),
/// so the families — plain float intrinsics, the vector reducers (`dot` /
/// `distance`), and the scalar comparisons — extend independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BinaryOp {
    /// `pow(a, b)`, componentwise; result shares the operands' float type.
    Pow,
    /// `atan2(a, b)`, componentwise.
    Atan2,
    /// `a % b`, componentwise.
    Modulo,
    /// `step(edge, x)`, componentwise.
    Step,
    /// `dot(a, b)` of two like vectors → `f32`.
    Dot,
    /// `distance(a, b)` → `f32`.
    Distance,
    /// `a < b` (scalars) → `bool`.
    Less,
    /// `a > b` (scalars) → `bool`.
    Greater,
    /// `a <= b` (scalars) → `bool`.
    LessEqual,
    /// `a >= b` (scalars) → `bool`.
    GreaterEqual,
    /// `a == b` (scalars) → `bool`.
    Equal,
    /// `a != b` (scalars) → `bool`.
    NotEqual,
}

impl BinaryOp {
    /// Lower to a WGSL expression given the child value names `a` / `b`.
    pub fn lower(self, a: &str, b: &str) -> String {
        match self {
            BinaryOp::Pow => format!("pow({a}, {b})"),
            BinaryOp::Atan2 => format!("atan2({a}, {b})"),
            BinaryOp::Modulo => format!("({a} % {b})"),
            BinaryOp::Step => format!("step({a}, {b})"),
            BinaryOp::Dot => format!("dot({a}, {b})"),
            BinaryOp::Distance => format!("distance({a}, {b})"),
            BinaryOp::Less => format!("({a} < {b})"),
            BinaryOp::Greater => format!("({a} > {b})"),
            BinaryOp::LessEqual => format!("({a} <= {b})"),
            BinaryOp::GreaterEqual => format!("({a} >= {b})"),
            BinaryOp::Equal => format!("({a} == {b})"),
            BinaryOp::NotEqual => format!("({a} != {b})"),
        }
    }

    fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Less
                | BinaryOp::Greater
                | BinaryOp::LessEqual
                | BinaryOp::GreaterEqual
                | BinaryOp::Equal
                | BinaryOp::NotEqual
        )
    }
}

/// A ternary op. `Mix` / `Smoothstep` are float intrinsics; `Select` is a
/// condition-driven choice (`cond ? a : b`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TernaryOp {
    /// `mix(a, b, t)` — linear blend; `t` scalar or the operands' vector type.
    Mix,
    /// `smoothstep(edge0, edge1, x)`, componentwise.
    Smoothstep,
    /// `cond ? a : b` — `cond` is `bool`, `a` / `b` share a type.
    Select,
}

/// A vector component selector for [`Expr::Swizzle`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Component {
    X,
    Y,
    Z,
    W,
}

impl Component {
    /// WGSL component accessor (`"x"` .. `"w"`).
    pub fn wgsl(self) -> &'static str {
        match self {
            Component::X => "x",
            Component::Y => "y",
            Component::Z => "z",
            Component::W => "w",
        }
    }

    /// 1-based number of components the source vector must have.
    fn min_components(self) -> u32 {
        match self {
            Component::X => 1,
            Component::Y => 2,
            Component::Z => 3,
            Component::W => 4,
        }
    }
}

/// A type or arity error found by [`Module::infer_types`], carrying the offending
/// node and a human-readable message. Surfaced by
/// [`crate::EffectProgram::validate`] before WGSL compilation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeError {
    /// The node the error is about.
    pub handle: ExprHandle,
    /// Human-readable description.
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "expr v{}: {}", self.handle.0, self.message)
    }
}

impl std::error::Error for TypeError {}

/// Resolves the value type of [`Expr::Attribute`] / [`Expr::Property`] reads for
/// type inference. Supplied by the caller (e.g. [`crate::EffectProgram::validate`])
/// since attribute and property types live outside the [`Module`].
pub struct TypeEnv<'a> {
    /// Attribute name → type, or `None` if the name is not a known attribute.
    pub attribute: &'a dyn Fn(&str) -> Option<ValueType>,
    /// Property name → type, or `None` if the name is not a declared property.
    pub property: &'a dyn Fn(&str) -> Option<ValueType>,
}

/// A single node in the expression graph.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Expr {
    /// Scalar constant.
    LitF32(f32),
    /// Vector constant.
    LitVec3([f32; 3]),
    /// A built-in particle attribute by name. Valid names depend on the kernel:
    /// `"origin"` (emit and simulate), and `"position"`, `"velocity"`, `"age"`,
    /// `"lifetime"`, `"seed"` (simulate).
    Attribute(String),
    /// A CPU-updatable named property, read from the effect's per-frame property
    /// uniform. The name must match a
    /// [`PropertyDecl`](crate::effect::PropertyDecl) on the effect's program; the
    /// read yields that property's declared type (`f32` / `vec3` / `vec4`). Valid
    /// in both the emit and simulate kernels.
    Property(String),
    /// A fresh uniform random scalar in `0..1` (emit only; consumes rng).
    Rand,
    /// A fresh uniform random unit vector (emit only; consumes rng).
    RandUnit,
    /// Component-wise sum. Operands must share a type.
    Add(ExprHandle, ExprHandle),
    /// Component-wise difference. Operands must share a type.
    Sub(ExprHandle, ExprHandle),
    /// Product. `vec3 * f32` and `f32 * vec3` broadcast; result is `vec3` if
    /// either operand is `vec3`.
    Mul(ExprHandle, ExprHandle),
    /// Quotient, same typing rules as [`Expr::Mul`].
    Div(ExprHandle, ExprHandle),
    /// Sine of a scalar.
    Sin(ExprHandle),
    /// Cosine of a scalar.
    Cos(ExprHandle),
    /// Broadcast a scalar to a `vec3`.
    Splat3(ExprHandle),
    /// Normalize a `vec3` (zero-safe is the caller's responsibility).
    Normalize(ExprHandle),
    /// Length of a `vec3` (scalar).
    Length(ExprHandle),
    /// Cross product of two `vec3`s.
    Cross(ExprHandle, ExprHandle),
    /// Component-wise minimum.
    Min(ExprHandle, ExprHandle),
    /// Component-wise maximum.
    Max(ExprHandle, ExprHandle),
    /// Clamp `x` to `[lo, hi]`.
    Clamp(ExprHandle, ExprHandle, ExprHandle),
    /// 3D gradient value noise of a `vec3` sample point, roughly in `[-1, 1]`
    /// (scalar).
    Noise(ExprHandle),
    /// Divergence-free curl noise of a `vec3` sample point (`vec3`). Smooth
    /// turbulent flow; scale the sample point for frequency and the result for
    /// strength.
    CurlNoise(ExprHandle),
    /// A componentwise unary intrinsic (`abs`, `sqrt`, `exp`, `atan`, …). Result
    /// shares the operand's float type. See [`UnaryOp`].
    Unary(UnaryOp, ExprHandle),
    /// A binary op (`pow`, `dot`, `step`, comparisons, …). Result type depends on
    /// the op. See [`BinaryOp`].
    Binary(BinaryOp, ExprHandle, ExprHandle),
    /// A ternary op (`mix`, `smoothstep`, `select`). See [`TernaryOp`].
    Ternary(TernaryOp, ExprHandle, ExprHandle, ExprHandle),
    /// Construct a `vec2<f32>` from two scalars.
    Vec2(ExprHandle, ExprHandle),
    /// Construct a `vec3<f32>` from three scalars (runtime values, unlike the
    /// constant [`Expr::LitVec3`]).
    Vec3(ExprHandle, ExprHandle, ExprHandle),
    /// Construct a `vec4<f32>` from four scalars.
    Vec4(ExprHandle, ExprHandle, ExprHandle, ExprHandle),
    /// Extract one component of a vector as a scalar. See [`Component`].
    Swizzle(ExprHandle, Component),
    /// Elapsed simulation time in seconds (`f32`), from the params uniform. Valid
    /// in both kernels.
    Time,
    /// This frame's simulation time step in seconds (`f32`). Valid in both
    /// kernels (the emit kernel reads the same `params.dt`).
    DeltaTime,
    /// Whether the particle is alive (`bool`). Meaningful in the simulate kernel;
    /// resolves to `true` in the emit kernel.
    IsAlive,
}

/// Owns the expression nodes for one effect and hands out [`ExprHandle`]s.
///
/// Children are always created before their parents, so node indices are a
/// valid topological order: lowering can walk the arena front to back.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Module {
    nodes: Vec<Expr>,
}

impl Module {
    /// A fresh, empty module.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern an expression and return a handle to it.
    pub fn push(&mut self, expr: Expr) -> ExprHandle {
        let idx = self.nodes.len() as u32;
        self.nodes.push(expr);
        ExprHandle(idx)
    }

    /// Scalar literal.
    pub fn lit(&mut self, v: f32) -> ExprHandle {
        self.push(Expr::LitF32(v))
    }

    /// Vector literal.
    pub fn lit_vec3(&mut self, v: [f32; 3]) -> ExprHandle {
        self.push(Expr::LitVec3(v))
    }

    /// Read a built-in attribute by name.
    pub fn attr(&mut self, name: impl Into<String>) -> ExprHandle {
        self.push(Expr::Attribute(name.into()))
    }

    /// Read a CPU-updatable named property. The name must match a
    /// [`PropertyDecl`](crate::effect::PropertyDecl) declared on the program.
    pub fn property(&mut self, name: impl Into<String>) -> ExprHandle {
        self.push(Expr::Property(name.into()))
    }

    /// Fresh random scalar in `0..1`.
    pub fn rand(&mut self) -> ExprHandle {
        self.push(Expr::Rand)
    }

    /// Fresh random unit vector.
    pub fn rand_unit(&mut self) -> ExprHandle {
        self.push(Expr::RandUnit)
    }

    /// `a + b`.
    pub fn add(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.push(Expr::Add(a, b))
    }

    /// `a - b`.
    pub fn sub(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.push(Expr::Sub(a, b))
    }

    /// `a * b`.
    pub fn mul(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.push(Expr::Mul(a, b))
    }

    /// `a / b`.
    pub fn div(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.push(Expr::Div(a, b))
    }

    /// `sin(a)`.
    pub fn sin(&mut self, a: ExprHandle) -> ExprHandle {
        self.push(Expr::Sin(a))
    }

    /// `cos(a)`.
    pub fn cos(&mut self, a: ExprHandle) -> ExprHandle {
        self.push(Expr::Cos(a))
    }

    /// Broadcast a scalar to a vector.
    pub fn splat3(&mut self, a: ExprHandle) -> ExprHandle {
        self.push(Expr::Splat3(a))
    }

    /// `normalize(a)`.
    pub fn normalize(&mut self, a: ExprHandle) -> ExprHandle {
        self.push(Expr::Normalize(a))
    }

    /// `length(a)`.
    pub fn length(&mut self, a: ExprHandle) -> ExprHandle {
        self.push(Expr::Length(a))
    }

    /// `cross(a, b)`.
    pub fn cross(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.push(Expr::Cross(a, b))
    }

    /// `min(a, b)`.
    pub fn min(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.push(Expr::Min(a, b))
    }

    /// `max(a, b)`.
    pub fn max(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.push(Expr::Max(a, b))
    }

    /// `clamp(x, lo, hi)`.
    pub fn clamp(&mut self, x: ExprHandle, lo: ExprHandle, hi: ExprHandle) -> ExprHandle {
        self.push(Expr::Clamp(x, lo, hi))
    }

    /// 3D gradient value noise (scalar) of a `vec3` sample point.
    pub fn noise(&mut self, p: ExprHandle) -> ExprHandle {
        self.push(Expr::Noise(p))
    }

    /// Divergence-free curl noise (`vec3`) of a `vec3` sample point.
    pub fn curl_noise(&mut self, p: ExprHandle) -> ExprHandle {
        self.push(Expr::CurlNoise(p))
    }

    /// A componentwise unary intrinsic. Prefer the named helpers ([`abs`](Self::abs),
    /// [`sqrt`](Self::sqrt), …); this is the general entry point.
    pub fn unary(&mut self, op: UnaryOp, a: ExprHandle) -> ExprHandle {
        self.push(Expr::Unary(op, a))
    }

    /// A binary op. Prefer the named helpers ([`pow`](Self::pow),
    /// [`dot`](Self::dot), …); this is the general entry point.
    pub fn binary(&mut self, op: BinaryOp, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.push(Expr::Binary(op, a, b))
    }

    /// A ternary op. Prefer the named helpers ([`mix`](Self::mix),
    /// [`select`](Self::select), …); this is the general entry point.
    pub fn ternary(
        &mut self,
        op: TernaryOp,
        a: ExprHandle,
        b: ExprHandle,
        c: ExprHandle,
    ) -> ExprHandle {
        self.push(Expr::Ternary(op, a, b, c))
    }

    /// `abs(a)`.
    pub fn abs(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Abs, a)
    }
    /// `floor(a)`.
    pub fn floor(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Floor, a)
    }
    /// `fract(a)`.
    pub fn fract(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Fract, a)
    }
    /// `ceil(a)`.
    pub fn ceil(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Ceil, a)
    }
    /// `round(a)`.
    pub fn round(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Round, a)
    }
    /// `sign(a)`.
    pub fn sign(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Sign, a)
    }
    /// `saturate(a)` (clamp to `[0, 1]`).
    pub fn saturate(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Saturate, a)
    }
    /// `sqrt(a)`.
    pub fn sqrt(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Sqrt, a)
    }
    /// `inverseSqrt(a)`.
    pub fn inverse_sqrt(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::InverseSqrt, a)
    }
    /// `exp(a)`.
    pub fn exp(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Exp, a)
    }
    /// `exp2(a)`.
    pub fn exp2(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Exp2, a)
    }
    /// `log(a)`.
    pub fn log(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Log, a)
    }
    /// `log2(a)`.
    pub fn log2(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Log2, a)
    }
    /// `tan(a)`.
    pub fn tan(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Tan, a)
    }
    /// `asin(a)`.
    pub fn asin(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Asin, a)
    }
    /// `acos(a)`.
    pub fn acos(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Acos, a)
    }
    /// `atan(a)`.
    pub fn atan(&mut self, a: ExprHandle) -> ExprHandle {
        self.unary(UnaryOp::Atan, a)
    }

    /// `pow(a, b)`.
    pub fn pow(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::Pow, a, b)
    }
    /// `atan2(a, b)`.
    pub fn atan2(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::Atan2, a, b)
    }
    /// `a % b`.
    pub fn rem(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::Modulo, a, b)
    }
    /// `step(edge, x)`.
    pub fn step(&mut self, edge: ExprHandle, x: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::Step, edge, x)
    }
    /// `dot(a, b)` (→ `f32`).
    pub fn dot(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::Dot, a, b)
    }
    /// `distance(a, b)` (→ `f32`).
    pub fn distance(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::Distance, a, b)
    }
    /// `a < b` (→ `bool`).
    pub fn lt(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::Less, a, b)
    }
    /// `a > b` (→ `bool`).
    pub fn gt(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::Greater, a, b)
    }
    /// `a <= b` (→ `bool`).
    pub fn le(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::LessEqual, a, b)
    }
    /// `a >= b` (→ `bool`).
    pub fn ge(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::GreaterEqual, a, b)
    }
    /// `a == b` (→ `bool`).
    pub fn eq(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::Equal, a, b)
    }
    /// `a != b` (→ `bool`).
    pub fn ne(&mut self, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.binary(BinaryOp::NotEqual, a, b)
    }

    /// `mix(a, b, t)` — linear blend by `t`.
    pub fn mix(&mut self, a: ExprHandle, b: ExprHandle, t: ExprHandle) -> ExprHandle {
        self.ternary(TernaryOp::Mix, a, b, t)
    }
    /// `smoothstep(edge0, edge1, x)`.
    pub fn smoothstep(&mut self, edge0: ExprHandle, edge1: ExprHandle, x: ExprHandle) -> ExprHandle {
        self.ternary(TernaryOp::Smoothstep, edge0, edge1, x)
    }
    /// `cond ? a : b` — pick `a` when `cond` (a `bool`) is true, else `b`.
    pub fn select(&mut self, cond: ExprHandle, a: ExprHandle, b: ExprHandle) -> ExprHandle {
        self.ternary(TernaryOp::Select, cond, a, b)
    }

    /// Build a `vec2<f32>` from two scalars.
    pub fn vec2(&mut self, x: ExprHandle, y: ExprHandle) -> ExprHandle {
        self.push(Expr::Vec2(x, y))
    }
    /// Build a `vec3<f32>` from three scalars.
    pub fn vec3(&mut self, x: ExprHandle, y: ExprHandle, z: ExprHandle) -> ExprHandle {
        self.push(Expr::Vec3(x, y, z))
    }
    /// Build a `vec4<f32>` from four scalars.
    pub fn vec4(
        &mut self,
        x: ExprHandle,
        y: ExprHandle,
        z: ExprHandle,
        w: ExprHandle,
    ) -> ExprHandle {
        self.push(Expr::Vec4(x, y, z, w))
    }

    /// Extract one component of a vector (→ `f32`).
    pub fn swizzle(&mut self, v: ExprHandle, component: Component) -> ExprHandle {
        self.push(Expr::Swizzle(v, component))
    }
    /// `v.x`.
    pub fn x(&mut self, v: ExprHandle) -> ExprHandle {
        self.swizzle(v, Component::X)
    }
    /// `v.y`.
    pub fn y(&mut self, v: ExprHandle) -> ExprHandle {
        self.swizzle(v, Component::Y)
    }
    /// `v.z`.
    pub fn z(&mut self, v: ExprHandle) -> ExprHandle {
        self.swizzle(v, Component::Z)
    }
    /// `v.w`.
    pub fn w(&mut self, v: ExprHandle) -> ExprHandle {
        self.swizzle(v, Component::W)
    }

    /// Elapsed simulation time in seconds (`f32`).
    pub fn time(&mut self) -> ExprHandle {
        self.push(Expr::Time)
    }
    /// This frame's time step in seconds (`f32`).
    pub fn delta_time(&mut self) -> ExprHandle {
        self.push(Expr::DeltaTime)
    }
    /// Whether the particle is alive (`bool`).
    pub fn is_alive(&mut self) -> ExprHandle {
        self.push(Expr::IsAlive)
    }

    /// Read a node by handle.
    pub(crate) fn get(&self, handle: ExprHandle) -> &Expr {
        &self.nodes[handle.0 as usize]
    }

    /// Number of interned nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the module holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Handles reachable from `roots`, in ascending (child-before-parent) order.
    pub(crate) fn reachable(&self, roots: &[ExprHandle]) -> Vec<u32> {
        let mut seen = vec![false; self.nodes.len()];
        let mut stack: Vec<u32> = roots.iter().map(|h| h.0).collect();
        while let Some(i) = stack.pop() {
            if seen[i as usize] {
                continue;
            }
            seen[i as usize] = true;
            match self.nodes[i as usize] {
                Expr::Add(a, b)
                | Expr::Sub(a, b)
                | Expr::Mul(a, b)
                | Expr::Div(a, b)
                | Expr::Cross(a, b)
                | Expr::Min(a, b)
                | Expr::Max(a, b)
                | Expr::Binary(_, a, b)
                | Expr::Vec2(a, b) => {
                    stack.push(a.0);
                    stack.push(b.0);
                }
                Expr::Sin(a)
                | Expr::Cos(a)
                | Expr::Splat3(a)
                | Expr::Normalize(a)
                | Expr::Length(a)
                | Expr::Noise(a)
                | Expr::CurlNoise(a)
                | Expr::Unary(_, a)
                | Expr::Swizzle(a, _) => stack.push(a.0),
                Expr::Clamp(x, lo, hi)
                | Expr::Ternary(_, x, lo, hi)
                | Expr::Vec3(x, lo, hi) => {
                    stack.push(x.0);
                    stack.push(lo.0);
                    stack.push(hi.0);
                }
                Expr::Vec4(a, b, c, d) => {
                    stack.push(a.0);
                    stack.push(b.0);
                    stack.push(c.0);
                    stack.push(d.0);
                }
                _ => {}
            }
        }
        (0..self.nodes.len() as u32)
            .filter(|i| seen[*i as usize])
            .collect()
    }

    /// Infer every node's [`ValueType`], checking each op's operand arity and
    /// types, and return them indexed by handle (`result[h.0]`). The first
    /// ill-typed node is returned as an [`Err`]. `env` resolves attribute and
    /// property types, which live outside the module.
    ///
    /// Nodes are stored child-before-parent, so a single forward pass suffices:
    /// each node's operand types are already known. Used by
    /// [`crate::EffectProgram::validate`] to catch authoring errors before the
    /// generated WGSL reaches the shader compiler.
    pub fn infer_types(&self, env: &TypeEnv) -> Result<Vec<ValueType>, TypeError> {
        use ValueType as V;
        let mut t: Vec<ValueType> = Vec::with_capacity(self.nodes.len());
        for (i, node) in self.nodes.iter().enumerate() {
            let h = ExprHandle(i as u32);
            let err = |msg: String| TypeError { handle: h, message: msg };
            // Operand types (all children have a lower index, so already inferred).
            let ty = |x: &ExprHandle| t[x.0 as usize];
            let value = match node {
                Expr::LitF32(_) => V::F32,
                Expr::LitVec3(_) => V::Vec3,
                Expr::Attribute(name) => (env.attribute)(name)
                    .ok_or_else(|| err(format!("unknown attribute {name:?}")))?,
                Expr::Property(name) => (env.property)(name)
                    .ok_or_else(|| err(format!("unknown property {name:?}")))?,
                Expr::Rand => V::F32,
                Expr::RandUnit => V::Vec3,
                Expr::Add(a, b) | Expr::Sub(a, b) => {
                    let (ta, tb) = (ty(a), ty(b));
                    if !ta.is_float() || ta != tb {
                        return Err(err(format!(
                            "add/sub needs matching float operands, got {} and {}",
                            ta.label(),
                            tb.label()
                        )));
                    }
                    ta
                }
                Expr::Mul(a, b) | Expr::Div(a, b) => {
                    let (ta, tb) = (ty(a), ty(b));
                    match (ta, tb) {
                        _ if ta == tb && ta.is_float() => ta,
                        (V::F32, o) | (o, V::F32) if o.is_float() => o,
                        _ => {
                            return Err(err(format!(
                                "mul/div needs matching float types or a scalar factor, got {} and {}",
                                ta.label(),
                                tb.label()
                            )));
                        }
                    }
                }
                Expr::Sin(a) | Expr::Cos(a) => {
                    let ta = ty(a);
                    if !ta.is_float() {
                        return Err(err(format!("sin/cos needs a float, got {}", ta.label())));
                    }
                    ta
                }
                Expr::Splat3(a) => {
                    let ta = ty(a);
                    if ta != V::F32 {
                        return Err(err(format!("splat3 needs an f32, got {}", ta.label())));
                    }
                    V::Vec3
                }
                Expr::Normalize(a) => {
                    let ta = ty(a);
                    if !ta.is_vec() {
                        return Err(err(format!("normalize needs a vector, got {}", ta.label())));
                    }
                    ta
                }
                Expr::Length(a) => {
                    let ta = ty(a);
                    if !ta.is_vec() {
                        return Err(err(format!("length needs a vector, got {}", ta.label())));
                    }
                    V::F32
                }
                Expr::Cross(a, b) => {
                    let (ta, tb) = (ty(a), ty(b));
                    if ta != V::Vec3 || tb != V::Vec3 {
                        return Err(err(format!(
                            "cross needs two vec3, got {} and {}",
                            ta.label(),
                            tb.label()
                        )));
                    }
                    V::Vec3
                }
                Expr::Min(a, b) | Expr::Max(a, b) => {
                    let (ta, tb) = (ty(a), ty(b));
                    if !ta.is_float() || ta != tb {
                        return Err(err(format!(
                            "min/max needs matching float operands, got {} and {}",
                            ta.label(),
                            tb.label()
                        )));
                    }
                    ta
                }
                Expr::Clamp(x, lo, hi) => {
                    let (tx, tl, th) = (ty(x), ty(lo), ty(hi));
                    if !tx.is_float() || tx != tl || tx != th {
                        return Err(err(format!(
                            "clamp needs three matching float operands, got {}, {}, {}",
                            tx.label(),
                            tl.label(),
                            th.label()
                        )));
                    }
                    tx
                }
                Expr::Noise(a) => {
                    if ty(a) != V::Vec3 {
                        return Err(err(format!("noise needs a vec3, got {}", ty(a).label())));
                    }
                    V::F32
                }
                Expr::CurlNoise(a) => {
                    if ty(a) != V::Vec3 {
                        return Err(err(format!(
                            "curl_noise needs a vec3, got {}",
                            ty(a).label()
                        )));
                    }
                    V::Vec3
                }
                Expr::Unary(op, a) => {
                    let ta = ty(a);
                    if !ta.is_float() {
                        return Err(err(format!(
                            "{} needs a float operand, got {}",
                            op.wgsl_fn(),
                            ta.label()
                        )));
                    }
                    ta
                }
                Expr::Binary(op, a, b) => {
                    let (ta, tb) = (ty(a), ty(b));
                    if op.is_comparison() {
                        if ta != V::F32 || tb != V::F32 {
                            return Err(err(format!(
                                "comparison needs two f32 operands, got {} and {}",
                                ta.label(),
                                tb.label()
                            )));
                        }
                        V::Bool
                    } else {
                        match op {
                            BinaryOp::Dot => {
                                if !ta.is_vec() || ta != tb {
                                    return Err(err(format!(
                                        "dot needs two matching vectors, got {} and {}",
                                        ta.label(),
                                        tb.label()
                                    )));
                                }
                                V::F32
                            }
                            BinaryOp::Distance => {
                                if !ta.is_float() || ta != tb {
                                    return Err(err(format!(
                                        "distance needs two matching floats, got {} and {}",
                                        ta.label(),
                                        tb.label()
                                    )));
                                }
                                V::F32
                            }
                            // Pow / Atan2 / Modulo / Step: componentwise floats.
                            _ => {
                                if !ta.is_float() || ta != tb {
                                    return Err(err(format!(
                                        "{:?} needs matching float operands, got {} and {}",
                                        op,
                                        ta.label(),
                                        tb.label()
                                    )));
                                }
                                ta
                            }
                        }
                    }
                }
                Expr::Ternary(op, a, b, c) => {
                    let (ta, tb, tc) = (ty(a), ty(b), ty(c));
                    match op {
                        TernaryOp::Mix => {
                            if !ta.is_float() || ta != tb || (tc != V::F32 && tc != ta) {
                                return Err(err(format!(
                                    "mix needs matching float a/b and a scalar or matching t, got {}, {}, {}",
                                    ta.label(),
                                    tb.label(),
                                    tc.label()
                                )));
                            }
                            ta
                        }
                        TernaryOp::Smoothstep => {
                            if !ta.is_float() || ta != tb || ta != tc {
                                return Err(err(format!(
                                    "smoothstep needs three matching floats, got {}, {}, {}",
                                    ta.label(),
                                    tb.label(),
                                    tc.label()
                                )));
                            }
                            ta
                        }
                        TernaryOp::Select => {
                            if ta != V::Bool || tb != tc {
                                return Err(err(format!(
                                    "select needs a bool condition and matching a/b, got {}, {}, {}",
                                    ta.label(),
                                    tb.label(),
                                    tc.label()
                                )));
                            }
                            tb
                        }
                    }
                }
                Expr::Vec2(a, b) => {
                    if ty(a) != V::F32 || ty(b) != V::F32 {
                        return Err(err("vec2 needs two f32 components".to_string()));
                    }
                    V::Vec2
                }
                Expr::Vec3(a, b, c) => {
                    if ty(a) != V::F32 || ty(b) != V::F32 || ty(c) != V::F32 {
                        return Err(err("vec3 needs three f32 components".to_string()));
                    }
                    V::Vec3
                }
                Expr::Vec4(a, b, c, d) => {
                    if ty(a) != V::F32 || ty(b) != V::F32 || ty(c) != V::F32 || ty(d) != V::F32 {
                        return Err(err("vec4 needs four f32 components".to_string()));
                    }
                    V::Vec4
                }
                Expr::Swizzle(v, comp) => {
                    let tv = ty(v);
                    match tv.components() {
                        Some(n) if tv.is_vec() && n >= comp.min_components() => V::F32,
                        _ => {
                            return Err(err(format!(
                                "swizzle .{} needs a vector with that component, got {}",
                                comp.wgsl(),
                                tv.label()
                            )));
                        }
                    }
                }
                Expr::Time | Expr::DeltaTime => V::F32,
                Expr::IsAlive => V::Bool,
            };
            t.push(value);
        }
        Ok(t)
    }
}
