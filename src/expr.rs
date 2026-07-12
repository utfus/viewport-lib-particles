//! Expression graph.
//!
//! An effect's per-attribute logic is built as a small expression graph rather
//! than fixed config fields. A [`Module`] owns the nodes; an [`ExprHandle`] is
//! an index into it. Init and update modifiers reference expressions by handle,
//! and the codegen pass ([`crate::codegen`]) lowers the reachable graph into
//! WGSL that runs per particle in the emit and simulate kernels.
//!
//! Nodes are typed (scalar `f32` or `vec3<f32>`); the type is inferred from the
//! node and its children so lowering can emit correct WGSL and assign to the
//! right particle attribute.

/// Index into a [`Module`]'s expression arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExprHandle(pub(crate) u32);

/// A single node in the expression graph.
#[derive(Clone, Debug)]
pub enum Expr {
    /// Scalar constant.
    LitF32(f32),
    /// Vector constant.
    LitVec3([f32; 3]),
    /// A built-in particle attribute by name. Valid names depend on the kernel:
    /// `"origin"` (emit and simulate), and `"position"`, `"velocity"`, `"age"`,
    /// `"lifetime"`, `"seed"` (simulate).
    Attribute(&'static str),
    /// A CPU-updatable named property, read from the effect's per-frame property
    /// uniform. The name must match a
    /// [`PropertyDecl`](crate::effect::PropertyDecl) on the effect's program; the
    /// read yields that property's declared type (`f32` / `vec3` / `vec4`). Valid
    /// in both the emit and simulate kernels.
    Property(&'static str),
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
}

/// Owns the expression nodes for one effect and hands out [`ExprHandle`]s.
///
/// Children are always created before their parents, so node indices are a
/// valid topological order: lowering can walk the arena front to back.
#[derive(Clone, Debug, Default)]
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
    pub fn attr(&mut self, name: &'static str) -> ExprHandle {
        self.push(Expr::Attribute(name))
    }

    /// Read a CPU-updatable named property. The name must match a
    /// [`PropertyDecl`](crate::effect::PropertyDecl) declared on the program.
    pub fn property(&mut self, name: &'static str) -> ExprHandle {
        self.push(Expr::Property(name))
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
                | Expr::Max(a, b) => {
                    stack.push(a.0);
                    stack.push(b.0);
                }
                Expr::Sin(a)
                | Expr::Cos(a)
                | Expr::Splat3(a)
                | Expr::Normalize(a)
                | Expr::Length(a)
                | Expr::Noise(a)
                | Expr::CurlNoise(a) => stack.push(a.0),
                Expr::Clamp(x, lo, hi) => {
                    stack.push(x.0);
                    stack.push(lo.0);
                    stack.push(hi.0);
                }
                _ => {}
            }
        }
        (0..self.nodes.len() as u32)
            .filter(|i| seen[*i as usize])
            .collect()
    }
}
