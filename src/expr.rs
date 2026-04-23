//! Expression graph.
//!
//! An effect's per-attribute logic is built as a small expression graph rather
//! than fixed config fields. A [`Module`] owns the nodes; an [`ExprHandle`] is
//! an index into it. Modifiers reference expressions by handle, and the codegen
//! pass lowers the reachable graph into WGSL.
//!
//! This mirrors the shape of a modern GPU-particle expression system: literals,
//! attribute reads (position, velocity, age, lifetime), and arithmetic combine
//! into values that init and update modifiers consume. The variants here are a
//! starting set; the graph grows as modifiers need more.

/// Index into a [`Module`]'s expression arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExprHandle(pub(crate) u32);

/// A single node in the expression graph.
///
/// Kept deliberately small for the skeleton. Real coverage (trig, noise,
/// per-particle random streams, curve/gradient samples) lands during codegen.
#[derive(Clone, Debug)]
pub enum Expr {
    /// A compile-time scalar constant.
    LitF32(f32),
    /// A compile-time vector constant.
    LitVec3([f32; 3]),
    /// Read a built-in particle attribute by name, e.g. `"position"`,
    /// `"velocity"`, `"age"`, `"lifetime"`.
    Attribute(&'static str),
    /// Add two sub-expressions.
    Add(ExprHandle, ExprHandle),
    /// Multiply two sub-expressions.
    Mul(ExprHandle, ExprHandle),
}

/// Owns the expression nodes for one effect and hands out [`ExprHandle`]s.
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

    /// Convenience: a scalar literal node.
    pub fn lit(&mut self, v: f32) -> ExprHandle {
        self.push(Expr::LitF32(v))
    }

    /// Convenience: a vec3 literal node.
    pub fn lit_vec3(&mut self, v: [f32; 3]) -> ExprHandle {
        self.push(Expr::LitVec3(v))
    }

    /// Read a node by handle. Returns `None` if the handle is out of range
    /// (only possible with a handle from a different module).
    pub fn get(&self, handle: ExprHandle) -> Option<&Expr> {
        self.nodes.get(handle.0 as usize)
    }

    /// Number of interned nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the module holds no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}
