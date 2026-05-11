//! WGSL codegen for an effect's emit and simulate kernels.
//!
//! An [`EffectProgram`](crate::effect::EffectProgram) compiles to two compute
//! shaders. The emit kernel recycles dead slots and runs the init modifiers to
//! fill a new particle; the simulate kernel runs the update modifiers, sums the
//! accelerations, integrates, and ages the particle. Both lower the reachable
//! part of the effect's expression [`Module`](crate::Module) into WGSL `let`
//! bindings (one per node, in child-before-parent order) so shared
//! subexpressions and random draws evaluate exactly once.
//!
//! The generated shaders bind the same layout as the fixed-function kernels
//! (particles at binding 0, a params uniform at binding 1, and, for emit, the
//! spawn-budget atomic at binding 2), so the plugin reuses the Phase 2 bind
//! group layouts and only swaps the pipelines.

use crate::effect::{Attribute, EffectProgram, UpdateOp};
use crate::expr::{Expr, ExprHandle, Module};

/// The WGSL source for one compiled effect.
#[derive(Clone, Debug)]
pub(crate) struct GeneratedShaders {
    /// Emit compute shader source (entry point `emit_main`).
    pub emit: String,
    /// Simulate compute shader source (entry point `sim_main`).
    pub sim: String,
}

/// Which kernel a lowering runs in. Governs how attribute reads resolve.
#[derive(Clone, Copy)]
enum Ctx {
    Emit,
    Sim,
}

/// Shared struct + binding declarations for both kernels.
const PARTICLE_STRUCT: &str = r#"
struct Particle {
    position: vec3<f32>,
    lifetime: f32,
    velocity: vec3<f32>,
    max_lifetime: f32,
    colour: vec4<f32>,
    size: f32,
    seed: f32,
    pad: vec2<f32>,
};

struct GenParams {
    origin: vec4<f32>,
    dt: f32,
    time: f32,
    spawn_count: u32,
    capacity: u32,
    rng_seed: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};
"#;

/// Random helpers, emit kernel only.
const RNG_HELPERS: &str = r#"
fn hash_u32(x: u32) -> u32 {
    var v = x;
    v ^= v >> 16u;
    v = v * 0x7feb352du;
    v ^= v >> 15u;
    v = v * 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}
fn rand01(state: ptr<function, u32>) -> f32 {
    *state = hash_u32(*state);
    return f32(*state) / 4294967295.0;
}
fn rand_dir(state: ptr<function, u32>) -> vec3<f32> {
    let z = rand01(state) * 2.0 - 1.0;
    let a = rand01(state) * 6.2831853;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}
"#;

/// Format an f32 as a WGSL float literal (always with a decimal point).
fn flit(v: f32) -> String {
    format!("{v:?}")
}

/// Resolve an attribute read to its WGSL expression in the given kernel.
fn attribute_wgsl(name: &str, ctx: Ctx) -> String {
    match (ctx, name) {
        (_, "origin") => "params.origin.xyz".to_string(),
        (Ctx::Sim, "position") => "p.position".to_string(),
        (Ctx::Sim, "velocity") => "p.velocity".to_string(),
        (Ctx::Sim, "lifetime") => "p.lifetime".to_string(),
        (Ctx::Sim, "age") => "(p.max_lifetime - p.lifetime)".to_string(),
        (Ctx::Sim, "seed") => "p.seed".to_string(),
        // Not meaningful in this kernel; resolve to a zero of the right shape.
        (_, "position") | (_, "velocity") => "vec3<f32>(0.0, 0.0, 0.0)".to_string(),
        _ => "0.0".to_string(),
    }
}

/// Right-hand side for one node, referencing `v{child}` for its children.
fn lower_node(module: &Module, i: u32, ctx: Ctx) -> String {
    match *module.get(ExprHandle(i)) {
        Expr::LitF32(v) => flit(v),
        Expr::LitVec3([x, y, z]) => {
            format!("vec3<f32>({}, {}, {})", flit(x), flit(y), flit(z))
        }
        Expr::Attribute(name) => attribute_wgsl(name, ctx),
        Expr::Rand => "rand01(&rng)".to_string(),
        Expr::RandUnit => "rand_dir(&rng)".to_string(),
        Expr::Add(a, b) => format!("(v{} + v{})", a.0, b.0),
        Expr::Sub(a, b) => format!("(v{} - v{})", a.0, b.0),
        Expr::Mul(a, b) => format!("(v{} * v{})", a.0, b.0),
        Expr::Div(a, b) => format!("(v{} / v{})", a.0, b.0),
        Expr::Sin(a) => format!("sin(v{})", a.0),
        Expr::Cos(a) => format!("cos(v{})", a.0),
        Expr::Splat3(a) => format!("vec3<f32>(v{})", a.0),
        Expr::Normalize(a) => format!("normalize(v{})", a.0),
        Expr::Length(a) => format!("length(v{})", a.0),
        Expr::Cross(a, b) => format!("cross(v{}, v{})", a.0, b.0),
        Expr::Min(a, b) => format!("min(v{}, v{})", a.0, b.0),
        Expr::Max(a, b) => format!("max(v{}, v{})", a.0, b.0),
        Expr::Clamp(x, lo, hi) => format!("clamp(v{}, v{}, v{})", x.0, lo.0, hi.0),
    }
}

/// Emit `let v{i} = ...;` for every node reachable from `roots`.
fn lower_block(module: &Module, roots: &[ExprHandle], ctx: Ctx, indent: &str) -> String {
    let mut out = String::new();
    for i in module.reachable(roots) {
        out.push_str(indent);
        out.push_str(&format!("let v{} = {};\n", i, lower_node(module, i, ctx)));
    }
    out
}

/// Lower a program to its emit and simulate WGSL.
pub(crate) fn generate_program(program: &EffectProgram) -> GeneratedShaders {
    GeneratedShaders {
        emit: generate_emit(program),
        sim: generate_sim(program),
    }
}

fn find_attr(program: &EffectProgram, attr: Attribute) -> Option<ExprHandle> {
    // Last writer wins.
    program
        .init
        .iter()
        .rev()
        .find(|m| m.attribute == attr)
        .map(|m| m.value)
}

fn generate_emit(program: &EffectProgram) -> String {
    let module = &program.module;
    let roots: Vec<ExprHandle> = program.init.iter().map(|m| m.value).collect();
    let lets = lower_block(module, &roots, Ctx::Emit, "    ");

    let pos = find_attr(program, Attribute::Position)
        .map(|h| format!("v{}", h.0))
        .unwrap_or_else(|| "params.origin.xyz".to_string());
    let vel = find_attr(program, Attribute::Velocity)
        .map(|h| format!("v{}", h.0))
        .unwrap_or_else(|| "vec3<f32>(0.0, 0.0, 0.0)".to_string());
    let life = find_attr(program, Attribute::Lifetime)
        .map(|h| format!("v{}", h.0))
        .unwrap_or_else(|| {
            format!(
                "mix({}, {}, rand01(&rng))",
                flit(program.lifetime.0),
                flit(program.lifetime.1)
            )
        });
    let col = find_attr(program, Attribute::Colour)
        .map(|h| format!("v{}", h.0))
        .unwrap_or_else(|| "vec3<f32>(1.0, 1.0, 1.0)".to_string());
    let size = find_attr(program, Attribute::Size)
        .map(|h| format!("v{}", h.0))
        .unwrap_or_else(|| "0.1".to_string());

    format!(
        r#"{PARTICLE_STRUCT}
struct Budget {{ count: atomic<u32> }};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: GenParams;
@group(0) @binding(2) var<storage, read_write> budget: Budget;
{RNG_HELPERS}
@compute @workgroup_size(64)
fn emit_main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= params.capacity) {{ return; }}
    if (particles[i].lifetime > 0.0) {{ return; }}
    let claimed = atomicAdd(&budget.count, 1u);
    if (claimed >= params.spawn_count) {{ return; }}
    var rng = hash_u32(params.rng_seed ^ (i * 747796405u) ^ (claimed * 2891336453u));
{lets}
    var p: Particle;
    p.position = {pos};
    p.velocity = {vel};
    let life0 = {life};
    p.lifetime = life0;
    p.max_lifetime = life0;
    p.colour = vec4<f32>({col}, 1.0);
    p.size = {size};
    p.seed = rand01(&rng);
    p.pad = vec2<f32>(0.0, 0.0);
    particles[i] = p;
}}
"#
    )
}

fn generate_sim(program: &EffectProgram) -> String {
    let module = &program.module;
    let roots: Vec<ExprHandle> = program
        .update
        .iter()
        .map(|op| match op {
            UpdateOp::Accelerate(h) => *h,
        })
        .collect();
    let lets = lower_block(module, &roots, Ctx::Sim, "    ");

    let mut accel = String::new();
    for op in &program.update {
        match op {
            UpdateOp::Accelerate(h) => {
                accel.push_str(&format!("    accel = accel + v{};\n", h.0));
            }
        }
    }

    format!(
        r#"{PARTICLE_STRUCT}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: GenParams;

@compute @workgroup_size(64)
fn sim_main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&particles)) {{ return; }}
    var p = particles[i];
    if (p.lifetime <= 0.0) {{ return; }}
    let dt = params.dt;
{lets}
    var accel = vec3<f32>(0.0, 0.0, 0.0);
{accel}
    p.velocity = p.velocity + accel * dt;
    p.position = p.position + p.velocity * dt;
    p.lifetime = p.lifetime - dt;
    particles[i] = p;
}}
"#
    )
}
