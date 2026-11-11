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
//! spawn-budget atomic at binding 2), so the plugin reuses the fixed-function
//! bind group layouts and only swaps the pipelines.

use crate::effect::{Attribute, EffectProgram, PropertyDecl, PropertyValue, UpdateOp};
use crate::expr::{Expr, ExprHandle, Module, TernaryOp, TypeEnv, TypeError, ValueType};

/// The WGSL source for one compiled effect.
#[derive(Clone, Debug)]
pub(crate) struct GeneratedShaders {
    /// Emit compute shader source (entry point `emit_main`).
    pub emit: String,
    /// Simulate compute shader source (entry point `sim_main`).
    pub sim: String,
}

/// Which kernel a lowering runs in. Governs how attribute reads resolve.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Emit,
    Sim,
}

/// Lowering context: which kernel, plus the program's property declarations so
/// [`Expr::Property`] reads resolve to the right swizzle of the property uniform.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    kind: Kind,
    props: &'a [PropertyDecl],
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
    space: vec4<f32>,
    dt: f32,
    time: f32,
    spawn_count: u32,
    capacity: u32,
    rng_seed: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    plane: vec4<f32>,
};
"#;

/// Plane-lock helpers (`lock_position`, `lock_vector`, `plane_offset`), shared
/// with the fixed-function kernels. The lock rides `GenParams::plane`, so the
/// generated source does not depend on it and programs keep sharing pipelines.
pub(crate) const PLANE_WGSL: &str = include_str!("../runtime/shaders/particle_plane.wgsl");

/// 3D gradient (Perlin-style) value noise and divergence-free curl noise. Shared
/// by the fixed-function `CurlNoise` force (prepended to `particle_sim.wgsl`) and
/// the `noise` / `curl_noise` expression ops (prepended to generated kernels).
pub(crate) const NOISE_WGSL: &str = r#"
fn pn_hash33(p: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx) * 2.0 - 1.0;
}
fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let g000 = dot(pn_hash33(i + vec3<f32>(0.0, 0.0, 0.0)), f - vec3<f32>(0.0, 0.0, 0.0));
    let g100 = dot(pn_hash33(i + vec3<f32>(1.0, 0.0, 0.0)), f - vec3<f32>(1.0, 0.0, 0.0));
    let g010 = dot(pn_hash33(i + vec3<f32>(0.0, 1.0, 0.0)), f - vec3<f32>(0.0, 1.0, 0.0));
    let g110 = dot(pn_hash33(i + vec3<f32>(1.0, 1.0, 0.0)), f - vec3<f32>(1.0, 1.0, 0.0));
    let g001 = dot(pn_hash33(i + vec3<f32>(0.0, 0.0, 1.0)), f - vec3<f32>(0.0, 0.0, 1.0));
    let g101 = dot(pn_hash33(i + vec3<f32>(1.0, 0.0, 1.0)), f - vec3<f32>(1.0, 0.0, 1.0));
    let g011 = dot(pn_hash33(i + vec3<f32>(0.0, 1.0, 1.0)), f - vec3<f32>(0.0, 1.0, 1.0));
    let g111 = dot(pn_hash33(i + vec3<f32>(1.0, 1.0, 1.0)), f - vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(g000, g100, u.x);
    let x10 = mix(g010, g110, u.x);
    let x01 = mix(g001, g101, u.x);
    let x11 = mix(g011, g111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
fn pn_potential(p: vec3<f32>) -> vec3<f32> {
    let o = vec3<f32>(31.416, 47.853, 12.793);
    return vec3<f32>(value_noise(p), value_noise(p + o), value_noise(p + 2.0 * o));
}
fn curl_noise(p: vec3<f32>) -> vec3<f32> {
    let e = 0.1;
    let dx = vec3<f32>(e, 0.0, 0.0);
    let dy = vec3<f32>(0.0, e, 0.0);
    let dz = vec3<f32>(0.0, 0.0, e);
    let x = (pn_potential(p + dy).z - pn_potential(p - dy).z)
          - (pn_potential(p + dz).y - pn_potential(p - dz).y);
    let y = (pn_potential(p + dz).x - pn_potential(p - dz).x)
          - (pn_potential(p + dx).z - pn_potential(p - dx).z);
    let z = (pn_potential(p + dx).y - pn_potential(p - dx).y)
          - (pn_potential(p + dy).x - pn_potential(p - dy).x);
    return vec3<f32>(x, y, z) / (2.0 * e);
}
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
    match (ctx.kind, name) {
        (_, "origin") => "params.origin.xyz".to_string(),
        (Kind::Sim, "position") => "p.position".to_string(),
        (Kind::Sim, "velocity") => "p.velocity".to_string(),
        (Kind::Sim, "lifetime") => "p.lifetime".to_string(),
        (Kind::Sim, "age") => "(p.max_lifetime - p.lifetime)".to_string(),
        (Kind::Sim, "seed") => "p.seed".to_string(),
        // Not meaningful in this kernel; resolve to a zero of the right shape.
        (_, "position") | (_, "velocity") => "vec3<f32>(0.0, 0.0, 0.0)".to_string(),
        _ => "0.0".to_string(),
    }
}

/// Resolve a property read to the right swizzle of its `vec4` uniform lane.
fn property_wgsl(name: &str, props: &[PropertyDecl]) -> String {
    match props.iter().find(|d| d.name == name).map(|d| d.default) {
        Some(PropertyValue::F32(_)) => format!("props.{name}.x"),
        Some(PropertyValue::Vec3(_)) => format!("props.{name}.xyz"),
        // Full vec4, or an unknown name (authoring error): read the whole lane.
        Some(PropertyValue::Vec4(_)) | None => format!("props.{name}"),
    }
}

/// The `Properties` uniform struct for a program's declared properties. Every
/// property is stored as a `vec4<f32>` lane (16-byte aligned, no packing), read
/// via the swizzle in [`property_wgsl`]. An empty set still declares one lane so
/// the binding and layout are uniform across every generated effect.
fn properties_struct(props: &[PropertyDecl]) -> String {
    let mut s = String::from("struct Properties {\n");
    if props.is_empty() {
        s.push_str("    _pad: vec4<f32>,\n");
    } else {
        for d in props {
            s.push_str(&format!("    {}: vec4<f32>,\n", d.name));
        }
    }
    s.push_str("};\n");
    s
}

/// Right-hand side for one node, referencing `v{child}` for its children.
fn lower_node(module: &Module, i: u32, ctx: Ctx) -> String {
    match module.get(ExprHandle(i)) {
        Expr::LitF32(v) => flit(*v),
        Expr::LitVec3([x, y, z]) => {
            format!("vec3<f32>({}, {}, {})", flit(*x), flit(*y), flit(*z))
        }
        Expr::Attribute(name) => attribute_wgsl(name, ctx),
        Expr::Property(name) => property_wgsl(name, ctx.props),
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
        Expr::Noise(a) => format!("value_noise(v{})", a.0),
        Expr::CurlNoise(a) => format!("curl_noise(v{})", a.0),
        Expr::Unary(op, a) => format!("{}(v{})", op.wgsl_fn(), a.0),
        Expr::Binary(op, a, b) => op.lower(&format!("v{}", a.0), &format!("v{}", b.0)),
        Expr::Ternary(op, a, b, c) => match op {
            TernaryOp::Mix => format!("mix(v{}, v{}, v{})", a.0, b.0, c.0),
            TernaryOp::Smoothstep => format!("smoothstep(v{}, v{}, v{})", a.0, b.0, c.0),
            // WGSL `select(false_value, true_value, condition)`; our Select is
            // `cond ? a : b`, so a is the true value, b the false value.
            TernaryOp::Select => format!("select(v{}, v{}, v{})", c.0, b.0, a.0),
        },
        Expr::Vec2(a, b) => format!("vec2<f32>(v{}, v{})", a.0, b.0),
        Expr::Vec3(a, b, c) => format!("vec3<f32>(v{}, v{}, v{})", a.0, b.0, c.0),
        Expr::Vec4(a, b, c, d) => format!("vec4<f32>(v{}, v{}, v{}, v{})", a.0, b.0, c.0, d.0),
        Expr::Swizzle(v, comp) => format!("v{}.{}", v.0, comp.wgsl()),
        Expr::Time => "params.time".to_string(),
        Expr::DeltaTime => "params.dt".to_string(),
        // In the simulate kernel a live particle reaches this code (`p` in scope);
        // in emit there is no particle yet, so it is trivially alive.
        Expr::IsAlive => match ctx.kind {
            Kind::Sim => "(p.lifetime > 0.0)".to_string(),
            Kind::Emit => "true".to_string(),
        },
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

/// The value type of a built-in attribute read (kernel-independent for typing;
/// availability per kernel is a separate concern). `None` for an unknown name.
fn attribute_value_type(name: &str) -> Option<ValueType> {
    match name {
        "origin" | "position" | "velocity" => Some(ValueType::Vec3),
        "lifetime" | "age" | "seed" => Some(ValueType::F32),
        _ => None,
    }
}

/// The value type an init modifier's expression must have to write `attr`.
fn set_attribute_type(attr: Attribute) -> ValueType {
    match attr {
        Attribute::Position | Attribute::Velocity | Attribute::Colour => ValueType::Vec3,
        Attribute::Lifetime | Attribute::Size => ValueType::F32,
    }
}

impl EffectProgram {
    /// Type-check the program's expression graph: infer every node's
    /// [`ValueType`], verify each op's operand arity and types, and verify each
    /// init / update modifier's expression matches the attribute it writes
    /// (`vec3` for position/velocity/colour and acceleration, `f32` for
    /// lifetime/size). Returns all errors found, or `Ok(())`.
    ///
    /// Call this before registering the effect to catch authoring errors with
    /// node-level messages, instead of leaving them to the WGSL compiler on the
    /// generated kernel.
    pub fn validate(&self) -> Result<(), Vec<TypeError>> {
        let props = &self.properties;
        let env = TypeEnv {
            attribute: &attribute_value_type,
            property: &|name: &str| {
                props.iter().find(|d| d.name == name).map(|d| match d.default {
                    PropertyValue::F32(_) => ValueType::F32,
                    PropertyValue::Vec3(_) => ValueType::Vec3,
                    PropertyValue::Vec4(_) => ValueType::Vec4,
                })
            },
        };

        // Node-level inference first; on failure that is the whole story.
        let types = match self.module.infer_types(&env) {
            Ok(t) => t,
            Err(e) => return Err(vec![e]),
        };

        // Then check each modifier root against the attribute it drives.
        let mut errors = Vec::new();
        let mut check = |handle: ExprHandle, expected: ValueType, what: &str| {
            let got = types[handle.0 as usize];
            if got != expected {
                errors.push(TypeError {
                    handle,
                    message: format!(
                        "{what} expects {:?} but expression is {:?}",
                        expected, got
                    ),
                });
            }
        };
        for m in &self.init {
            check(
                m.value,
                set_attribute_type(m.attribute),
                &format!("init {:?}", m.attribute),
            );
        }
        for op in &self.update {
            match op {
                UpdateOp::Accelerate(h) => check(*h, ValueType::Vec3, "accelerate"),
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Lower a program to its emit and simulate WGSL.
pub(crate) fn generate_program(program: &EffectProgram) -> GeneratedShaders {
    GeneratedShaders {
        emit: generate_emit(program),
        sim: generate_sim(program),
    }
}

/// Whether any node reachable from `roots` draws a random value, so a kernel
/// knows to seed an rng and pull in [`RNG_HELPERS`]. The emit kernel always seeds
/// one (it randomises `p.seed`); the simulate kernel only when `rand` is used.
fn uses_rng(module: &Module, roots: &[ExprHandle]) -> bool {
    module
        .reachable(roots)
        .iter()
        .any(|&i| matches!(module.get(ExprHandle(i)), Expr::Rand | Expr::RandUnit))
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
    let ctx = Ctx {
        kind: Kind::Emit,
        props: &program.properties,
    };
    let roots: Vec<ExprHandle> = program.init.iter().map(|m| m.value).collect();
    let lets = lower_block(module, &roots, ctx, "    ");
    let props_struct = properties_struct(&program.properties);

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
{props_struct}
struct Budget {{ count: atomic<u32> }};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: GenParams;
@group(0) @binding(2) var<storage, read_write> budget: Budget;
@group(0) @binding(3) var<uniform> props: Properties;
{RNG_HELPERS}
{NOISE_WGSL}
{PLANE_WGSL}
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
    p.position = lock_position({pos}, params.plane);
    p.velocity = lock_vector({vel}, params.plane);
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
    let ctx = Ctx {
        kind: Kind::Sim,
        props: &program.properties,
    };
    let roots: Vec<ExprHandle> = program
        .update
        .iter()
        .map(|op| match op {
            UpdateOp::Accelerate(h) => *h,
        })
        .collect();
    let lets = lower_block(module, &roots, ctx, "    ");
    let props_struct = properties_struct(&program.properties);

    // Per-particle rand in the update kernel: seed an rng from the
    // particle's stored seed, decorrelated per frame by the params seed. Only
    // pulled in when the update graph actually draws randoms.
    let (rng_helpers, rng_seed) = if uses_rng(module, &roots) {
        (
            RNG_HELPERS,
            "    var rng = hash_u32(bitcast<u32>(p.seed) ^ params.rng_seed ^ (i * 2654435761u));\n",
        )
    } else {
        ("", "")
    };

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
{props_struct}
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: GenParams;
@group(0) @binding(2) var<uniform> props: Properties;
{NOISE_WGSL}
{PLANE_WGSL}
{rng_helpers}
@compute @workgroup_size(64)
fn sim_main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= arrayLength(&particles)) {{ return; }}
    var p = particles[i];
    if (p.lifetime <= 0.0) {{ return; }}
    let dt = params.dt;
{rng_seed}{lets}
    var accel = vec3<f32>(0.0, 0.0, 0.0);
{accel}
    p.velocity = lock_vector(p.velocity + accel * dt, params.plane);
    p.position = p.position + p.velocity * dt;
    p.position = lock_position(p.position + params.space.xyz, params.plane);
    p.lifetime = p.lifetime - dt;
    particles[i] = p;
}}
"#
    )
}
