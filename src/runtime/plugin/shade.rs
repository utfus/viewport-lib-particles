//! Custom fragment-shading hook composition (Phase 33 / issue 018).
//!
//! An effect carrying a [`FragmentHook`](crate::effect::FragmentHook) computes its
//! colour from custom WGSL instead of the built-in `texture_mode`. This module
//! splices the hook body into the billboard draw shaders at the
//! `// __PARTICLE_HOOK_SLOT__` marker, producing a per-effect variant that is
//! compiled once and cached (by composed-source hash) like the codegen compute
//! pipelines.
//!
//! The hook contract mirrors viewport-lib's `MaterialPlugin` shading hooks: the
//! body defines a single function (`shade_particle`, the particle-domain analogue
//! of `shade_surface`), receives a surface struct (`ParticleFragment`, analogous
//! to `ShadingSurface`), and its symbols are prefixed with `{name}__` so a body is
//! portable and cannot collide with the template. Extra textures bind to
//! `htex0..` and params to a `hook_params` uniform, the same slot conventions the
//! mesh-side `MaterialPlugin` uses (`material_texture_0..` / a params window).

use super::*;
use crate::effect::FragmentHook;

/// Which billboard route a hook is composed for. They share the vertex stage and
/// the `ParticleFragment` build; only the fragment's return differs.
#[derive(Clone, Copy)]
pub(crate) enum HookRoute {
    /// Main HDR draw (`particle_draw.wgsl`): premultiplied `vec4` out.
    Draw,
    /// Weighted-blended OIT (`particle_oit.wgsl`): `OitOutput` via
    /// `viewport_oit_pack`.
    Oit,
}

/// The marker in the billboard templates where the hook is spliced.
const HOOK_SLOT: &str = "// __PARTICLE_HOOK_SLOT__";

/// Build the WGSL snippet for `hook` on `route`: the group-3 hook bindings, the
/// `ParticleFragment` struct, the (prefixed) hook body, and the `fs_hook` entry
/// point that builds the fragment and calls the hook. References the template's
/// `VsOut` / `flip_uv` / `sprite_tex` / `sprite_samp`, which precede the marker.
pub(crate) fn hook_snippet(hook: &FragmentHook, route: HookRoute) -> String {
    let name = &hook.name;
    // Minimal, module-local prefixing: rename the entry function. Helpers the body
    // declares stay as written — each hooked effect is its own shader module, so
    // there is no cross-effect symbol collision to guard against, only against the
    // template's reserved names (documented on `FragmentHook`).
    let body = hook
        .body
        .replace("fn shade_particle(", &format!("fn {name}__shade_particle("));

    let mut tex_decls = String::new();
    for i in 0..HOOK_TEXTURES {
        tex_decls.push_str(&format!(
            "@group(3) @binding({}) var htex{}: texture_2d<f32>;\n",
            4 + i,
            i
        ));
    }

    let (ret_ty, output) = match route {
        HookRoute::Draw => (
            "@location(0) vec4<f32>",
            "    return vec4<f32>(c.rgb * c.a, c.a);",
        ),
        HookRoute::Oit => ("OitOutput", "    return viewport_oit_pack(c.rgb, c.a, in.view_z);"),
    };

    let lanes = HOOK_PARAM_LANES;
    format!(
        r#"
struct HookParams {{ lanes: array<vec4<f32>, {lanes}> }};
@group(3) @binding(3) var<uniform> hook_params: HookParams;
{tex_decls}
struct ParticleFragment {{
    uv: vec2<f32>,
    texel: vec4<f32>,
    colour: vec4<f32>,
    age: f32,
    age_sec: f32,
    world: vec3<f32>,
    time: f32,
}};

{body}

@fragment
fn fs_hook(in: VsOut) -> {ret_ty} {{
    let huv = flip_uv(in.uv, in.age, in.age_sec);
    var f: ParticleFragment;
    f.uv = huv;
    f.texel = textureSample(sprite_tex, sprite_samp, huv);
    f.colour = in.color;
    f.age = in.age;
    f.age_sec = in.age_sec;
    f.world = in.world;
    f.time = hook_params.lanes[0].w;
    let c = {name}__shade_particle(f);
{output}
}}
"#
    )
}

/// Compose a full custom billboard shader source: `prelude` (shared bindings, and
/// for OIT the shared OIT helpers) + the route `template` with the hook snippet
/// spliced at the marker. The entry point for the custom fragment is `fs_hook`.
pub(crate) fn compose_billboard(
    prelude: &str,
    template: &str,
    hook: &FragmentHook,
    route: HookRoute,
) -> String {
    let snippet = hook_snippet(hook, route);
    let body = template.replace(HOOK_SLOT, &snippet);
    format!("{prelude}\n{body}")
}
