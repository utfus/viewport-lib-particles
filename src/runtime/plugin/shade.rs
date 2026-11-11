//! Custom shading hook composition.
//!
//! An effect carrying a [`FragmentHook`](crate::effect::FragmentHook) computes its
//! colour, and optionally moves its corners, with custom WGSL. This module
//! splices the hook body into the billboard draw shaders, producing a
//! per-effect variant that is compiled once and cached (by composed-source
//! hash) like the codegen compute pipelines.
//!
//! The contract is viewport-lib's sprite material contract (see
//! [`FragmentHook`](crate::effect::FragmentHook)), so one body draws a sprite
//! and a particle: a body defines `shade_particle(f: ParticleFragment)` or
//! `shade_sprite(f: SpriteFragment)` for the colour and `displace_particle(v:
//! ParticleVertex)` or `displace_sprite(v: SpriteVertex)` for the corners, reads
//! textures as `htex0..` through `sprite_samp` and params as
//! `hook_params.lanes[1..]` or by name as `params.<name>`.

use super::*;
use crate::effect::FragmentHook;

/// Which billboard route a hook is composed for. They share the vertex stage and
/// the fragment build; the bindings and the fragment's output differ.
#[derive(Clone, Copy)]
pub(crate) enum HookRoute {
    /// Main HDR draw (`particle_draw.wgsl`): premultiplied `vec4` out. Hook
    /// bindings in group 3.
    Draw,
    /// Weighted-blended OIT (`particle_oit.wgsl`): `OitOutput` via
    /// `viewport_oit_pack`. Hook bindings in group 3.
    Oit,
    /// The folded billboard shader (`particle_soft.wgsl` with
    /// `particle_painter.wgsl`), for the soft pass and the painter pass: the
    /// soft fade, and 2D light in the painter entries. Hook bindings fold into
    /// group 1 after the billboard ones, since group 3 is the depth-read group.
    Folded,
}

/// The marker in the billboard templates where the hook is spliced.
const HOOK_SLOT: &str = "// __PARTICLE_HOOK_SLOT__";
/// The markers around the templates' default `particle_displace`.
const DISPLACE_BEGIN: &str = "// __PARTICLE_DISPLACE_BEGIN__";
const DISPLACE_END: &str = "// __PARTICLE_DISPLACE_END__";

/// What a sprite material body receives, field for field as viewport-lib's
/// `sprite_contract.wgsl` declares it. The first seven fields are
/// `ParticleFragment`'s.
const SPRITE_FRAGMENT: &str = r#"
struct SpriteFragment {
    uv:          vec2<f32>,
    texel:       vec4<f32>,
    colour:      vec4<f32>,
    age:         f32,
    age_sec:     f32,
    world:       vec3<f32>,
    time:        f32,
    local_uv:    vec2<f32>,
    uv_rect:     vec4<f32>,
    flash:       f32,
    dissolve:    f32,
    palette_row: u32,
    uv_ddx:      vec2<f32>,
    uv_ddy:      vec2<f32>,
};
"#;

/// Build the WGSL snippet for `hook` on `route`: the hook bindings, the
/// fragment structs, the named `params`, the (prefixed) body, and the entry
/// points that build the fragment and call it. References the template's
/// `VsOut`, `flip_uv`, `base_colour`, `sprite_tex` and `sprite_samp`, which
/// precede the marker.
pub(crate) fn hook_snippet(hook: &FragmentHook, route: HookRoute) -> String {
    let name = &hook.name;
    // Minimal, module-local prefixing: rename the entry functions. Helpers the
    // body declares stay as written: each hooked effect is its own shader
    // module, so there is no cross-effect symbol collision to guard against,
    // only against the template's reserved names (documented on
    // `FragmentHook`).
    let mut body = hook.body.clone();
    for f in ["shade_particle", "shade_sprite", "displace_particle", "displace_sprite"] {
        body = body.replace(&format!("fn {f}("), &format!("fn {name}__{f}("));
    }

    let (group, first) = match route {
        HookRoute::Draw | HookRoute::Oit => (3, 3),
        HookRoute::Folded => (1, 5),
    };
    let mut tex_decls = String::new();
    for i in 0..HOOK_TEXTURES {
        tex_decls.push_str(&format!(
            "@group({group}) @binding({}) var htex{i}: texture_2d<f32>;\n",
            first + 1 + i,
        ));
    }

    // The colour: the body's, or the built-in one for a body that only moves
    // the corners.
    let shades_sprite = body.contains(&format!("fn {name}__shade_sprite("));
    let call = if body.contains(&format!("fn {name}__shade_particle(")) {
        format!("{name}__shade_particle(f)")
    } else if shades_sprite {
        format!("{name}__shade_sprite(hook_sprite_fragment(f, in, huv))")
    } else {
        "base_colour(in)".to_string()
    };
    let sprite = if shades_sprite {
        format!(
            "{SPRITE_FRAGMENT}
// A sprite body's fragment, from the particle's. A particle has no flash,
// dissolve or palette row of its own; those read zero.
fn hook_sprite_fragment(f: ParticleFragment, in: VsOut, huv: vec2<f32>) -> SpriteFragment {{
    var s: SpriteFragment;
    s.uv = f.uv;
    s.texel = f.texel;
    s.colour = f.colour;
    s.age = f.age;
    s.age_sec = f.age_sec;
    s.world = f.world;
    s.time = f.time;
    s.local_uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    s.uv_rect = vec4<f32>(flip_uv(vec2<f32>(0.0), in.age, in.age_sec), flip_uv(vec2<f32>(1.0), in.age, in.age_sec));
    s.flash = 0.0;
    s.dissolve = 0.0;
    s.palette_row = 0u;
    s.uv_ddx = dpdx(huv);
    s.uv_ddy = dpdy(huv);
    return s;
}}
"
        )
    } else {
        String::new()
    };

    let (params_decl, params_fill) = params_wgsl(hook);

    let entries = match route {
        HookRoute::Draw => r#"
@fragment
fn fs_hook(in: VsOut) -> @location(0) vec4<f32> {
    let c = hook_shade(in);
    return vec4<f32>(c.rgb * c.a, c.a);
}

@fragment
fn fs_hook_multiply(in: VsOut) -> @location(0) vec4<f32> {
    let c = hook_shade(in);
    return vec4<f32>(mix(vec3<f32>(1.0), c.rgb, c.a), c.a);
}
"#,
        HookRoute::Oit => r#"
@fragment
fn fs_hook(in: VsOut) -> OitOutput {
    let c = hook_shade(in);
    return viewport_oit_pack(c.rgb, c.a, in.view_z);
}
"#,
        HookRoute::Folded => r#"
// Premultiplied and faded, as the folded shader's `shade`.
fn hook_shade_faded(in: VsOut) -> vec4<f32> {
    let c = hook_shade(in);
    let a = c.a * soft_fade(in);
    return vec4<f32>(c.rgb * a, a);
}

@fragment
fn fs_hook(in: VsOut) -> @location(0) vec4<f32> {
    return hook_shade_faded(in);
}

@fragment
fn fs_hook_multiply(in: VsOut) -> @location(0) vec4<f32> {
    let c = hook_shade_faded(in);
    return vec4<f32>(c.rgb + vec3<f32>(1.0 - c.a), c.a);
}

@fragment
fn fs_hook_painter(in: VsOut) -> @location(0) vec4<f32> {
    return painter_lit(hook_shade_faded(in), in.pos);
}

@fragment
fn fs_hook_painter_multiply(in: VsOut) -> @location(0) vec4<f32> {
    let c = hook_shade_faded(in);
    return vec4<f32>(c.rgb + vec3<f32>(1.0 - c.a), c.a);
}
"#,
    };

    let lanes = HOOK_PARAM_LANES;
    let params_binding = first;
    format!(
        r#"
struct HookParams {{ lanes: array<vec4<f32>, {lanes}> }};
@group({group}) @binding({params_binding}) var<uniform> hook_params: HookParams;
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
// A sprite material's vertex body reads the same corner.
alias SpriteVertex = ParticleVertex;
{sprite}{params_decl}
{body}

// The hook's straight-alpha colour.
fn hook_shade(in: VsOut) -> vec4<f32> {{
{params_fill}    let huv = flip_uv(in.uv, in.age, in.age_sec);
    var f: ParticleFragment;
    f.uv = huv;
    f.texel = textureSample(sprite_tex, sprite_samp, huv);
    f.colour = in.color;
    f.age = in.age;
    f.age_sec = in.age_sec;
    f.world = in.world;
    f.time = hook_params.lanes[0].w;
    return {call};
}}
{entries}"#
    )
}

/// The `params` struct a body with named params reads, and the line that
/// fills it from the lanes, as viewport-lib's sprite materials declare theirs.
/// Empty for a hook without names.
fn params_wgsl(hook: &FragmentHook) -> (String, String) {
    if hook.param_names.is_empty() {
        return (String::new(), String::new());
    }
    let mut decl = String::from("struct HookMaterialParams {\n");
    for d in &hook.param_names {
        decl.push_str(&format!("    {}: {},\n", d.name, d.kind.wgsl()));
    }
    decl.push_str("};\nvar<private> params: HookMaterialParams;\n");
    decl.push_str("fn hook_material_params() -> HookMaterialParams {\n    var p: HookMaterialParams;\n");
    for (i, d) in hook.param_names.iter().enumerate() {
        let swizzle = match d.kind.wgsl() {
            "f32" => ".x",
            "vec2<f32>" => ".xy",
            "vec3<f32>" => ".xyz",
            _ => "",
        };
        decl.push_str(&format!("    p.{} = hook_params.lanes[{}]{swizzle};\n", d.name, i + 1));
    }
    decl.push_str("    return p;\n}\n");
    (decl, "    params = hook_material_params();\n".to_string())
}

/// Compose a full custom billboard shader source: `prelude` (shared bindings,
/// and the route's shared helpers) + the route `template` with the hook snippet
/// spliced at the marker and, when the body moves corners, the template's
/// default `particle_displace` replaced by a call to it.
pub(crate) fn compose_billboard(
    prelude: &str,
    template: &str,
    hook: &FragmentHook,
    route: HookRoute,
) -> String {
    let snippet = hook_snippet(hook, route);
    let mut body = template.replace(HOOK_SLOT, &snippet);
    let name = &hook.name;
    let displace = if hook.body.contains("fn displace_particle(") {
        Some("displace_particle")
    } else if hook.body.contains("fn displace_sprite(") {
        Some("displace_sprite")
    } else {
        None
    };
    if let (Some(f), Some(begin), Some(end)) =
        (displace, body.find(DISPLACE_BEGIN), body.find(DISPLACE_END))
    {
        body.replace_range(
            begin..end + DISPLACE_END.len(),
            &format!(
                "fn particle_displace(v: ParticleVertex) -> vec3<f32> {{\n{}    \
                 return {name}__{f}(v);\n}}",
                params_wgsl(hook).1
            ),
        );
    }
    format!("{prelude}\n{body}")
}
