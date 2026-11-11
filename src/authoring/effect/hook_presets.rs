//! Ready-made [`FragmentHook`]s for viewport-lib's sprite effects: flash,
//! dissolve, palette swap and outline, with the same maths, so a particle
//! flashes or dissolves the way a sprite does.
//!
//! Each colours the particle as its tint times its texture (the
//! [`TextureMode::Modulate`](crate::TextureMode::Modulate) look) before its
//! effect, and declares named params that
//! [`ParticleItem::with_hook_param`](crate::ParticleItem::with_hook_param) sets
//! per instance.

use super::*;
use viewport_lib_types::plugin::material::ParamKind;

/// Smooth value noise in 0..1 on an 8x8 lattice per unit, as the sprite
/// dissolve's built-in noise.
const VALUE_NOISE: &str = r#"
fn fx_hash(p: vec2<f32>) -> f32 {
    let q = fract(p * vec2<f32>(0.1031, 0.1030));
    let r = q + dot(q, q.yx + 33.33);
    return fract((r.x + r.y) * r.x);
}

fn fx_value_noise(p: vec2<f32>) -> f32 {
    let g = p * 8.0;
    let i = floor(g);
    let f = fract(g);
    let u = f * f * (3.0 - 2.0 * f);
    let a = fx_hash(i);
    let b = fx_hash(i + vec2<f32>(1.0, 0.0));
    let c = fx_hash(i + vec2<f32>(0.0, 1.0));
    let d = fx_hash(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
"#;

impl FragmentHook {
    /// Mix the particle's colour towards `colour` by `amount`, 0 to 1, as a
    /// sprite's flash. Params `flash_colour` (vec3) and `flash` (f32): set
    /// `flash` per instance for a hit flash.
    pub fn flash(colour: [f32; 3], amount: f32) -> Self {
        Self::new(
            "flash",
            "fn shade_particle(f: ParticleFragment) -> vec4<f32> {
    let c = f.colour * f.texel;
    return vec4<f32>(mix(c.rgb, params.flash_colour, params.flash), c.a);
}
",
        )
        .named_param("flash_colour", ParamKind::Vec3, [colour[0], colour[1], colour[2], 0.0])
        .named_param("flash", ParamKind::F32, [amount, 0.0, 0.0, 0.0])
    }

    /// Dissolve the particle against noise, as a sprite's dissolve: fragments
    /// whose noise is below `threshold` (0 to 1) are cut, and those within
    /// `edge_width` above it take `edge_colour`. The noise is `noise` when
    /// given (its red channel, tiled `noise_scale` times across the quad, read
    /// after the sRGB decode every hook texture gets), else built-in value
    /// noise at that scale. Params `dissolve`, `edge_colour`,
    /// `edge_width` and `noise_scale`; animate `dissolve` per instance.
    pub fn dissolve(
        threshold: f32,
        edge_colour: [f32; 3],
        edge_width: f32,
        noise_scale: f32,
        noise: Option<InlineTexture>,
    ) -> Self {
        let has_noise = noise.is_some();
        let mut hook = Self::new(
            "dissolve",
            format!(
                "{VALUE_NOISE}
fn shade_sprite(f: SpriteFragment) -> vec4<f32> {{
    var c = f.colour * f.texel;
    let p = f.local_uv * params.noise_scale;
    var n = fx_value_noise(p);
    if ({has_noise}) {{
        n = textureSampleLevel(htex0, sprite_samp, fract(p), 0.0).r;
    }}
    if (n < params.dissolve || params.dissolve >= 1.0) {{
        c = vec4<f32>(c.rgb, 0.0);
    }} else if (n < params.dissolve + params.edge_width) {{
        c = vec4<f32>(params.edge_colour, c.a);
    }}
    return c;
}}
"
            ),
        )
        .named_param("dissolve", ParamKind::F32, [threshold, 0.0, 0.0, 0.0])
        .named_param(
            "edge_colour",
            ParamKind::Vec3,
            [edge_colour[0], edge_colour[1], edge_colour[2], 0.0],
        )
        .named_param("edge_width", ParamKind::F32, [edge_width, 0.0, 0.0, 0.0])
        .named_param("noise_scale", ParamKind::F32, [noise_scale, 0.0, 0.0, 0.0]);
        if let Some(n) = noise {
            hook.textures.push(n);
        }
        hook
    }

    /// Recolour the texture through a palette, as a sprite's palette swap: the
    /// texel's red channel (or its luminance, when `luminance`) picks a column
    /// of `palette` and `row` picks the row, so one sheet takes many palettes.
    /// The texel's alpha is kept. Param `palette_row` (f32); set it per
    /// instance to give instances their own palettes.
    pub fn palette_swap(palette: InlineTexture, row: u32, luminance: bool) -> Self {
        let mut hook = Self::new(
            "palette_swap",
            format!(
                "fn shade_particle(f: ParticleFragment) -> vec4<f32> {{
    var v = f.texel.r;
    if ({luminance}) {{
        v = dot(f.texel.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    }}
    let dims = textureDimensions(htex0);
    let column = u32(round(clamp(v, 0.0, 1.0) * f32(dims.x - 1u)));
    let row = min(u32(max(params.palette_row, 0.0)), dims.y - 1u);
    let p = textureLoad(htex0, vec2<i32>(i32(column), i32(row)), 0);
    return f.colour * vec4<f32>(p.rgb, p.a * f.texel.a);
}}
"
            ),
        )
        .named_param("palette_row", ParamKind::F32, [row as f32, 0.0, 0.0, 0.0]);
        hook.textures.push(palette);
        hook
    }

    /// Draw an outline of `colour` around the texture's opaque texels,
    /// `width_px` device pixels wide, as a sprite's outline. A sprite grows its
    /// quad to fit the outline; a particle cannot, so the outline stays inside
    /// the quad and needs transparent margin in the texture or flipbook cell.
    /// Params `outline_colour` (vec4) and `outline_px` (f32).
    pub fn outline(colour: [f32; 4], width_px: f32) -> Self {
        Self::new(
            "outline",
            "// The texture's alpha at `uv`, zero outside the cell so an outline never
// picks up a neighbouring flipbook cell.
fn outline_alpha(uv: vec2<f32>, rect: vec4<f32>, ddx: vec2<f32>, ddy: vec2<f32>) -> f32 {
    let lo = min(rect.xy, rect.zw);
    let hi = max(rect.xy, rect.zw);
    if (any(uv < lo) || any(uv > hi)) {
        return 0.0;
    }
    return textureSampleGrad(sprite_tex, sprite_samp, uv, ddx, ddy).a;
}

fn shade_sprite(f: SpriteFragment) -> vec4<f32> {
    let c = f.colour * f.texel;
    // One device pixel in UV along each axis, times the width.
    let px = vec2<f32>(abs(f.uv_ddx.x) + abs(f.uv_ddy.x), abs(f.uv_ddx.y) + abs(f.uv_ddy.y));
    let o = px * params.outline_px;
    let d = o * 0.70710678;
    let r = f.uv_rect;
    var m = outline_alpha(f.uv + vec2<f32>(o.x, 0.0), r, f.uv_ddx, f.uv_ddy);
    m = max(m, outline_alpha(f.uv - vec2<f32>(o.x, 0.0), r, f.uv_ddx, f.uv_ddy));
    m = max(m, outline_alpha(f.uv + vec2<f32>(0.0, o.y), r, f.uv_ddx, f.uv_ddy));
    m = max(m, outline_alpha(f.uv - vec2<f32>(0.0, o.y), r, f.uv_ddx, f.uv_ddy));
    m = max(m, outline_alpha(f.uv + d, r, f.uv_ddx, f.uv_ddy));
    m = max(m, outline_alpha(f.uv - d, r, f.uv_ddx, f.uv_ddy));
    m = max(m, outline_alpha(f.uv + vec2<f32>(d.x, -d.y), r, f.uv_ddx, f.uv_ddy));
    m = max(m, outline_alpha(f.uv + vec2<f32>(-d.x, d.y), r, f.uv_ddx, f.uv_ddy));
    // The particle over its outline.
    let ring = params.outline_colour.a * m * (1.0 - c.a);
    let a = c.a + ring;
    let rgb = (c.rgb * c.a + params.outline_colour.rgb * ring) / max(a, 1e-5);
    return vec4<f32>(rgb, a);
}
",
        )
        .named_param("outline_colour", ParamKind::Vec4, colour)
        .named_param("outline_px", ParamKind::F32, [width_px, 0.0, 0.0, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_validate() {
        let tex = InlineTexture {
            width: 4,
            height: 2,
            rgba: vec![255; 32],
        };
        for hook in [
            FragmentHook::flash([1.0, 1.0, 1.0], 0.5),
            FragmentHook::dissolve(0.3, [1.0, 0.5, 0.0], 0.05, 1.0, None),
            FragmentHook::dissolve(0.3, [1.0, 0.5, 0.0], 0.05, 1.0, Some(tex.clone())),
            FragmentHook::palette_swap(tex, 1, false),
            FragmentHook::outline([0.0, 0.0, 0.0, 1.0], 2.0),
        ] {
            hook.validate().unwrap_or_else(|e| panic!("{}: {e}", hook.name));
            assert_eq!(hook.param_names.len(), hook.params.len());
        }
        assert_eq!(FragmentHook::flash([1.0; 3], 0.0).param_lane("flash"), Some(2));
    }
}
