// Painter entry points for the folded billboard shader. Appended to
// particle_soft.wgsl, with SHARED_PAINTER_MASK_WGSL prepended. The painter pass
// binds the renderer's depth-read group at group 3, whose bindings 4 and 5 are
// this frame's 2D light layers.

@group(3) @binding(4) var painter_light_tex: texture_2d_array<f32>;
@group(3) @binding(5) var<uniform> painter_lights: PainterLights;

// A premultiplied colour lit by the light layer of the instance's painter
// range, read under the fragment. An unlit placement, and a layer no range
// holds, keep their colour.
fn painter_lit(c: vec4<f32>, frag: vec4<f32>) -> vec4<f32> {
    let slice = painter_light_slice(painter_lights, draw_params.painter_layer);
    if (slice < 0 || draw_params.painter_unlit != 0u) {
        return c;
    }
    // The scene depth shares the render target's size, so it turns the
    // framebuffer position into NDC for a light layer of any size.
    let dims = vec2<f32>(textureDimensions(scene_depth));
    let ndc = vec2<f32>(frag.x / dims.x * 2.0 - 1.0, 1.0 - frag.y / dims.y * 2.0);
    let texel = painter_mask_texel(ndc, textureDimensions(painter_light_tex));
    let light = textureLoad(painter_light_tex, texel, slice, 0);
    return vec4<f32>(painter_light_apply(c.rgb, light, painter_lights), c.a);
}

@fragment
fn fs_painter(in: VsOut) -> @location(0) vec4<f32> {
    return painter_lit(shade(in), in.pos);
}

// Multiply darkens content that is lit already, so it takes no light of its
// own, as viewport-lib's sprites do.
@fragment
fn fs_painter_multiply(in: VsOut) -> @location(0) vec4<f32> {
    let c = shade(in);
    return vec4<f32>(c.rgb + vec3<f32>(1.0 - c.a), c.a);
}
