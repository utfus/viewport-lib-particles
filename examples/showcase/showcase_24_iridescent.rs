//! Showcase 24: Iridescent bubbles (custom fragment hook).
//!
//! The one thing no built-in colour mode can do: **view-dependent colour**. Each
//! particle is a soft bubble whose thin-film sheen is computed in a custom WGSL
//! `shade_particle` hook from the *view direction* (`camera.eye_pos - world`), a
//! faked spherical normal from `uv`, and a noise texture, so the bubbles shift
//! through the rainbow as you orbit the camera. Alpha-blended, so they draw
//! through the hooked OIT pipeline.
//!
//! Orbit the view (left / middle drag) to see the colours move.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, FragmentHook, Orient, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, SpawnShape, VelocityDist,
};

/// A soft low-frequency value-noise texture (`size x size` RGBA8) for organic
/// per-bubble film-thickness variation. A coarse random grid, bilinearly smoothed.
fn film_noise(size: usize) -> Vec<u8> {
    // Coarse grid of pseudo-random values (deterministic hash, no rng crate).
    // Kept low-frequency so the film reads as broad soft regions, not a fine maze.
    let g = 5usize;
    let hash = |x: usize, y: usize| -> f32 {
        let mut h = (x as u32).wrapping_mul(374761393) ^ (y as u32).wrapping_mul(668265263);
        h = (h ^ (h >> 13)).wrapping_mul(1274126177);
        ((h ^ (h >> 16)) & 0xffff) as f32 / 65535.0
    };
    let mut px = vec![0u8; size * size * 4];
    for y in 0..size {
        for x in 0..size {
            let fx = x as f32 / size as f32 * g as f32;
            let fy = y as f32 / size as f32 * g as f32;
            let (x0, y0) = (fx.floor() as usize % g, fy.floor() as usize % g);
            let (x1, y1) = ((x0 + 1) % g, (y0 + 1) % g);
            let (tx, ty) = (fx.fract(), fy.fract());
            // Smoothstep the interpolants for a soft field.
            let sx = tx * tx * (3.0 - 2.0 * tx);
            let sy = ty * ty * (3.0 - 2.0 * ty);
            let a = hash(x0, y0) * (1.0 - sx) + hash(x1, y0) * sx;
            let b = hash(x0, y1) * (1.0 - sx) + hash(x1, y1) * sx;
            let v = (a * (1.0 - sy) + b * sy).clamp(0.0, 1.0);
            let l = (v * 255.0) as u8;
            let i = (y * size + x) * 4;
            px[i..i + 4].copy_from_slice(&[l, l, l, 255]);
        }
    }
    px
}

/// The hook: thin-film iridescence driven by the view angle to the particle
/// (`camera.eye_pos`), a faked spherical normal from `uv` (so the sprite reads as
/// a bubble), and a noise texture. Params: `lanes[1] = (thickness, hue_speed, .., ..)`.
const BUBBLE_HOOK: &str = "\
fn iri(t: f32) -> vec3<f32> {
    return 0.5 + 0.5 * cos(6.2831853 * (vec3<f32>(t, t, t) + vec3<f32>(0.0, 0.33, 0.67)));
}
fn shade_particle(f: ParticleFragment) -> vec4<f32> {
    // Faked spherical bubble normal across the sprite; discard outside the disc.
    let d = f.uv * 2.0 - vec2<f32>(1.0, 1.0);
    let r2 = dot(d, d);
    if (r2 > 1.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let nz = sqrt(max(1.0 - r2, 0.0));

    // World-space view direction to the particle: orbiting the camera changes
    // this, which rotates the film hue. That is the whole point of the demo.
    let view = normalize(camera.eye_pos - f.world);
    let azimuth = atan2(view.y, view.x);

    let rim = pow(1.0 - nz, 2.5);              // bright glancing edge
    let noise = textureSample(htex0, sprite_samp, f.uv).r;
    let thickness = hook_params.lanes[1].x;
    let hue_speed = hook_params.lanes[1].y;

    // Keep the rim's contribution to the *hue* small so the edge does not band
    // into a tight rainbow; the rim still drives the alpha (edge brightness).
    let t = azimuth * hue_speed + rim * 0.4 + noise * thickness + f.time * 0.04;
    // Ease the saturation toward a soft base so the sheen reads as a film, not neon.
    let colour = mix(vec3<f32>(0.6, 0.62, 0.65), iri(t), 0.7);

    // Mostly transparent with a luminous iridescent rim, fading over life.
    let a = clamp(rim * 1.3 + 0.05, 0.0, 1.0) * (1.0 - f.age * 0.4);
    return vec4<f32>(colour, a);
}";

fn bubbles() -> EffectAsset {
    let hook = FragmentHook::new("bubble", BUBBLE_HOOK)
        .texture(64, 64, film_noise(64))
        .param([1.2, 1.0, 0.0, 0.0]); // (thickness, hue_speed, _, _)

    EffectAsset::new("iridescent_bubbles")
        .with_capacity(6_000)
        // Alpha blend routes through the OIT pass (order-independent), right for
        // transparent bubbles and exercising the hooked OIT pipeline.
        .with_blend(ParticleBlend::Alpha)
        // Face the camera so each sprite reads as a round bubble.
        .with_orient(Orient::FaceCamera)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(1.0),
            lifetime: (5.0, 8.0),
            spawn: SpawnShape::Sphere {
                radius: 1.5,
                volume: true,
            },
            // Slow, gently rising drift.
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.6,
                min_speed: 0.15,
                max_speed: 0.4,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.7,
        })
        .with_fragment_hook(hook)
}

/// The preset effects this showcase registers, in menu order.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("Iridescent bubbles", bubbles())]
}

/// UI state.
pub struct State {
    pub selected: usize,
    pub paused: bool,
    pub position: [f32; 3],
    /// Live spawn rate (per-frame emitter override).
    pub rate: f32,
    /// Live bubble size (per-frame emitter override).
    pub size: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            selected: 0,
            paused: false,
            position: [0.0, 0.0, 0.0],
            rate: 1.0,
            size: 0.7,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Preset:");
    let names: Vec<&'static str> = presets().into_iter().map(|(n, _)| n).collect();
    for (i, name) in names.iter().enumerate() {
        ui.selectable_value(&mut state.selected, i, *name);
    }
    ui.separator();
    ui.checkbox(&mut state.paused, "Pause");
    ui.separator();
    ui.label("Emitter position:");
    ui.add(egui::Slider::new(&mut state.position[0], -5.0..=5.0).text("x"));
    ui.add(egui::Slider::new(&mut state.position[1], -5.0..=5.0).text("y"));
    ui.add(egui::Slider::new(&mut state.position[2], -2.0..=5.0).text("z"));
    ui.separator();
    ui.add(egui::Slider::new(&mut state.rate, 0.1..=50.0).text("spawn rate"));
    ui.add(egui::Slider::new(&mut state.size, 0.2..=1.5).text("bubble size"));
    ui.separator();
    ui.label("Colour is computed per fragment by a custom WGSL hook:");
    ui.label("- thin-film sheen from the view direction (camera.eye_pos)");
    ui.label("- a faked spherical normal from uv makes each sprite a bubble");
    ui.label("- orbit the camera: the hues shift with the view angle,");
    ui.label("  which no built-in colour mode can express.");
}

/// Build this frame's submission.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.get(state.selected) {
        // Live rate / size via a per-frame emitter override.
        let emitter = Emitter {
            rate: SpawnRate::PerSecond(state.rate),
            lifetime: (5.0, 8.0),
            spawn: SpawnShape::Sphere {
                radius: 1.5,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.6,
                min_speed: 0.15,
                max_speed: 0.4,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: state.size,
        };
        let mut item = ParticleItem::new(id).at(state.position).with_emitter(emitter);
        item.settings.hidden = state.paused;
        items.push(item);
    }
    items
}
