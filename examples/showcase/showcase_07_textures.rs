//! Showcase 7: Textures (Phase 12).
//!
//! Three billboard effects driven by procedurally-synthesised textures (no asset
//! files): a soft smoke puff (`TextureMode::Modulate`, alpha-blended), an
//! explosion flipbook animating through a 4x4 atlas over each particle's life,
//! and a masked spark (`TextureMode::ModulateAlphaFromR`, the texture's red
//! channel is the alpha mask). Textures need a queue to upload, so this module
//! registers through [`register`] rather than the `presets()` used elsewhere.

use eframe::egui;
use viewport_lib::wgpu;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, Flipbook, ForceModifier, ParticleBlend, ParticleItem,
    ParticleItems, ParticlePlugin, SpawnRate, SpawnShape, TextureMode, VelocityDist,
};

/// Atlas grid for the explosion flipbook.
const EXPLOSION_COLS: u32 = 4;
const EXPLOSION_ROWS: u32 = 4;
const CELL: u32 = 64;

/// A soft radial puff: white rgb, gaussian-ish alpha falloff to the edge.
fn smoke_pixels() -> (Vec<u8>, u32, u32) {
    let size = 64u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let r = (u * u + v * v).sqrt();
            let a = (1.0 - smoothstep(0.15, 1.0, r)).clamp(0.0, 1.0);
            px.extend_from_slice(&[225, 228, 235, (a * 255.0) as u8]);
        }
    }
    (px, size, size)
}

/// A 4x4 fireball atlas: each cell is a later frame of an expanding, fading,
/// white-hot-to-red blast.
fn explosion_pixels() -> (Vec<u8>, u32, u32) {
    let w = EXPLOSION_COLS * CELL;
    let h = EXPLOSION_ROWS * CELL;
    let cells = EXPLOSION_COLS * EXPLOSION_ROWS;
    let mut px = vec![0u8; (w * h * 4) as usize];
    for cell in 0..cells {
        let cx = cell % EXPLOSION_COLS;
        let cy = cell / EXPLOSION_COLS;
        let t = cell as f32 / (cells - 1) as f32;
        // Blast expands then the shell thins; brightness fades over the sheet.
        let rad = 0.18 + t * 0.9;
        let bright = (1.0 - t).powf(1.3);
        // Colour shifts white-hot -> orange -> deep red.
        let hot = [1.6, 1.5, 1.2];
        let cool = [1.4, 0.25, 0.05];
        for ly in 0..CELL {
            for lx in 0..CELL {
                let u = (lx as f32 + 0.5) / CELL as f32 * 2.0 - 1.0;
                let v = (ly as f32 + 0.5) / CELL as f32 * 2.0 - 1.0;
                let r = (u * u + v * v).sqrt();
                // A filled disc early, hollowing to a shell as it expands.
                let outer = 1.0 - smoothstep(rad * 0.75, rad, r);
                let inner = smoothstep(rad * (0.1 + 0.5 * t) - 0.15, rad * (0.1 + 0.5 * t), r);
                let mask = (outer * inner).clamp(0.0, 1.0);
                let col = [
                    hot[0] + (cool[0] - hot[0]) * t,
                    hot[1] + (cool[1] - hot[1]) * t,
                    hot[2] + (cool[2] - hot[2]) * t,
                ];
                let i = (((cy * CELL + ly) * w + (cx * CELL + lx)) * 4) as usize;
                px[i] = tone(col[0] * bright * mask);
                px[i + 1] = tone(col[1] * bright * mask);
                px[i + 2] = tone(col[2] * bright * mask);
                px[i + 3] = (mask * 255.0) as u8;
            }
        }
    }
    (px, w, h)
}

/// A four-point star in the red channel, used as an alpha mask.
fn spark_mask_pixels() -> (Vec<u8>, u32, u32) {
    let size = 64u32;
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            let v = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
            // Star = bright core + thin axis-aligned spikes.
            let r = (u * u + v * v).sqrt();
            let core = (1.0 - smoothstep(0.0, 0.35, r)).clamp(0.0, 1.0);
            let spike_h = (1.0 - smoothstep(0.02, 0.12, v.abs())) * (1.0 - smoothstep(0.2, 0.95, u.abs()));
            let spike_v = (1.0 - smoothstep(0.02, 0.12, u.abs())) * (1.0 - smoothstep(0.2, 0.95, v.abs()));
            let m = core.max(spike_h.max(spike_v)).clamp(0.0, 1.0);
            let r8 = (m * 255.0) as u8;
            px.extend_from_slice(&[r8, r8, r8, 255]);
        }
    }
    (px, size, size)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Tone-map an HDR channel value into an 8-bit sRGB-store byte (the texture is
/// sampled as sRGB, so store the perceptual value).
fn tone(v: f32) -> u8 {
    let c = (v / (1.0 + v)).clamp(0.0, 1.0);
    (c.powf(1.0 / 2.2) * 255.0) as u8
}

/// Upload the three textures and register their effects, returning the ids
/// aligned with [`Preset`].
pub fn register(
    plugin: &mut ParticlePlugin,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Vec<EffectId> {
    let (smoke, sw, sh) = smoke_pixels();
    let smoke_tex = plugin.upload_texture(device, queue, sw, sh, &smoke);
    let (expl, ew, eh) = explosion_pixels();
    let expl_tex = plugin.upload_texture(device, queue, ew, eh, &expl);
    let (spark, kw, kh) = spark_mask_pixels();
    let spark_tex = plugin.upload_texture(device, queue, kw, kh, &spark);

    let smoke_effect = EffectAsset::new("smoke")
        .with_capacity(6_000)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(60.0),
            lifetime: (2.5, 3.5),
            spawn: SpawnShape::Sphere {
                radius: 0.25,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.35,
                min_speed: 0.6,
                max_speed: 1.1,
            },
            colour: [0.7, 0.72, 0.8, 0.5],
            size: 1.6,
        })
        .force(ForceModifier::Drag(0.3))
        .with_texture(smoke_tex)
        .with_texture_mode(TextureMode::Modulate);

    let explosion_effect = EffectAsset::new("explosion")
        .with_capacity(8_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(90.0),
            lifetime: (0.8, 1.1),
            spawn: SpawnShape::Sphere {
                radius: 0.6,
                volume: true,
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 1.4,
        })
        .with_texture(expl_tex)
        .with_texture_mode(TextureMode::Modulate)
        .with_flipbook(Flipbook::new(EXPLOSION_COLS, EXPLOSION_ROWS));

    let spark_effect = EffectAsset::new("spark")
        .with_capacity(12_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(400.0),
            lifetime: (0.9, 1.6),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: std::f32::consts::PI,
                min_speed: 2.5,
                max_speed: 5.0,
            },
            colour: [1.5, 0.9, 0.4, 1.0],
            size: 0.5,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -4.0]))
        .with_texture(spark_tex)
        .with_texture_mode(TextureMode::ModulateAlphaFromR);

    vec![
        plugin.add_effect(device, smoke_effect),
        plugin.add_effect(device, explosion_effect),
        plugin.add_effect(device, spark_effect),
    ]
}

/// The presets in menu order, aligned with the ids from [`register`].
const PRESETS: &[&str] = &["Smoke sprite", "Explosion flipbook", "Masked spark"];

/// UI state for the textures showcase.
pub struct State {
    pub selected: usize,
    pub paused: bool,
    pub position: [f32; 3],
}

impl Default for State {
    fn default() -> Self {
        Self {
            selected: 0,
            paused: false,
            position: [0.0, 0.0, 0.0],
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Preset:");
    for (i, name) in PRESETS.iter().enumerate() {
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
    ui.label("All textures are synthesised procedurally (no asset files):");
    ui.label("a soft smoke puff, a 4x4 explosion atlas played as a");
    ui.label("flipbook over each particle's life, and a star mask");
    ui.label("whose red channel drives the spark's alpha.");
}

/// Build this frame's submission for the selected preset.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.get(state.selected) {
        let mut item = ParticleItem::new(id).at(state.position);
        item.settings.hidden = state.paused;
        items.push(item);
    }
    items
}
