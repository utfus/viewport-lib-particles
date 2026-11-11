//! Showcase 16: Rendering.
//!
//! Billboard **orientation modes**, **screen-space size**, and **per-key alpha /
//! HDR colour**:
//! - An arrow sprite fountain drawn with a selectable [`Orient`] mode: parallel
//!   to the camera plane (the classic axis-aligned billboard), facing the camera
//!   *position* (spherical), along its velocity, or a fixed screen roll, so the
//!   asymmetric arrow makes the orientation visible. A toggle switches the same
//!   effect between world-space and screen-space size (distant sprites keep their
//!   pixel size).
//! - An HDR ember with a per-key **alpha** ramp that fades *in* at birth and out
//!   at death (a lifetime fade alone is always full at birth), its colour ramp
//!   HDR-bright so it blooms.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, Orient, ParticleBlend, ParticleItem,
    ParticleItems, ParticlePlugin, SpawnRate, SpawnShape, TextureMode, VelocityDist,
};

/// A 32x32 white arrow (triangle) pointing up on a transparent field, so a
/// billboard's orientation is legible.
fn arrow_rgba() -> (u32, u32, Vec<u8>) {
    let n = 32u32;
    let mut px = vec![0u8; (n * n * 4) as usize];
    for y in 0..n {
        for x in 0..n {
            let fx = x as f32 / (n - 1) as f32;
            let fy = y as f32 / (n - 1) as f32;
            // Apex at top-centre (fy = 0), widening to the base at fy = 1.
            let inside = (fx - 0.5).abs() < 0.45 * fy && fy > 0.08;
            let i = ((y * n + x) * 4) as usize;
            if inside {
                px[i] = 255;
                px[i + 1] = 255;
                px[i + 2] = 255;
                px[i + 3] = 255;
            }
        }
    }
    (n, n, px)
}

fn arrow_emitter() -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(24.0),
        lifetime: (1.6, 2.2),
        spawn: SpawnShape::Point,
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: 0.5,
            min_speed: 1.6,
            max_speed: 2.4,
        },
        colour: [1.0, 1.0, 1.0, 1.0],
        size: 0.5,
    }
}

/// One arrow variant with a given orientation + size mode. Uses the shared arrow
/// texture; alpha-blended so the arrow shape reads over the background.
fn arrow_variant(tex: viewport_lib_particles::ParticleTextureId, orient: Orient, screen: bool) -> EffectAsset {
    EffectAsset::new("rendering-arrow")
        .with_capacity(2_000)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(arrow_emitter())
        .force(ForceModifier::Accel([0.0, 0.0, -0.6]))
        .with_texture(tex)
        .with_texture_mode(TextureMode::Modulate)
        .with_orient(orient)
        .with_screen_size(screen)
        .with_gradient(
            Gradient::new().with_colour(vec![(0.0, [0.5, 0.9, 1.0]), (1.0, [0.2, 0.4, 0.8])]),
        )
}

/// HDR ember with a per-key alpha fade in *and* out.
fn ember() -> EffectAsset {
    EffectAsset::new("rendering-ember")
        .with_capacity(4_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(120.0),
            lifetime: (1.2, 2.0),
            spawn: SpawnShape::Circle {
                radius: 0.6,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.35,
                min_speed: 0.6,
                max_speed: 1.3,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.09,
        })
        .force(ForceModifier::Accel([0.0, 0.0, 0.5]))
        .force(ForceModifier::Drag(0.3))
        .with_gradient(
            Gradient::new()
                // HDR colours (> 1) so the ember blooms.
                .with_colour(vec![
                    (0.0, [2.2, 1.6, 0.7]),
                    (0.5, [1.8, 0.7, 0.15]),
                    (1.0, [0.7, 0.1, 0.02]),
                ])
                // Fade in over the first fifth of life, hold, then fade out.
                .with_alpha(vec![(0.0, 0.0), (0.2, 1.0), (0.7, 1.0), (1.0, 0.0)])
                .with_size(vec![(0.0, 0.5), (0.3, 1.0), (1.0, 0.4)]),
        )
}

/// Registers the eight arrow variants (4 orientations x world/screen size) plus
/// the ember, returning their ids. Uploads the arrow texture.
pub struct Ids {
    /// Indexed `[orient (0..4)] + [screen ? 4 : 0]`.
    pub arrows: Vec<EffectId>,
    pub ember: EffectId,
}

pub fn register(
    plugin: &mut ParticlePlugin,
    device: &viewport_lib::wgpu::Device,
    queue: &viewport_lib::wgpu::Queue,
) -> Ids {
    let (w, h, px) = arrow_rgba();
    let tex = plugin.upload_texture(device, queue, w, h, &px);
    let orients = [
        Orient::ParallelCameraPlane,
        Orient::FaceCamera,
        Orient::AlongVelocity,
        Orient::Fixed { radians: 0.6 },
    ];
    let mut arrows = Vec::with_capacity(8);
    for &screen in &[false, true] {
        for &o in &orients {
            arrows.push(plugin.add_effect(device, arrow_variant(tex, o, screen)));
        }
    }
    let ember = plugin.add_effect(device, ember());
    Ids { arrows, ember }
}

/// UI state.
pub struct State {
    /// Orientation index 0..4 (parallel, face-camera, along-velocity, fixed).
    pub orient: usize,
    /// Size in screen space (constant apparent size).
    pub screen_size: bool,
    /// Draw the HDR ember.
    pub ember: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            orient: 0,
            screen_size: false,
            ember: true,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Billboard orientation:");
    ui.radio_value(&mut state.orient, 0, "Parallel camera plane");
    ui.radio_value(&mut state.orient, 1, "Face camera position");
    ui.radio_value(&mut state.orient, 2, "Along velocity");
    ui.radio_value(&mut state.orient, 3, "Fixed roll (0.6 rad)");
    ui.separator();
    ui.checkbox(&mut state.screen_size, "Screen-space size");
    ui.checkbox(&mut state.ember, "HDR ember (per-key alpha)");
    ui.separator();
    ui.label("Arrow sprite shows the orientation; the ember fades in and out via a per-key alpha ramp and blooms via HDR colour.");
}

/// Submit the selected arrow variant and the ember.
pub fn items(ids: &Ids, state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let idx = state.orient.min(3) + if state.screen_size { 4 } else { 0 };
    if let Some(&id) = ids.arrows.get(idx) {
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
    }
    if state.ember {
        items.push(ParticleItem::new(ids.ember).at([1.6, 0.0, 0.0]));
    }
    items
}
