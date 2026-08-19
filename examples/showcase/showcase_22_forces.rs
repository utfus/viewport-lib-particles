//! Showcase 22: Forces and kill regions.
//!
//! One base effect driven entirely by per-frame overrides, so each modifier gets
//! an emitter, palette, and force set tuned to show it off: a vortex that winds
//! particles into a glowing tornado, a radial burst, a conform-to-sphere swarm
//! that collapses onto a shell, and a kill sphere / kill box that carves its
//! silhouette out of a filled slab of particles. The strength slider (vortex /
//! radial) is live.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, Gradient, Modifier, ParticleBlend, ParticleItem, ParticleItems,
    SpawnRate, SpawnShape, VelocityDist,
};

/// The base effect. Its emitter / gradient / modifiers are all replaced per frame
/// by the selected demo, so only its capacity and blend matter here.
fn base() -> EffectAsset {
    EffectAsset::new("forces")
        .with_capacity(40_000)
        .with_blend(ParticleBlend::Additive)
}

pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("forces", base())]
}

/// Which modifier to show off.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Demo {
    Vortex,
    Radial,
    Conform,
    KillSphere,
    KillBox,
}

/// UI state.
pub struct State {
    pub demo: Demo,
    /// Strength for the vortex / radial demos.
    pub strength: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            demo: Demo::Vortex,
            strength: 10.0,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Modifier:");
    ui.radio_value(&mut state.demo, Demo::Vortex, "Vortex tornado");
    ui.radio_value(&mut state.demo, Demo::Radial, "Radial burst");
    ui.radio_value(&mut state.demo, Demo::Conform, "Conform to sphere");
    ui.radio_value(&mut state.demo, Demo::KillSphere, "Kill sphere (void)");
    ui.radio_value(&mut state.demo, Demo::KillBox, "Kill box (void)");
    ui.separator();
    if matches!(state.demo, Demo::Vortex | Demo::Radial) {
        ui.add(egui::Slider::new(&mut state.strength, -24.0..=24.0).text("strength"));
        ui.label("Negative reverses the swirl / pulls inward.");
    } else {
        ui.label("Conform and the kill regions use fixed parameters.");
    }
}

/// A gradient with HDR colour keys (so additive particles bloom) plus a fade-in
/// / fade-out alpha and a size taper.
fn glow(colours: Vec<(f32, [f32; 3])>) -> Gradient {
    Gradient::new()
        .with_colour(colours)
        .with_alpha(vec![(0.0, 0.0), (0.15, 1.0), (0.8, 1.0), (1.0, 0.0)])
        .with_size(vec![(0.0, 0.4), (0.25, 1.0), (1.0, 0.25)])
}

/// The full per-frame override for one demo: where to place the emitter, plus its
/// emitter, gradient, and modifiers.
struct DemoConfig {
    origin: [f32; 3],
    emitter: Emitter,
    gradient: Gradient,
    modifiers: Vec<Modifier>,
}

fn config(state: &State) -> DemoConfig {
    let up_cone = |half, lo, hi| VelocityDist::UniformCone {
        axis: [0.0, 0.0, 1.0],
        half_angle: half,
        min_speed: lo,
        max_speed: hi,
    };
    match state.demo {
        // Wind rising particles into a tornado: strong tangential swirl plus a
        // gentle inward pull so they spiral up a column instead of flying out.
        Demo::Vortex => DemoConfig {
            origin: [0.0, 0.0, 0.0],
            emitter: Emitter {
                rate: SpawnRate::PerSecond(16_000.0),
                lifetime: (2.2, 3.2),
                spawn: SpawnShape::Circle {
                    radius: 1.6,
                    volume: false,
                },
                velocity: up_cone(0.25, 1.4, 2.6),
                colour: [1.0, 1.0, 1.0, 1.0],
                size: 0.08,
            },
            gradient: glow(vec![
                (0.0, [0.4, 1.6, 2.2]),
                (0.5, [1.4, 0.6, 2.0]),
                (1.0, [1.6, 0.2, 0.8]),
            ]),
            modifiers: vec![
                Modifier::Accel([0.0, 0.0, -0.6]),
                Modifier::Vortex {
                    center: [0.0, 0.0, 1.5],
                    axis: [0.0, 0.0, 1.0],
                    strength: state.strength,
                    falloff: 0.0,
                },
                Modifier::Radial {
                    center: [0.0, 0.0, 1.5],
                    strength: -3.0,
                    falloff: 0.0,
                },
            ],
        },
        // An omnidirectional burst: particles spawn in a small ball at the centre
        // with a small outward seed velocity (so none sit exactly on the radial
        // centre), then the radial blasts them out equally in every direction.
        Demo::Radial => DemoConfig {
            origin: [0.0, 0.0, 1.6],
            emitter: Emitter {
                rate: SpawnRate::PerSecond(12_000.0),
                lifetime: (1.4, 2.0),
                spawn: SpawnShape::Sphere {
                    radius: 0.15,
                    volume: true,
                },
                velocity: VelocityDist::UniformCone {
                    axis: [0.0, 0.0, 1.0],
                    half_angle: std::f32::consts::PI, // full sphere
                    min_speed: 1.0,
                    max_speed: 2.0,
                },
                colour: [1.0, 1.0, 1.0, 1.0],
                size: 0.09,
            },
            gradient: glow(vec![
                (0.0, [2.4, 1.7, 0.6]),
                (0.5, [1.8, 0.7, 0.15]),
                (1.0, [0.8, 0.15, 0.05]),
            ]),
            // Radial centred on the spawn point, so the push is symmetric.
            modifiers: vec![Modifier::Radial {
                center: [0.0, 0.0, 1.6],
                strength: state.strength,
                falloff: 0.0,
            }],
        },
        // A wide, slow cloud pulled onto a sphere shell: a glowing orb of swarm.
        Demo::Conform => DemoConfig {
            origin: [0.0, 0.0, 0.0],
            emitter: Emitter {
                rate: SpawnRate::PerSecond(14_000.0),
                lifetime: (2.5, 3.5),
                spawn: SpawnShape::Sphere {
                    radius: 2.6,
                    volume: true,
                },
                velocity: up_cone(std::f32::consts::PI, 0.2, 0.6),
                colour: [1.0, 1.0, 1.0, 1.0],
                size: 0.07,
            },
            gradient: glow(vec![
                (0.0, [0.5, 2.2, 1.2]),
                (0.6, [0.3, 1.6, 1.8]),
                (1.0, [0.2, 0.7, 1.6]),
            ]),
            modifiers: vec![Modifier::Conform {
                center: [0.0, 0.0, 1.5],
                radius: 1.5,
                attraction: 14.0,
                influence: 4.0,
                sticky: 1.2,
            }],
        },
        // A wide, thin, densely-filled slab of near-static particles: the kill
        // region carves out its own silhouette (a sphere / box-shaped void) that
        // reads clearly, because the slab is only one particle deep toward the eye.
        Demo::KillSphere => DemoConfig {
            origin: [0.0, 0.0, 0.0],
            emitter: kill_slab(),
            gradient: glow(vec![
                (0.0, [2.4, 1.4, 0.4]),
                (0.5, [1.8, 0.6, 0.2]),
                (1.0, [0.9, 0.2, 0.1]),
            ]),
            modifiers: vec![Modifier::KillSphere {
                center: [0.0, 0.0, 2.4],
                radius: 1.2,
                kill_inside: true,
            }],
        },
        Demo::KillBox => DemoConfig {
            origin: [0.0, 0.0, 0.0],
            emitter: kill_slab(),
            gradient: glow(vec![
                (0.0, [1.0, 2.2, 0.6]),
                (0.5, [0.5, 1.6, 0.4]),
                (1.0, [0.2, 0.8, 0.3]),
            ]),
            modifiers: vec![Modifier::KillAabb {
                min: [-1.5, -1.0, 1.6],
                max: [1.5, 1.0, 3.2],
                kill_inside: true,
            }],
        },
    }
}

/// A wide, tall, thin slab of particles spawned throughout its volume with
/// almost no motion, so it stays filled and a kill region leaves a clean hole.
/// Thin along Y (toward the camera) so front-to-back overlap does not fill the
/// void back in.
fn kill_slab() -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(40_000.0),
        lifetime: (1.0, 1.6),
        spawn: SpawnShape::Box {
            min: [-3.2, -0.25, 0.3],
            max: [3.2, 0.25, 4.6],
        },
        velocity: VelocityDist::UniformBox {
            min: [-0.15, -0.15, -0.15],
            max: [0.15, 0.15, 0.15],
        },
        colour: [1.0, 1.0, 1.0, 1.0],
        size: 0.055,
    }
}

/// Submit the base effect with the selected demo's overrides.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.first() {
        let cfg = config(state);
        items.push(
            ParticleItem::new(id)
                .at(cfg.origin)
                .with_emitter(cfg.emitter)
                .with_gradient(cfg.gradient)
                .with_modifiers(cfg.modifiers),
        );
    }
    items
}
