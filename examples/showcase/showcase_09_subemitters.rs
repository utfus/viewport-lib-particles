//! Showcase 9: Sub-emitters.
//!
//! Particles that spawn particles on the GPU. A parent effect's simulate pass
//! appends spawn events; a child effect's emit pass turns them into new
//! particles, seeded at the parent's position and inheriting a share of its
//! velocity.
//!
//! The fireworks scene chains three levels: a comet shell rises and, on death,
//! breaks into a willow of sparks; each spark, on its own death, crackles into a
//! shower of glitter. The plugin sequences the passes so each level consumes its
//! parent's events the same frame (shell simulate -> spark emit, spark simulate
//! -> glitter emit). Shells, sparks, and the rocket draw as ribbon trails, so
//! they leave comet and willow streaks rather than dots. Four colour palettes
//! launch together for a grand-finale sky.
//!
//! Because a parent names its child by id, the effects are registered in a
//! dedicated `register` step (children before parents) rather than a `presets`
//! list.

use eframe::egui;
use viewport_lib::wgpu;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, EventCondition, ForceModifier, Gradient, ParticleBlend,
    ParticleItem, ParticleItems, ParticleRender, ParticlePlugin, SpawnRate, SpawnShape, SubEmitter,
    VelocityDist,
};

/// A firework colour: the shell/comet tint and the three-key colour ramp the
/// sparks cool through over their life.
struct Palette {
    tint: [f32; 4],
    spark: [(f32, [f32; 3]); 3],
}

/// The four finale colours: gold, emerald, sapphire, magenta.
const PALETTES: [Palette; 4] = [
    Palette {
        tint: [2.2, 1.5, 0.6, 1.0],
        spark: [
            (0.0, [3.2, 2.6, 1.2]),
            (0.45, [2.4, 1.0, 0.2]),
            (1.0, [0.7, 0.08, 0.02]),
        ],
    },
    Palette {
        tint: [0.7, 2.2, 0.9, 1.0],
        spark: [
            (0.0, [2.4, 3.2, 2.0]),
            (0.45, [0.3, 2.0, 0.7]),
            (1.0, [0.02, 0.5, 0.15]),
        ],
    },
    Palette {
        tint: [0.7, 1.2, 2.4, 1.0],
        spark: [
            (0.0, [2.2, 2.8, 3.4]),
            (0.45, [0.4, 0.9, 2.6]),
            (1.0, [0.05, 0.12, 0.6]),
        ],
    },
    Palette {
        tint: [2.2, 0.7, 1.8, 1.0],
        spark: [
            (0.0, [3.2, 2.0, 3.0]),
            (0.45, [2.2, 0.4, 1.6]),
            (1.0, [0.5, 0.03, 0.35]),
        ],
    },
];

/// A three-level firework chain for one colour.
pub struct Chain {
    pub shell: EffectId,
    pub sparks: EffectId,
    pub glitter: EffectId,
}

/// The registered effect ids: one chain per palette, plus the rocket pair. A
/// parent is submitted alongside its children every frame so it drives the
/// simulation and appends events while the children consume and draw them.
pub struct Ids {
    pub chains: Vec<Chain>,
    pub rocket: EffectId,
    pub rocket_smoke: EffectId,
}

/// The shell emitter, parameterised so the launch rate and horizontal spread can
/// be tuned live from the controls (rebuilt each frame as a per-frame override).
fn shell_emitter(tint: [f32; 4], rate: f32, spread: f32) -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(rate),
        lifetime: (1.2, 1.6),
        spawn: SpawnShape::Box {
            min: [-spread, -spread * 0.4, 0.0],
            max: [spread, spread * 0.4, 0.0],
        },
        velocity: VelocityDist::UniformCone {
            axis: [0.0, 0.0, 1.0],
            half_angle: 0.16,
            min_speed: 5.2,
            max_speed: 6.6,
        },
        colour: tint,
        size: 0.12,
    }
}

/// The shell: a comet that climbs, arcs under gravity, and on death breaks into
/// sparks (which inherit a little of its upward momentum). Drawn as a ribbon
/// trail, so it reads as a rising comet.
fn shell_asset(sparks: EffectId, tint: [f32; 4]) -> EffectAsset {
    EffectAsset::new("firework-shell")
        .with_capacity(256)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(shell_emitter(tint, 1.0, 2.6))
        .force(ForceModifier::Accel([0.0, 0.0, -6.2]))
        .with_render(ParticleRender::Trail {
            width: 0.05,
            segments: 16,
        })
        .with_sub_emitter(
            SubEmitter::new(sparks, EventCondition::OnDeath, 150).with_inherit_velocity(0.12),
        )
}

/// The sparks: a willow of embers thrown out from the shell's break, cooling
/// through the palette ramp as they fall. Drawn as trails for the drooping
/// willow streak. On death each spark crackles into glitter. Rate zero: every
/// spark is a shell event.
fn spark_asset(glitter: EffectId, ramp: [(f32, [f32; 3]); 3]) -> EffectAsset {
    EffectAsset::new("firework-sparks")
        .with_capacity(24_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(0.0),
            lifetime: (0.9, 1.5),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 2.7,
                min_speed: 2.4,
                max_speed: 5.0,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.05,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -8.0]))
        .force(ForceModifier::Drag(0.7))
        .with_render(ParticleRender::Trail {
            width: 0.03,
            segments: 10,
        })
        .with_gradient(
            Gradient::new()
                .with_colour(ramp.to_vec())
                .with_size(vec![(0.0, 1.0), (1.0, 0.25)]),
        )
        .with_sub_emitter(
            SubEmitter::new(glitter, EventCondition::OnDeath, 6).with_inherit_velocity(0.15),
        )
}

/// The glitter: a brief bright sparkle at each spark's death point, the crackle
/// at the tips of the willow. Small round billboards, very short life. Rate
/// zero: every fleck is a spark event.
fn glitter_asset() -> EffectAsset {
    EffectAsset::new("firework-glitter")
        .with_capacity(40_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(0.0),
            lifetime: (0.25, 0.55),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 3.0,
                min_speed: 0.5,
                max_speed: 1.6,
            },
            colour: [3.0, 2.6, 1.7, 1.0],
            size: 0.05,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -5.0]))
        .with_gradient(
            Gradient::new().with_size(vec![(0.0, 0.3), (0.2, 1.0), (1.0, 0.0)]),
        )
}

/// The smoke: soft puffs trailing the rocket, drifting and fading from warm
/// exhaust to cool grey as they grow. Rate zero: every puff is a rocket event.
fn rocket_smoke() -> EffectAsset {
    EffectAsset::new("rocket-smoke")
        .with_capacity(60_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(0.0),
            lifetime: (1.2, 2.2),
            spawn: SpawnShape::Sphere {
                radius: 0.05,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, -1.0],
                half_angle: 0.7,
                min_speed: 0.2,
                max_speed: 0.6,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.14,
        })
        .force(ForceModifier::Accel([0.0, 0.0, 0.4]))
        .force(ForceModifier::Drag(0.8))
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [0.9, 0.6, 0.35]),
                    (0.3, [0.35, 0.35, 0.4]),
                    (1.0, [0.05, 0.05, 0.07]),
                ])
                .with_size(vec![(0.0, 0.4), (1.0, 1.6)]),
        )
}

/// The rocket: a climbing comet (ribbon trail) that emits smoke every step along
/// its path (the puffs inherit a little of its velocity).
fn rocket_body(smoke: EffectId) -> EffectAsset {
    EffectAsset::new("rocket")
        .with_capacity(64)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(1.2),
            lifetime: (1.8, 2.4),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.15,
                min_speed: 3.5,
                max_speed: 4.5,
            },
            colour: [2.4, 1.6, 0.8, 1.0],
            size: 0.13,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -1.2]))
        .with_render(ParticleRender::Trail {
            width: 0.045,
            segments: 14,
        })
        .with_sub_emitter(
            SubEmitter::new(smoke, EventCondition::EveryStep, 2).with_inherit_velocity(0.1),
        )
}

/// Register both scenes' effects, children before parents so a parent can name
/// its child. Each palette is its own three-level chain (glitter, then sparks
/// that break into it, then the shell that breaks into the sparks).
pub fn register(plugin: &mut ParticlePlugin, device: &wgpu::Device) -> Ids {
    let mut chains = Vec::with_capacity(PALETTES.len());
    for palette in &PALETTES {
        let glitter = plugin.add_effect(device, glitter_asset());
        let sparks = plugin.add_effect(device, spark_asset(glitter, palette.spark));
        let shell = plugin.add_effect(device, shell_asset(sparks, palette.tint));
        chains.push(Chain {
            shell,
            sparks,
            glitter,
        });
    }
    let rocket_smoke = plugin.add_effect(device, rocket_smoke());
    let rocket = plugin.add_effect(device, rocket_body(rocket_smoke));
    Ids {
        chains,
        rocket,
        rocket_smoke,
    }
}

/// Which sub-emitter scene is showing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Scene {
    Fireworks,
    Rocket,
}

/// UI state for the sub-emitter showcase.
pub struct State {
    pub scene: Scene,
    pub paused: bool,
    pub position: [f32; 3],
    /// Shells launched per second, per colour.
    pub rate: f32,
    /// Half-width of the launch box.
    pub spread: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            scene: Scene::Fireworks,
            paused: false,
            position: [0.0, 0.0, -2.5],
            rate: 1.1,
            spread: 2.6,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Scene:");
    ui.radio_value(&mut state.scene, Scene::Fireworks, "Fireworks (three-level break)");
    ui.radio_value(&mut state.scene, Scene::Rocket, "Rocket (smoke trail)");
    ui.separator();
    ui.checkbox(&mut state.paused, "Pause");
    if state.scene == Scene::Fireworks {
        ui.add(egui::Slider::new(&mut state.rate, 0.1..=3.0).text("launch rate"));
        ui.add(egui::Slider::new(&mut state.spread, 0.0..=5.0).text("spread"));
    }
    ui.separator();
    ui.label("Launch position:");
    ui.add(egui::Slider::new(&mut state.position[0], -6.0..=6.0).text("x"));
    ui.add(egui::Slider::new(&mut state.position[1], -6.0..=6.0).text("y"));
    ui.add(egui::Slider::new(&mut state.position[2], -4.0..=2.0).text("z"));
    if ui.button("Reset").clicked() {
        *state = State::default();
    }
    ui.separator();
    match state.scene {
        Scene::Fireworks => {
            ui.label("Comet shells rise and break into a willow of sparks, and");
            ui.label("each spark crackles into glitter as it dies. Four colours");
            ui.label("launch together, each its own parent -> child -> grandchild");
            ui.label("chain spawned entirely on the GPU.");
        }
        Scene::Rocket => {
            ui.label("A rocket climbs and appends a smoke event every step, so");
            ui.label("the trail follows the exact path it took.");
        }
    }
}

/// Build this frame's submission. Pausing hides every instance, which stops the
/// simulation.
pub fn items(ids: &Ids, state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let hidden = state.paused;
    match state.scene {
        Scene::Fireworks => {
            for (i, chain) in ids.chains.iter().enumerate() {
                let tint = PALETTES[i].tint;
                // The shell rate and spread come from the sliders (a per-frame
                // emitter override); the children draw what the shells break into.
                let mut shell = ParticleItem::new(chain.shell)
                    .at(state.position)
                    .with_emitter(shell_emitter(tint, state.rate, state.spread));
                let mut sparks = ParticleItem::new(chain.sparks).at(state.position);
                let mut glitter = ParticleItem::new(chain.glitter).at(state.position);
                shell.settings.hidden = hidden;
                sparks.settings.hidden = hidden;
                glitter.settings.hidden = hidden;
                items.push(shell);
                items.push(sparks);
                items.push(glitter);
            }
        }
        Scene::Rocket => {
            let mut rocket = ParticleItem::new(ids.rocket).at(state.position);
            let mut smoke = ParticleItem::new(ids.rocket_smoke).at(state.position);
            rocket.settings.hidden = hidden;
            smoke.settings.hidden = hidden;
            items.push(rocket);
            items.push(smoke);
        }
    }
    items
}
