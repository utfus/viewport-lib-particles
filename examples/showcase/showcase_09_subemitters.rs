//! Showcase 9: Sub-emitters.
//!
//! Particles that spawn particles on the GPU. A parent effect's simulate pass
//! appends spawn events; a child effect's emit pass turns them into new
//! particles, seeded at the parent's position and inheriting a share of its
//! velocity.
//!
//! Two scenes:
//! - Fireworks: shells launch upward, arc under gravity, and burst into a
//!   spherical spray of sparks on death.
//! - Rocket smoke: rockets climb and lay down a smoke trail, emitting puffs
//!   every step along their path.
//!
//! Because a parent names its child by id, the effects are registered in a
//! dedicated `register` step (child first) rather than a plain `presets` list.

use eframe::egui;
use viewport_lib::wgpu;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, EventCondition, ForceModifier, Gradient, ParticleBlend,
    ParticleItem, ParticleItems, ParticlePlugin, SpawnRate, SpawnShape, SubEmitter, VelocityDist,
};

/// The registered effect ids for both scenes. A parent is submitted alongside
/// its child every frame: the parent drives the simulation and appends events,
/// the child consumes them and draws.
pub struct Ids {
    pub firework_shell: EffectId,
    pub firework_sparks: EffectId,
    pub rocket: EffectId,
    pub rocket_smoke: EffectId,
}

/// The spark burst: additive embers that fly out from the shell's death point,
/// fall under gravity, and cool from white through orange to a dim red. Rate
/// zero, so every spark comes from a shell's death event.
fn firework_sparks() -> EffectAsset {
    EffectAsset::new("firework-sparks")
        .with_capacity(80_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(0.0),
            lifetime: (0.7, 1.4),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 3.0,
                min_speed: 2.0,
                max_speed: 4.5,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.09,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -7.0]))
        .force(ForceModifier::Drag(0.6))
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [3.0, 2.4, 1.2]),
                    (0.4, [2.0, 0.6, 0.15]),
                    (1.0, [0.6, 0.05, 0.02]),
                ])
                .with_size(vec![(0.0, 1.0), (1.0, 0.3)]),
        )
}

/// The shell: a bright ember launched upward that arcs under gravity and, on
/// death near its apex, bursts into sparks (inheriting a little of its upward
/// momentum).
fn firework_shell(sparks: EffectId) -> EffectAsset {
    EffectAsset::new("firework-shell")
        .with_capacity(128)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(1.8),
            lifetime: (1.3, 1.7),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.2,
                min_speed: 5.5,
                max_speed: 6.5,
            },
            colour: [1.5, 1.2, 0.6, 1.0],
            size: 0.12,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -6.0]))
        .with_sub_emitter(
            SubEmitter::new(sparks, EventCondition::OnDeath, 220).with_inherit_velocity(0.15),
        )
}

/// The smoke: soft puffs that trail behind the rocket, drift slowly, and fade
/// from warm exhaust to cool grey as they grow. Rate zero: every puff is a
/// rocket event.
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

/// The rocket: a bright climbing ember that emits smoke every step along its
/// path (the puffs inherit a little of its velocity).
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
            colour: [2.0, 1.4, 0.7, 1.0],
            size: 0.13,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -1.2]))
        .with_sub_emitter(
            SubEmitter::new(smoke, EventCondition::EveryStep, 2).with_inherit_velocity(0.1),
        )
}

/// Register both scenes' effects, children before parents so a parent can name
/// its child.
pub fn register(plugin: &mut ParticlePlugin, device: &wgpu::Device) -> Ids {
    let firework_sparks = plugin.add_effect(device, firework_sparks());
    let firework_shell = plugin.add_effect(device, firework_shell(firework_sparks));
    let rocket_smoke = plugin.add_effect(device, rocket_smoke());
    let rocket = plugin.add_effect(device, rocket_body(rocket_smoke));
    Ids {
        firework_shell,
        firework_sparks,
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
}

impl Default for State {
    fn default() -> Self {
        Self {
            scene: Scene::Fireworks,
            paused: false,
            position: [0.0, 0.0, -2.0],
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Scene:");
    ui.radio_value(&mut state.scene, Scene::Fireworks, "Fireworks (burst on death)");
    ui.radio_value(&mut state.scene, Scene::Rocket, "Rocket (smoke trail)");
    ui.separator();
    ui.checkbox(&mut state.paused, "Pause");
    ui.separator();
    ui.label("Launch position:");
    ui.add(egui::Slider::new(&mut state.position[0], -5.0..=5.0).text("x"));
    ui.add(egui::Slider::new(&mut state.position[1], -5.0..=5.0).text("y"));
    ui.add(egui::Slider::new(&mut state.position[2], -4.0..=2.0).text("z"));
    if ui.button("Reset").clicked() {
        *state = State::default();
    }
    ui.separator();
    match state.scene {
        Scene::Fireworks => {
            ui.label("Shells launch upward and burst into sparks the moment");
            ui.label("they die. Each spark inherits a share of the shell's");
            ui.label("upward momentum, then falls under gravity.");
        }
        Scene::Rocket => {
            ui.label("Rockets climb and append a smoke event every step, so");
            ui.label("the trail follows the exact path each rocket took.");
        }
    }
}

/// Build this frame's submission: the selected scene's parent and child, both at
/// the launch position. Pausing hides both, which stops their simulation.
pub fn items(ids: &Ids, state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let (parent, child) = match state.scene {
        Scene::Fireworks => (ids.firework_shell, ids.firework_sparks),
        Scene::Rocket => (ids.rocket, ids.rocket_smoke),
    };
    let mut parent_item = ParticleItem::new(parent).at(state.position);
    let mut child_item = ParticleItem::new(child).at(state.position);
    parent_item.settings.hidden = state.paused;
    child_item.settings.hidden = state.paused;
    // Submit the parent first so it drives its events; order does not matter to
    // the plugin, which sequences parent simulate before child emit itself.
    items.push(parent_item);
    items.push(child_item);
    items
}
