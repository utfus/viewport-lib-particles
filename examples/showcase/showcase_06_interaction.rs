//! Showcase 6: Interaction.
//!
//! Wires the particle plugin's interaction hooks: the fountain casts blob
//! shadows onto a ground plane under a directional light (`cast_shadow_pass`),
//! and clicking the cloud picks the system through the shared GPU pick pass
//! (`render_pick`). Picking marks the system selected, which drives the
//! selection outline through `outline_mask`.

use eframe::egui;
use viewport_lib::renderer::PickId;
use viewport_lib::{
    BackfacePolicy, LightKind, LightSource, LightingSettings, Material, MeshId, SceneRenderItem,
};
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, SpawnShape, VelocityDist,
};

/// Pick id assigned to the demo particle system.
pub const PICK_ID: u64 = 7;

/// A fountain that arcs up and falls, casting shadows onto the ground.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![(
        "Fountain",
        EffectAsset::new("interaction_fountain")
            .with_capacity(20_000)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(8_000.0),
                lifetime: (1.4, 2.2),
                spawn: SpawnShape::Sphere {
                    radius: 0.15,
                    volume: true,
                },
                velocity: VelocityDist::UniformCone {
                    axis: [0.0, 0.0, 1.0],
                    half_angle: 0.5,
                    min_speed: 3.5,
                    max_speed: 5.5,
                },
                colour: [0.6, 0.8, 1.0, 1.0],
                size: 0.14,
            })
            .force(ForceModifier::Accel([0.0, 0.0, -5.0]))
            .with_gradient(
                Gradient::new()
                    .with_colour(vec![
                        (0.0, [0.7, 1.3, 1.8]),
                        (0.6, [0.3, 0.6, 1.2]),
                        (1.0, [0.1, 0.15, 0.4]),
                    ])
                    .with_size(vec![(0.0, 1.0), (1.0, 0.5)]),
            ),
    )]
}

/// A ground plane (a wide flat cuboid) with its top surface at z = 0.
pub fn ground_item(mesh: MeshId) -> SceneRenderItem {
    let mut ground = SceneRenderItem::default();
    ground.mesh_id = mesh;
    ground.model =
        glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.25)).to_cols_array_2d();
    ground.material = Material::from_colour([0.82, 0.80, 0.74]);
    ground.material.roughness = 0.9;
    ground.material.backface_policy = BackfacePolicy::Cull;
    ground
}

/// A directional light (overhead sun in Z-up) casting cascaded shadows.
pub fn lighting() -> LightingSettings {
    let mut ls = LightingSettings::default();
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.35, 0.25, 0.9],
    };
    ls.lights = vec![light];
    ls.shadows_enabled = true;
    ls.shadow_cascade_count = 3;
    ls
}

/// UI state.
pub struct State {
    pub picked: bool,
    pub position: [f32; 3],
}

impl Default for State {
    fn default() -> Self {
        Self {
            picked: false,
            position: [0.0, 0.0, 0.0],
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Click the particle cloud to pick it.");
    ui.separator();
    if state.picked {
        ui.colored_label(egui::Color32::LIGHT_GREEN, "PICKED: fountain (id 7)");
    } else {
        ui.label("Nothing picked.");
    }
    if ui.button("Clear selection").clicked() {
        state.picked = false;
    }
    ui.separator();
    ui.label("Emitter position:");
    ui.add(egui::Slider::new(&mut state.position[0], -5.0..=5.0).text("x"));
    ui.add(egui::Slider::new(&mut state.position[1], -5.0..=5.0).text("y"));
    ui.add(egui::Slider::new(&mut state.position[2], 0.0..=5.0).text("z"));
    ui.separator();
    ui.label("The fountain casts shadows onto the ground plane; the");
    ui.label("directional light drives cascaded shadow maps.");
}

/// Build this frame's particle submission, tagged with the pick id and the
/// current selection state.
pub fn items(effect_id: EffectId, state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    let mut item = ParticleItem::new(effect_id).at(state.position);
    item.settings.pick_id = PickId(PICK_ID);
    item.settings.selected = state.picked;
    item.settings.cast_shadows = true;
    items.push(item);
    items
}
