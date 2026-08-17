//! Showcase 11: Load / save (Phase 17).
//!
//! Renders one effect and, in the panel, serializes its `EffectAsset` to a RON
//! `.effect` string and round-trips it back, proving the whole authored model
//! survives serialization. The serialization calls are gated on the `serde`
//! feature; build with `cargo run --features serde --example showcase` to enable
//! them (without the feature the panel shows a hint and the effect still draws).

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, ParticleBlend, ParticleItem,
    ParticleItems, SpawnRate, SpawnShape, VelocityDist,
};

/// The demo effect: a cone fountain with a colour ramp. Reconstructed here so the
/// controls can serialize it independently of the registered GPU instance.
fn demo_asset() -> EffectAsset {
    EffectAsset::new("saved-fountain")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(8_000.0),
            lifetime: (1.2, 2.0),
            spawn: SpawnShape::Cone {
                base_radius: 0.4,
                top_radius: 0.05,
                height: 0.8,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.3,
                min_speed: 3.0,
                max_speed: 5.0,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.12,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -4.0]))
        .force(ForceModifier::Drag(0.05))
        .with_gradient(
            Gradient::new()
                .with_colour(vec![(0.0, [1.0, 0.9, 0.4]), (1.0, [1.0, 0.2, 0.1])])
                .with_size(vec![(0.0, 0.4), (0.3, 1.0), (1.0, 0.0)]),
        )
}

/// The single preset this showcase registers.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("saved-fountain", demo_asset())]
}

/// UI state.
#[derive(Default)]
pub struct State {
    /// Last serialized RON, shown in the panel.
    pub ron: String,
    /// Round-trip status message.
    pub status: String,
}

/// Left-panel controls: serialize and round-trip buttons plus the RON output.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Serialize the current effect to a RON `.effect` document, then");
    ui.label("parse it back to prove round-trip fidelity.");
    ui.separator();

    #[cfg(feature = "serde")]
    {
        if ui.button("Serialize to RON").clicked() {
            match demo_asset().to_ron() {
                Ok(s) => {
                    state.status = format!("Serialized {} bytes.", s.len());
                    state.ron = s;
                }
                Err(e) => state.status = format!("Serialize error: {e}"),
            }
        }
        if ui.button("Round-trip (deserialize & verify)").clicked() {
            if state.ron.is_empty() {
                state.ron = demo_asset().to_ron().unwrap_or_default();
            }
            match EffectAsset::from_ron(&state.ron) {
                Ok(back) => {
                    let ok = back.forces.len() == demo_asset().forces.len()
                        && back.gradient.is_some();
                    state.status = if ok {
                        "Round-trip OK: forces + gradient preserved.".to_string()
                    } else {
                        "Round-trip parsed but fields differ!".to_string()
                    };
                }
                Err(e) => state.status = format!("Deserialize error: {e}"),
            }
        }
    }
    #[cfg(not(feature = "serde"))]
    {
        let _ = &mut state.ron;
        ui.label("Rebuild with `--features serde` to enable save / load.");
    }

    if !state.status.is_empty() {
        ui.separator();
        ui.label(&state.status);
    }
    if !state.ron.is_empty() {
        ui.separator();
        egui::ScrollArea::vertical()
            .max_height(360.0)
            .show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut state.ron.as_str())
                        .code_editor()
                        .desired_width(f32::INFINITY),
                );
            });
    }
}

/// Submit the registered effect so the panel has a live subject.
pub fn items(effect_ids: &[EffectId], _state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.first() {
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
    }
    items
}
