//! Showcase 13: Transient bursts.
//!
//! Fire-and-forget bursts (muzzle-flash / impact style) are auto-fired at
//! scattered points on a timer. Each is submitted once via `ParticleItem::burst`
//! and the plugin plays it in a pooled transient slot, then self-retires it when
//! its particles die: no per-burst registration, no growing state. The pool cap
//! (`with_max_transients`) bounds concurrency; over it, the oldest recycles.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, Gradient, ParticleBlend, ParticleItem, ParticleItems,
    SpawnRate, SpawnShape, VelocityDist,
};

/// A short, bright one-shot spray: a burst count with a brief lifetime.
fn spark_burst() -> EffectAsset {
    EffectAsset::new("spark-burst")
        .with_capacity(1_500)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::Burst { count: 500 },
            lifetime: (0.4, 0.7),
            spawn: SpawnShape::Sphere {
                radius: 0.05,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: std::f32::consts::PI, // full sphere
                min_speed: 1.5,
                max_speed: 3.5,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.06,
        })
        .with_gradient(
            Gradient::new()
                .with_colour(vec![(0.0, [1.0, 0.9, 0.5]), (1.0, [1.0, 0.3, 0.1])])
                .with_size(vec![(0.0, 1.0), (1.0, 0.0)]),
        )
}

/// The single burst effect this showcase registers.
pub fn presets() -> Vec<(&'static str, EffectAsset)> {
    vec![("spark-burst", spark_burst())]
}

/// UI state.
pub struct State {
    /// Bursts fired per second.
    pub rate: f32,
    /// Concurrent transient-slot cap.
    pub max_transients: usize,
    /// Scatter radius for burst positions.
    pub spread: f32,
    /// Internal clock, advanced by the app each frame.
    pub t: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            rate: 8.0,
            max_transients: 32,
            spread: 2.5,
            t: 0.0,
        }
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.add(egui::Slider::new(&mut state.rate, 0.0..=40.0).text("bursts / sec"));
    ui.add(egui::Slider::new(&mut state.max_transients, 1..=64).text("pool size"));
    ui.add(egui::Slider::new(&mut state.spread, 0.0..=5.0).text("spread"));
    ui.separator();
    ui.label("Each burst is submitted once and self-retires.");
    ui.label("Raise the rate past the pool size to see recycling.");
}

/// A scattered position derived from an integer burst index (no RNG state).
fn burst_pos(n: u32, spread: f32) -> [f32; 3] {
    let h = |k: u32| {
        let mut v = n.wrapping_mul(2654435761).wrapping_add(k.wrapping_mul(40503));
        v ^= v >> 15;
        v = v.wrapping_mul(0x2c1b3c6d);
        v ^= v >> 12;
        (v as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    [h(1) * spread, h(2) * spread, 0.2 + h(3).abs() * spread * 0.5]
}

/// Fire a burst each time the timer crosses a `1/rate` boundary this frame.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new()
        .with_dt(dt)
        .with_max_transients(state.max_transients);
    let Some(&id) = effect_ids.first() else {
        return items;
    };
    if state.rate > 0.0 {
        let interval = 1.0 / state.rate;
        let prev = ((state.t - dt).max(0.0) / interval).floor();
        let now = (state.t / interval).floor();
        // Fire one burst per boundary crossed this frame (clamped so a huge dt
        // can't spawn an unbounded number).
        let count = ((now - prev) as i32).clamp(0, 8);
        for k in 0..count {
            let n = (now as u32).wrapping_sub(k as u32);
            items.push(ParticleItem::new(id).at(burst_pos(n, state.spread)).burst());
        }
    }
    items
}
