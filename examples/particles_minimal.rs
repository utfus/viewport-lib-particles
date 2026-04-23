//! Minimal authoring example.
//!
//! Builds a fountain effect and shows how it would be registered and submitted.
//! Registering the plugin needs a `wgpu::Device`, so this headless example only
//! exercises the authoring API; `tests/simulates.rs` drives a real frame.
//!
//! Run: `cargo run --example particles-minimal`

use viewport_lib_particles::{
    EffectAsset, Emitter, ForceModifier, ParticleBlend, ParticlePlugin, SpawnRate, SpawnShape,
    VelocityDist,
};

fn main() {
    // A cone fountain that falls under gravity, drawn additively.
    let effect = EffectAsset::new("fountain")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(5_000.0),
            lifetime: (1.5, 3.0),
            spawn: SpawnShape::Sphere {
                radius: 0.2,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.35,
                min_speed: 2.0,
                max_speed: 4.5,
            },
            colour: [1.0, 0.55, 0.15, 1.0],
            size: 0.15,
        })
        // Z-up: gravity pulls along -Z.
        .force(ForceModifier::Accel([0.0, 0.0, -2.5]))
        .force(ForceModifier::Drag(0.1));

    let plugin = ParticlePlugin::new();
    println!("plugin type name: {}", ParticlePlugin::TYPE_NAME);
    println!(
        "effect '{}': capacity {}, {} forces, blend {:?}",
        effect.name,
        effect.capacity,
        effect.forces.len(),
        effect.blend,
    );
    println!("registered effects: {}", plugin.effect_count());

    // In a real app, with a `wgpu::Device` in hand:
    //   let id = plugin.add_effect(&device, effect);
    //   renderer.with_item_type_plugin(&device, Box::new(plugin));
    // and each frame:
    //   let mut items = ParticleItems::new().with_dt(dt);
    //   items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
    //   frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);

    println!("particles-minimal: authoring API wired; see tests/simulates.rs for a live frame");
}
