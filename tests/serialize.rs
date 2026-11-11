//! Effect serialization round-trip (RON, behind the `serde` feature).
//!
//! Pure-CPU tests: they exercise the serde derives on the whole asset model
//! (emitter, forces, expression `Module`, gradient, render route, simulation
//! space) with no GPU, so they run anywhere. Gated on the `serde` feature.
#![cfg(feature = "serde")]

use viewport_lib_particles::{
    Attribute, EffectAsset, EffectProgram, Emitter, ForceModifier, Gradient, Module,
    ModifierRegistry, ParticleBlend, ParticleRender, PropertyValue, SimulationSpace, SpawnShape,
    UpdateOp,
};

#[test]
fn round_trip_fixed_function() {
    let asset = EffectAsset::new("fixed")
        .with_capacity(4321)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(Emitter {
            spawn: SpawnShape::Cone {
                base_radius: 1.0,
                top_radius: 0.25,
                height: 3.0,
                volume: true,
            },
            ..Default::default()
        })
        .force(ForceModifier::Accel([0.0, 0.0, -9.81]))
        .force(ForceModifier::CurlNoise {
            scale: 0.4,
            strength: 2.0,
            speed: 0.5,
        })
        .with_simulation_space(SimulationSpace::Local)
        .with_render(ParticleRender::Trail {
            width: 0.1,
            segments: 12,
        })
        .with_gradient(
            Gradient::new().with_colour(vec![(0.0, [1.0, 0.0, 0.0]), (1.0, [0.0, 0.0, 1.0])]),
        );

    let ron = asset.to_ron().expect("serialize");
    let back = EffectAsset::from_ron(&ron).expect("deserialize");

    assert_eq!(back.capacity, 4321);
    assert_eq!(back.blend, ParticleBlend::Alpha);
    assert_eq!(back.sim_space, SimulationSpace::Local);
    assert_eq!(back.modifiers.len(), 2);
    assert!(matches!(
        back.emitter.spawn,
        SpawnShape::Cone {
            volume: true,
            height,
            ..
        } if (height - 3.0).abs() < 1e-6
    ));
    assert!(matches!(
        back.render,
        ParticleRender::Trail { segments: 12, .. }
    ));
    let g = back.gradient.expect("gradient");
    assert_eq!(g.colour.len(), 2);
}

#[test]
fn round_trip_program_with_property() {
    // A codegen program exercises the `String` attribute / property names and the
    // expression `Module` arena through serde.
    let mut m = Module::new();
    let pos = m.attr("position");
    let spring = m.property("spring");
    let accel = m.mul(pos, spring);
    let node_count = m.len();

    let mut program = EffectProgram::new().property("spring", PropertyValue::F32(2.0));
    program.module = m;
    program = program.update(UpdateOp::Accelerate(accel));
    let _ = Attribute::Velocity; // (kept in scope for the fixed-path parity above)

    let asset = EffectAsset::new("program").with_program(program);
    let ron = asset.to_ron().expect("serialize");
    let back = EffectAsset::from_ron(&ron).expect("deserialize");

    let bp = back.program.expect("program survived round-trip");
    assert_eq!(bp.module.len(), node_count);
    assert_eq!(bp.properties.len(), 1);
    assert_eq!(bp.properties[0].name, "spring");
    assert_eq!(bp.update.len(), 1);
}

#[test]
fn file_round_trip() {
    let asset = EffectAsset::new("disc").with_emitter(Emitter {
        spawn: SpawnShape::Circle {
            radius: 2.0,
            volume: false,
        },
        ..Default::default()
    });

    let path = std::env::temp_dir().join("vlp_serialize_test.effect");
    asset.save_ron(&path).expect("save");
    let back = EffectAsset::load_ron(&path).expect("load");
    assert!(matches!(
        back.emitter.spawn,
        SpawnShape::Circle { volume: false, .. }
    ));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn modifier_registry_reconstructs_by_name() {
    let mut reg = ModifierRegistry::new();
    reg.register_force("gravity", || ForceModifier::Accel([0.0, 0.0, -9.81]));
    reg.register_effect("spark", || {
        EffectAsset::new("spark").with_blend(ParticleBlend::Additive)
    });

    assert!(matches!(
        reg.force("gravity"),
        Some(ForceModifier::Accel(_))
    ));
    assert!(reg.force("missing").is_none());
    let spark = reg.effect("spark").expect("named effect");
    assert_eq!(spark.blend, ParticleBlend::Additive);
    assert!(reg.force_names().any(|n| n == "gravity"));
}
