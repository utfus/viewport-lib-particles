//! Showcase 4: Render routes.
//!
//! The same simulation drawn three ways: round billboards, velocity-stretched
//! billboards (motion streaks), and mesh instances (tumbling cubes). The mesh
//! route needs a mesh uploaded to the plugin first, so this module registers its
//! effects through `&mut ParticlePlugin` instead of returning static presets.

use eframe::egui;
use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ForceModifier, Gradient, MeshAlign, ParticleBlend,
    ParticleItem, ParticleItems, ParticlePlugin, ParticleRender, SpawnRate, SpawnShape,
    VelocityDist,
};

/// Names of the registered effects, in menu order.
pub const NAMES: &[&str] = &[
    "Round billboards",
    "Stretched sparks",
    "Mesh cubes",
    "Comet trails",
];

/// A cone fountain shared by the three routes, minus the render mode.
fn base() -> EffectAsset {
    EffectAsset::new("route")
        .with_capacity(30_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(7_000.0),
            lifetime: (1.2, 2.2),
            spawn: SpawnShape::Sphere {
                radius: 0.15,
                volume: true,
            },
            velocity: VelocityDist::UniformCone {
                axis: [0.0, 0.0, 1.0],
                half_angle: 0.5,
                min_speed: 3.0,
                max_speed: 5.5,
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.16,
        })
        .force(ForceModifier::Accel([0.0, 0.0, -5.0]))
        // Cool electric palette, distinct from the warm fire look the Gradients
        // showcase uses: this tab is about the draw route, not the colour.
        .with_gradient(
            Gradient::new()
                .with_colour(vec![
                    (0.0, [0.6, 1.5, 1.9]),
                    (0.5, [0.15, 0.5, 1.4]),
                    (1.0, [0.03, 0.08, 0.35]),
                ])
                .with_size(vec![(0.0, 1.0), (1.0, 0.4)]),
        )
}

/// Register the three route effects, uploading the cube mesh the mesh route
/// needs. Returns the effect ids aligned with [`NAMES`].
pub fn register(plugin: &mut ParticlePlugin, device: &eframe::wgpu::Device) -> Vec<EffectId> {
    let (positions, indices) = cube();
    let mesh = plugin.upload_mesh(device, &positions, &indices);

    let round = base().with_render(ParticleRender::Billboard { stretch: 0.0 });
    let stretched = base().with_render(ParticleRender::Billboard { stretch: 0.15 });
    let mesh_cubes = base()
        // Cubes read better a touch larger and opaque-ish over-blended.
        .with_render(ParticleRender::Mesh {
            mesh,
            align: MeshAlign::Random,
        });
    // A ribbon swept through each particle's actual path: the one route that
    // follows the simulated arc, not just the instantaneous velocity.
    let trails = base().with_render(ParticleRender::Trail {
        width: 0.06,
        segments: 20,
    });

    vec![
        plugin.add_effect(device, round),
        plugin.add_effect(device, stretched),
        plugin.add_effect(device, mesh_cubes),
        plugin.add_effect(device, trails),
    ]
}

/// Unit cube centered at the origin.
fn cube() -> (Vec<[f32; 3]>, Vec<u32>) {
    let p = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let i = vec![
        0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4,
        1, 5, 6, 1, 6, 2,
    ];
    (p, i)
}

/// UI state.
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
    ui.label("Route:");
    for (i, name) in NAMES.iter().enumerate() {
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
    ui.label("One simulation, four draws: round billboards, velocity-");
    ui.label("stretched streaks, instanced tumbling cubes, and comet");
    ui.label("trails swept through each particle's recorded path.");
}

/// Build this frame's submission for the selected route.
pub fn items(effect_ids: &[EffectId], state: &State, dt: f32) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(dt);
    if let Some(&id) = effect_ids.get(state.selected) {
        let mut item = ParticleItem::new(id).at(state.position);
        item.settings.hidden = state.paused;
        items.push(item);
    }
    items
}
