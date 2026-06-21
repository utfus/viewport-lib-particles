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
    "Alpha (sorted)",
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
    // Alpha (over) blend is order-dependent, so this route is depth-sorted
    // back-to-front each frame (toggle in the controls to see the artefact).
    let alpha = base()
        .with_blend(ParticleBlend::Alpha)
        .with_render(ParticleRender::Billboard { stretch: 0.0 });

    vec![
        plugin.add_effect(device, round),
        plugin.add_effect(device, stretched),
        plugin.add_effect(device, mesh_cubes),
        plugin.add_effect(device, trails),
        plugin.add_effect(device, alpha),
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
    /// Depth-sort non-additive effects back-to-front (only affects the alpha
    /// route; additive routes are order-independent).
    pub sort: bool,
    /// Render the environment map as a skybox background. The alpha route draws
    /// in the OIT pass (after the skybox), so it composites over it; toggle off
    /// to compare against the dark viewport background.
    pub skybox: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            selected: 0,
            paused: false,
            position: [0.0, 0.0, 0.0],
            sort: true,
            skybox: false,
        }
    }
}

/// A small equirectangular sky gradient (row-major RGBA f32) for the skybox
/// toggle: blue zenith, warm horizon, dark nadir. Values are linear; the
/// renderer tone-maps them.
pub fn skybox_pixels() -> (Vec<f32>, u32, u32) {
    const W: u32 = 32;
    const H: u32 = 16;
    let zenith = [0.10, 0.28, 0.75];
    let horizon = [0.95, 0.72, 0.48];
    let nadir = [0.06, 0.06, 0.09];
    let mut px = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        // 0 at the top row (zenith), 1 at the bottom (nadir); horizon at 0.5.
        let t = y as f32 / (H - 1) as f32;
        let rgb = if t < 0.5 {
            let k = t / 0.5;
            [
                zenith[0] + (horizon[0] - zenith[0]) * k,
                zenith[1] + (horizon[1] - zenith[1]) * k,
                zenith[2] + (horizon[2] - zenith[2]) * k,
            ]
        } else {
            let k = (t - 0.5) / 0.5;
            [
                horizon[0] + (nadir[0] - horizon[0]) * k,
                horizon[1] + (nadir[1] - horizon[1]) * k,
                horizon[2] + (nadir[2] - horizon[2]) * k,
            ]
        };
        for _ in 0..W {
            px.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 1.0]);
        }
    }
    (px, W, H)
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Route:");
    for (i, name) in NAMES.iter().enumerate() {
        ui.selectable_value(&mut state.selected, i, *name);
    }
    ui.separator();
    ui.checkbox(&mut state.paused, "Pause");
    ui.add_enabled(
        state.selected == 4,
        egui::Checkbox::new(&mut state.sort, "Sort back-to-front (alpha)"),
    );
    ui.checkbox(&mut state.skybox, "Skybox background");
    if state.skybox {
        ui.label("Alpha draws in the OIT pass, after the skybox, so it");
        ui.label("composites over it. Additive routes stay in the main pass.");
    }
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
    let mut items = ParticleItems::new()
        .with_dt(dt)
        .with_sort_transparent(state.sort);
    if let Some(&id) = effect_ids.get(state.selected) {
        let mut item = ParticleItem::new(id).at(state.position);
        item.settings.hidden = state.paused;
        items.push(item);
    }
    items
}
