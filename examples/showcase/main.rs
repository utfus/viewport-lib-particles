//! Feature showcase for `viewport-lib-particles` using `eframe` / `egui`.
//!
//! Mirrors the structure of `viewport-lib`'s `eframe_showcase`: a top bar
//! switches between showcase modes, a left panel holds per-mode controls, and
//! the central panel is the 3D viewport. The `ParticlePlugin` is registered on
//! the renderer at startup; each frame the active mode submits a `ParticleItems`
//! collection and the renderer runs the emit / simulate / draw passes.
//!
//! Run: `cargo run --release --example showcase`
//!
//! Navigation: left / middle drag orbit, right drag pan, scroll zoom.

mod showcase_01_emitters;
mod showcase_02_expression;
mod showcase_03_gradients;
mod showcase_04_render_routes;
mod showcase_05_refraction;
mod showcase_06_interaction;
mod showcase_07_textures;
mod showcase_08_turbulence;
mod showcase_09_subemitters;
mod showcase_10_simspace;
mod showcase_11_serialization;
mod showcase_12_instances;
mod showcase_13_bursts;
mod showcase_14_lifecycle;
mod showcase_15_fire;
mod showcase_16_rendering;
mod showcase_17_meshspawn;
mod showcase_18_soft;
mod showcase_19_fire_improved;
mod showcase_20_rotation;
mod showcase_21_timedbursts;
mod showcase_22_forces;
mod showcase_23_exprops;
mod showcase_24_iridescent;
mod viewport_callback;

use eframe::egui;
use viewport_lib::renderer::{EnvironmentLighting, PickBackend, PickMask};
use viewport_lib::{EnvironmentMapId, EnvironmentOptions, TextureData};
use viewport_lib::{
    BindingPreset, ButtonState, Camera, CameraFrame, FrameData, MeshId, OrbitCameraController,
    SceneFrame, ScrollUnits, ViewportContext, ViewportEvent, ViewportInput, ViewportRenderer,
    primitives,
};
use viewport_lib_particles::{EffectId, ParticlePlugin};

const BG_COLOUR: egui::Color32 = egui::Color32::from_rgb(18, 18, 22);

// ---------------------------------------------------------------------------
// Showcase mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum ShowcaseMode {
    Emitters,
    Expression,
    Gradients,
    RenderRoutes,
    Refraction,
    Interaction,
    Textures,
    Turbulence,
    SubEmitters,
    SimSpace,
    Serialization,
    Instances,
    Bursts,
    Lifecycle,
    Fire,
    Rendering,
    MeshSpawn,
    Soft,
    FireImproved,
    Rotation,
    TimedBursts,
    Forces,
    ExprOps,
    Iridescence,
}

impl ShowcaseMode {
    const ALL: &'static [ShowcaseMode] = &[
        ShowcaseMode::Emitters,
        ShowcaseMode::Expression,
        ShowcaseMode::Gradients,
        ShowcaseMode::RenderRoutes,
        ShowcaseMode::Refraction,
        ShowcaseMode::Interaction,
        ShowcaseMode::Textures,
        ShowcaseMode::Turbulence,
        ShowcaseMode::SubEmitters,
        ShowcaseMode::SimSpace,
        ShowcaseMode::Serialization,
        ShowcaseMode::Instances,
        ShowcaseMode::Bursts,
        ShowcaseMode::Lifecycle,
        ShowcaseMode::Fire,
        ShowcaseMode::Rendering,
        ShowcaseMode::MeshSpawn,
        ShowcaseMode::Soft,
        ShowcaseMode::FireImproved,
        ShowcaseMode::Rotation,
        ShowcaseMode::TimedBursts,
        ShowcaseMode::Forces,
        ShowcaseMode::ExprOps,
        ShowcaseMode::Iridescence,
    ];

    fn label(self) -> &'static str {
        match self {
            ShowcaseMode::Emitters => "1: Emitters",
            ShowcaseMode::Expression => "2: Expression",
            ShowcaseMode::Gradients => "3: Gradients",
            ShowcaseMode::RenderRoutes => "4: Render routes",
            ShowcaseMode::Refraction => "5: Refraction",
            ShowcaseMode::Interaction => "6: Interaction",
            ShowcaseMode::Textures => "7: Textures",
            ShowcaseMode::Turbulence => "8: Turbulence",
            ShowcaseMode::SubEmitters => "9: Sub-emitters",
            ShowcaseMode::SimSpace => "10: Sim space",
            ShowcaseMode::Serialization => "11: Load / save",
            ShowcaseMode::Instances => "12: Instances",
            ShowcaseMode::Bursts => "13: Bursts",
            ShowcaseMode::Lifecycle => "14: Lifecycle",
            ShowcaseMode::Fire => "15: Fire",
            ShowcaseMode::Rendering => "16: Rendering",
            ShowcaseMode::MeshSpawn => "17: Mesh spawn",
            ShowcaseMode::Soft => "18: Soft particles",
            ShowcaseMode::FireImproved => "19: Fire+",
            ShowcaseMode::Rotation => "20: Rotation",
            ShowcaseMode::TimedBursts => "21: Timed bursts",
            ShowcaseMode::Forces => "22: Forces",
            ShowcaseMode::ExprOps => "23: Expr ops",
            ShowcaseMode::Iridescence => "24: Iridescence",
        }
    }

    /// The next (`step = 1`) or previous (`step = -1`) mode, wrapping around.
    fn cycle(self, step: isize) -> ShowcaseMode {
        let n = Self::ALL.len() as isize;
        let cur = Self::ALL.iter().position(|&m| m == self).unwrap_or(0) as isize;
        Self::ALL[(cur + step).rem_euclid(n) as usize]
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> eframe::Result {
    eframe::run_native(
        "viewport-lib-particles : Showcase",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1180.0, 760.0]),
            depth_buffer: 24,
            stencil_buffer: 8,
            ..Default::default()
        },
        Box::new(|cc| {
            let rs = cc
                .wgpu_render_state
                .as_ref()
                .expect("wgpu backend required");
            let device = &rs.device;
            let queue = &rs.queue;
            let format = rs.target_format;

            let mut renderer = ViewportRenderer::new(device, format);

            // Environment map for the Render routes skybox toggle: a simple sky
            // gradient the alpha route can be shown compositing over via OIT.
            let (sky, sky_w, sky_h) = showcase_04_render_routes::skybox_pixels();
            let sky_env = renderer
                .upload_environment(
                    device,
                    queue,
                    TextureData::hdr(sky_w, sky_h, sky),
                    EnvironmentOptions::default(),
                )
                .expect("skybox env map");

            // Register the plugin and its preset effects. Effects must be added
            // before the plugin is handed to the renderer, so their ids are
            // captured here and kept aligned with the preset list by index.
            let mut plugin = ParticlePlugin::new();
            let emitter_ids: Vec<EffectId> = showcase_01_emitters::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let expression_ids: Vec<EffectId> = showcase_02_expression::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let gradient_ids: Vec<EffectId> = showcase_03_gradients::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let render_route_ids = showcase_04_render_routes::register(&mut plugin, device);
            let texture_ids = showcase_07_textures::register(&mut plugin, device, queue);
            let turbulence_ids: Vec<EffectId> = showcase_08_turbulence::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let subemitter_ids = showcase_09_subemitters::register(&mut plugin, device);
            let simspace_ids: Vec<EffectId> = showcase_10_simspace::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let serialization_ids: Vec<EffectId> = showcase_11_serialization::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let instances_ids: Vec<EffectId> = showcase_12_instances::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let bursts_ids: Vec<EffectId> = showcase_13_bursts::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let lifecycle_ids: Vec<EffectId> = showcase_14_lifecycle::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let fire_ids: Vec<EffectId> = showcase_15_fire::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let rendering_ids = showcase_16_rendering::register(&mut plugin, device, queue);
            let meshspawn_ids = showcase_17_meshspawn::register(&mut plugin, device);
            let soft_ids: Vec<EffectId> = showcase_18_soft::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let fire2_ids = showcase_19_fire_improved::register(&mut plugin, device, queue);
            let rotation_ids = showcase_20_rotation::register(&mut plugin, device, queue);
            let timedbursts_ids: Vec<EffectId> = showcase_21_timedbursts::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let forces_ids: Vec<EffectId> = showcase_22_forces::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let exprops_ids: Vec<EffectId> = showcase_23_exprops::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let iridescent_ids: Vec<EffectId> = showcase_24_iridescent::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .collect();
            let interaction_id = showcase_06_interaction::presets()
                .into_iter()
                .map(|(_, asset)| plugin.add_effect(device, asset))
                .next()
                .expect("one interaction effect");
            renderer.with_item_type_plugin(device, Box::new(plugin));

            // Ground plane for the interaction showcase (shadow receiver).
            let ground_mesh = renderer
                .resources_mut()
                .upload_mesh_data(device, &primitives::cuboid(20.0, 20.0, 0.5))
                .expect("ground mesh");
            // Crossed logs the improved fire spawns from and fades into (occluder).
            let fire2_logs_mesh = renderer
                .resources_mut()
                .upload_mesh_data(device, &showcase_19_fire_improved::logs_mesh_data())
                .expect("fire logs mesh");

            let mut writer = rs.renderer.write();
            writer.callback_resources.insert(renderer);
            // The refraction showcase drives a GpuPlugin directly; its GPU
            // resources live alongside the renderer in the callback map.
            writer
                .callback_resources
                .insert(showcase_05_refraction::RefractionResources::new(
                    device, format,
                ));
            drop(writer);

            Ok(Box::new(App {
                sky_env,
                camera: Camera {
                    distance: 10.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                },
                controller: OrbitCameraController::new_stateless(),
                input: ViewportInput::from_preset(BindingPreset::Viewer),
                mode: ShowcaseMode::Emitters,
                emitter_ids,
                expression_ids,
                gradient_ids,
                render_route_ids,
                texture_ids,
                turbulence_ids,
                emitters: showcase_01_emitters::State::default(),
                expression: showcase_02_expression::State::default(),
                gradients: showcase_03_gradients::State::default(),
                render_routes: showcase_04_render_routes::State::default(),
                refraction: showcase_05_refraction::State::default(),
                interaction_id,
                ground_mesh,
                interaction: showcase_06_interaction::State::default(),
                textures: showcase_07_textures::State::default(),
                turbulence: showcase_08_turbulence::State::default(),
                subemitter_ids,
                subemitters: showcase_09_subemitters::State::default(),
                simspace_ids,
                simspace: showcase_10_simspace::State::default(),
                serialization_ids,
                serialization: showcase_11_serialization::State::default(),
                instances_ids,
                instances: showcase_12_instances::State::default(),
                bursts_ids,
                bursts: showcase_13_bursts::State::default(),
                lifecycle_ids,
                lifecycle: showcase_14_lifecycle::State::default(),
                fire_ids,
                fire: showcase_15_fire::State::default(),
                rendering_ids,
                rendering: showcase_16_rendering::State::default(),
                meshspawn_ids,
                meshspawn: showcase_17_meshspawn::State::default(),
                soft_ids,
                soft: showcase_18_soft::State::default(),
                fire2_ids,
                fire2: showcase_19_fire_improved::State::default(),
                fire2_logs_mesh,
                rotation_ids,
                rotation: showcase_20_rotation::State::default(),
                timedbursts_ids,
                timedbursts: showcase_21_timedbursts::State::default(),
                forces_ids,
                forces: showcase_22_forces::State::default(),
                exprops_ids,
                exprops: showcase_23_exprops::State::default(),
                iridescent_ids,
                iridescent: showcase_24_iridescent::State::default(),
            }))
        }),
    )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    sky_env: EnvironmentMapId,
    camera: Camera,
    /// Turns the camera from a frame of resolved actions. The control scheme lives on
    /// `input`, so there is one answer to what a drag means.
    controller: OrbitCameraController,
    /// The frame's events, and the scheme that reads them.
    input: ViewportInput,
    mode: ShowcaseMode,
    /// Registered effects, aligned with each showcase's `presets()`.
    emitter_ids: Vec<EffectId>,
    expression_ids: Vec<EffectId>,
    gradient_ids: Vec<EffectId>,
    render_route_ids: Vec<EffectId>,
    texture_ids: Vec<EffectId>,
    turbulence_ids: Vec<EffectId>,
    emitters: showcase_01_emitters::State,
    expression: showcase_02_expression::State,
    gradients: showcase_03_gradients::State,
    render_routes: showcase_04_render_routes::State,
    refraction: showcase_05_refraction::State,
    interaction_id: EffectId,
    ground_mesh: MeshId,
    interaction: showcase_06_interaction::State,
    textures: showcase_07_textures::State,
    turbulence: showcase_08_turbulence::State,
    subemitter_ids: showcase_09_subemitters::Ids,
    subemitters: showcase_09_subemitters::State,
    simspace_ids: Vec<EffectId>,
    simspace: showcase_10_simspace::State,
    serialization_ids: Vec<EffectId>,
    serialization: showcase_11_serialization::State,
    instances_ids: Vec<EffectId>,
    instances: showcase_12_instances::State,
    bursts_ids: Vec<EffectId>,
    bursts: showcase_13_bursts::State,
    lifecycle_ids: Vec<EffectId>,
    lifecycle: showcase_14_lifecycle::State,
    fire_ids: Vec<EffectId>,
    fire: showcase_15_fire::State,
    rendering_ids: showcase_16_rendering::Ids,
    rendering: showcase_16_rendering::State,
    meshspawn_ids: showcase_17_meshspawn::Ids,
    meshspawn: showcase_17_meshspawn::State,
    soft_ids: Vec<EffectId>,
    soft: showcase_18_soft::State,
    fire2_ids: showcase_19_fire_improved::Ids,
    fire2: showcase_19_fire_improved::State,
    fire2_logs_mesh: MeshId,
    rotation_ids: showcase_20_rotation::Ids,
    rotation: showcase_20_rotation::State,
    timedbursts_ids: Vec<EffectId>,
    timedbursts: showcase_21_timedbursts::State,
    forces_ids: Vec<EffectId>,
    forces: showcase_22_forces::State,
    exprops_ids: Vec<EffectId>,
    exprops: showcase_23_exprops::State,
    iridescent_ids: Vec<EffectId>,
    iridescent: showcase_24_iridescent::State,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, eframe_frame: &mut eframe::Frame) {
        // Particles animate every frame.
        ctx.request_repaint();

        // A click in the Interaction mode requests a GPU pick, resolved after the
        // panels close (where the wgpu render state is reachable).
        let mut pick_request: Option<(glam::Vec2, [f32; 2], f32)> = None;

        // Cmd/Ctrl + [ / ] cycles through the showcase tabs.
        let step = ctx.input(|i| {
            if !i.modifiers.command {
                0
            } else if i.key_pressed(egui::Key::CloseBracket) {
                1
            } else if i.key_pressed(egui::Key::OpenBracket) {
                -1
            } else {
                0
            }
        });
        if step != 0 {
            self.mode = self.mode.cycle(step);
        }

        // ---- Top bar: mode switch ----
        egui::TopBottomPanel::top("mode_panel").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Showcase:");
                for mode in ShowcaseMode::ALL.iter().copied() {
                    if ui
                        .selectable_label(self.mode == mode, mode.label())
                        .clicked()
                    {
                        self.mode = mode;
                    }
                }
            });
        });

        // ---- Left panel: per-mode controls ----
        egui::SidePanel::left("controls_panel")
            .default_width(240.0)
            .show(ctx, |ui| match self.mode {
                ShowcaseMode::Emitters => showcase_01_emitters::controls(&mut self.emitters, ui),
                ShowcaseMode::Expression => {
                    showcase_02_expression::controls(&mut self.expression, ui)
                }
                ShowcaseMode::Gradients => showcase_03_gradients::controls(&mut self.gradients, ui),
                ShowcaseMode::RenderRoutes => {
                    showcase_04_render_routes::controls(&mut self.render_routes, ui)
                }
                ShowcaseMode::Refraction => {
                    showcase_05_refraction::controls(&mut self.refraction, ui)
                }
                ShowcaseMode::Interaction => {
                    showcase_06_interaction::controls(&mut self.interaction, ui)
                }
                ShowcaseMode::Textures => showcase_07_textures::controls(&mut self.textures, ui),
                ShowcaseMode::Turbulence => {
                    showcase_08_turbulence::controls(&mut self.turbulence, ui)
                }
                ShowcaseMode::SubEmitters => {
                    showcase_09_subemitters::controls(&mut self.subemitters, ui)
                }
                ShowcaseMode::SimSpace => {
                    showcase_10_simspace::controls(&mut self.simspace, ui)
                }
                ShowcaseMode::Serialization => {
                    showcase_11_serialization::controls(&mut self.serialization, ui)
                }
                ShowcaseMode::Instances => {
                    showcase_12_instances::controls(&mut self.instances, ui)
                }
                ShowcaseMode::Bursts => {
                    showcase_13_bursts::controls(&mut self.bursts, ui)
                }
                ShowcaseMode::Lifecycle => {
                    showcase_14_lifecycle::controls(&mut self.lifecycle, ui)
                }
                ShowcaseMode::Fire => showcase_15_fire::controls(&mut self.fire, ui),
                ShowcaseMode::Rendering => {
                    showcase_16_rendering::controls(&mut self.rendering, ui)
                }
                ShowcaseMode::MeshSpawn => {
                    showcase_17_meshspawn::controls(&mut self.meshspawn, ui)
                }
                ShowcaseMode::Soft => showcase_18_soft::controls(&mut self.soft, ui),
                ShowcaseMode::FireImproved => {
                    showcase_19_fire_improved::controls(&mut self.fire2, ui)
                }
                ShowcaseMode::Rotation => {
                    showcase_20_rotation::controls(&mut self.rotation, ui)
                }
                ShowcaseMode::TimedBursts => {
                    showcase_21_timedbursts::controls(&mut self.timedbursts, ui)
                }
                ShowcaseMode::Forces => showcase_22_forces::controls(&mut self.forces, ui),
                ShowcaseMode::ExprOps => showcase_23_exprops::controls(&mut self.exprops, ui),
                ShowcaseMode::Iridescence => showcase_24_iridescent::controls(&mut self.iridescent, ui),
            });

        // ---- Central panel: viewport ----
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG_COLOUR))
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
                let (w, h) = (rect.width(), rect.height());

                // The size stays on the controller, which needs it to turn a pan in pixels
                // into one in world units; the events go to the input, which owns the
                // scheme that reads them.
                self.controller.set_viewport_size([w, h]);
                self.input.begin_frame(ViewportContext {
                    hovered: response.hovered(),
                    focused: response.has_focus(),
                    viewport_size: [w, h],
                });
                self.feed_input(ui, rect);

                let actions = self.input.resolve();
                self.controller
                    .apply(&mut self.camera, &actions, ctx.input(|i| i.stable_dt));
                self.camera.set_aspect_ratio(w, h);

                let dt = ctx.input(|i| i.stable_dt).min(1.0 / 30.0);

                // Refraction runs a GpuPlugin post-pass rather than submitting
                // particle items, so it uses a dedicated callback.
                if self.mode == ShowcaseMode::Refraction {
                    self.refraction.time += dt;
                    ui.painter()
                        .add(eframe::egui_wgpu::Callback::new_paint_callback(
                            rect,
                            showcase_05_refraction::RefractionCallback {
                                center: [0.5, 0.5],
                                radius: self.refraction.radius(),
                                width: self.refraction.width,
                                strength: self.refraction.strength,
                            },
                        ));
                    if response.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Default);
                    }
                    return;
                }

                let ppp = ui.ctx().pixels_per_point();

                // Interaction submits a ground plane + directional light alongside
                // the particles, and a click requests a GPU pick.
                if self.mode == ShowcaseMode::Interaction {
                    if response.clicked() {
                        if let Some(p) = response.interact_pointer_pos() {
                            let cursor = glam::Vec2::new(p.x - rect.left(), p.y - rect.top());
                            pick_request = Some((cursor, [w, h], ppp));
                        }
                    }
                    let frame_data = self.interaction_frame([w, h], ppp, dt);
                    ui.painter()
                        .add(eframe::egui_wgpu::Callback::new_paint_callback(
                            rect,
                            viewport_callback::ViewportCallback { frame: frame_data },
                        ));
                    if response.dragged() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                    } else if response.hovered() {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    }
                    return;
                }

                // Soft particles need opaque geometry to fade against: submit the
                // shared ground plane (and a sun light) for those modes; the
                // improved fire adds its crossed logs.
                let surfaces = match self.mode {
                    ShowcaseMode::Soft => {
                        vec![showcase_06_interaction::ground_item(self.ground_mesh)]
                    }
                    ShowcaseMode::FireImproved => vec![
                        showcase_19_fire_improved::ground_item(self.ground_mesh),
                        showcase_19_fire_improved::logs_item(self.fire2_logs_mesh),
                    ],
                    _ => Vec::new(),
                };
                let mut scene = SceneFrame::from_surface_items(surfaces);
                let items = match self.mode {
                    ShowcaseMode::Emitters => {
                        showcase_01_emitters::items(&self.emitter_ids, &self.emitters, dt)
                    }
                    ShowcaseMode::Expression => {
                        showcase_02_expression::items(&self.expression_ids, &self.expression, dt)
                    }
                    ShowcaseMode::Gradients => {
                        showcase_03_gradients::items(&self.gradient_ids, &self.gradients, dt)
                    }
                    ShowcaseMode::RenderRoutes => showcase_04_render_routes::items(
                        &self.render_route_ids,
                        &self.render_routes,
                        dt,
                    ),
                    ShowcaseMode::Textures => {
                        showcase_07_textures::items(&self.texture_ids, &self.textures, dt)
                    }
                    ShowcaseMode::Turbulence => {
                        showcase_08_turbulence::items(&self.turbulence_ids, &self.turbulence, dt)
                    }
                    ShowcaseMode::SubEmitters => {
                        showcase_09_subemitters::items(&self.subemitter_ids, &self.subemitters, dt)
                    }
                    ShowcaseMode::SimSpace => {
                        self.simspace.t += dt;
                        showcase_10_simspace::items(&self.simspace_ids, &self.simspace, dt)
                    }
                    ShowcaseMode::Serialization => showcase_11_serialization::items(
                        &self.serialization_ids,
                        &self.serialization,
                        dt,
                    ),
                    ShowcaseMode::Instances => {
                        self.instances.t += dt;
                        showcase_12_instances::items(&self.instances_ids, &self.instances, dt)
                    }
                    ShowcaseMode::Bursts => {
                        self.bursts.t += dt;
                        showcase_13_bursts::items(&self.bursts_ids, &self.bursts, dt)
                    }
                    ShowcaseMode::Lifecycle => {
                        showcase_14_lifecycle::items(&self.lifecycle_ids, &self.lifecycle, dt)
                    }
                    ShowcaseMode::Fire => showcase_15_fire::items(&self.fire_ids, &self.fire, dt),
                    ShowcaseMode::Rendering => {
                        showcase_16_rendering::items(&self.rendering_ids, &self.rendering, dt)
                    }
                    ShowcaseMode::MeshSpawn => {
                        showcase_17_meshspawn::items(&self.meshspawn_ids, &self.meshspawn, dt)
                    }
                    ShowcaseMode::Soft => {
                        showcase_18_soft::items(&self.soft_ids, &self.soft, dt)
                    }
                    ShowcaseMode::FireImproved => {
                        showcase_19_fire_improved::items(&self.fire2_ids, &self.fire2, dt)
                    }
                    ShowcaseMode::Rotation => {
                        showcase_20_rotation::items(&self.rotation_ids, &self.rotation, dt)
                    }
                    ShowcaseMode::TimedBursts => {
                        self.timedbursts.t += dt;
                        showcase_21_timedbursts::items(
                            &self.timedbursts_ids,
                            &self.timedbursts,
                            dt,
                        )
                    }
                    ShowcaseMode::Forces => {
                        showcase_22_forces::items(&self.forces_ids, &self.forces, dt)
                    }
                    ShowcaseMode::ExprOps => {
                        showcase_23_exprops::items(&self.exprops_ids, &self.exprops, dt)
                    }
                    ShowcaseMode::Iridescence => {
                        showcase_24_iridescent::items(&self.iridescent_ids, &self.iridescent, dt)
                    }
                    ShowcaseMode::Refraction | ShowcaseMode::Interaction => {
                        unreachable!("handled above")
                    }
                };
                scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);

                let mut frame_data = FrameData::new(
                    CameraFrame::from_camera(&self.camera, [w, h])
                        .with_pixels_per_point(ui.ctx().pixels_per_point()),
                    scene,
                );

                // Render routes can toggle a skybox to show the alpha route
                // compositing over it through the OIT pass.
                if self.mode == ShowcaseMode::RenderRoutes && self.render_routes.skybox {
                    frame_data.effects.environment = Some(EnvironmentLighting::new(self.sky_env));
                }

                // Soft particles / improved fire: light the geometry they fade into.
                if matches!(self.mode, ShowcaseMode::Soft | ShowcaseMode::FireImproved) {
                    frame_data.effects.lighting = showcase_06_interaction::lighting();
                }

                ui.painter()
                    .add(eframe::egui_wgpu::Callback::new_paint_callback(
                        rect,
                        viewport_callback::ViewportCallback { frame: frame_data },
                    ));

                if response.dragged() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else if response.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                }
            });

        // Resolve a pending interaction pick now that the panels have closed and
        // the wgpu render state (device / queue / renderer) is reachable.
        if let Some((cursor, size, ppp)) = pick_request {
            if let Some(rs) = eframe_frame.wgpu_render_state() {
                let frame_data = self.interaction_frame(size, ppp, 0.0);
                let mut writer = rs.renderer.write();
                if let Some(renderer) = writer.callback_resources.get_mut::<ViewportRenderer>() {
                    let hit = renderer.pick_object(
                        PickBackend::Gpu,
                        cursor,
                        &frame_data,
                        &rs.device,
                        &rs.queue,
                        PickMask::OBJECT,
                    );
                    self.interaction.picked =
                        matches!(hit, Some(h) if h.id == showcase_06_interaction::PICK_ID);
                }
            }
        }
    }
}

impl App {
    /// Build the Interaction mode's frame: the ground plane, the directional
    /// light, and the particle system tagged with its pick id / selection.
    fn interaction_frame(&self, size: [f32; 2], ppp: f32, dt: f32) -> FrameData {
        let ground = showcase_06_interaction::ground_item(self.ground_mesh);
        let mut scene = SceneFrame::from_surface_items(vec![ground]);
        let items = showcase_06_interaction::items(self.interaction_id, &self.interaction, dt);
        scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let mut frame_data = FrameData::new(
            CameraFrame::from_camera(&self.camera, size).with_pixels_per_point(ppp),
            scene,
        );
        frame_data.effects.lighting = showcase_06_interaction::lighting();
        frame_data.interaction.outline_selected = true;
        frame_data
    }

    /// Translate egui pointer / wheel events into `ViewportEvent`s for the orbit
    /// controller.
    fn feed_input(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        ui.input(|i| {
            self.input
                .push_event(ViewportEvent::ModifiersChanged(viewport_lib::Modifiers {
                    alt: i.modifiers.alt,
                    shift: i.modifiers.shift,
                    ctrl: i.modifiers.command,
                }));

            if let Some(p) = i.pointer.interact_pos() {
                let local = glam::Vec2::new(p.x - rect.left(), p.y - rect.top());
                self.input
                    .push_event(ViewportEvent::PointerMoved { position: local });
            }

            for event in &i.events {
                match event {
                    egui::Event::PointerButton {
                        button, pressed, ..
                    } => {
                        let vp_button = match button {
                            egui::PointerButton::Primary => viewport_lib::MouseButton::Left,
                            egui::PointerButton::Secondary => viewport_lib::MouseButton::Right,
                            egui::PointerButton::Middle => viewport_lib::MouseButton::Middle,
                            _ => continue,
                        };
                        self.input.push_event(ViewportEvent::MouseButton {
                            button: vp_button,
                            state: if *pressed {
                                ButtonState::Pressed
                            } else {
                                ButtonState::Released
                            },
                        });
                    }
                    egui::Event::MouseWheel { unit, delta, .. } => {
                        let units = match unit {
                            egui::MouseWheelUnit::Line => ScrollUnits::Lines,
                            egui::MouseWheelUnit::Point => ScrollUnits::Pixels,
                            egui::MouseWheelUnit::Page => ScrollUnits::Pages,
                        };
                        self.input.push_event(ViewportEvent::Wheel {
                            delta: glam::Vec2::new(delta.x, delta.y),
                            units,
                        });
                    }
                    _ => {}
                }
            }
        });
    }
}
