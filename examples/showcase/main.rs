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
mod viewport_callback;

use eframe::egui;
use viewport_lib::renderer::{EnvironmentMap, PickBackend, PickMask};
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, MeshId, OrbitCameraController, SceneFrame,
    ScrollUnits, ViewportContext, ViewportEvent, ViewportRenderer, primitives,
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
}

impl ShowcaseMode {
    const ALL: &'static [ShowcaseMode] = &[
        ShowcaseMode::Emitters,
        ShowcaseMode::Expression,
        ShowcaseMode::Gradients,
        ShowcaseMode::RenderRoutes,
        ShowcaseMode::Refraction,
        ShowcaseMode::Interaction,
    ];

    fn label(self) -> &'static str {
        match self {
            ShowcaseMode::Emitters => "1: Emitters",
            ShowcaseMode::Expression => "2: Expression",
            ShowcaseMode::Gradients => "3: Gradients",
            ShowcaseMode::RenderRoutes => "4: Render routes",
            ShowcaseMode::Refraction => "5: Refraction",
            ShowcaseMode::Interaction => "6: Interaction",
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
            renderer
                .upload_environment_map(device, queue, &sky, sky_w, sky_h)
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
                camera: Camera {
                    distance: 10.0,
                    orientation: glam::Quat::from_rotation_z(0.6)
                        * glam::Quat::from_rotation_x(1.1),
                    ..Camera::default()
                },
                controller: OrbitCameraController::viewport_primitives(),
                mode: ShowcaseMode::Emitters,
                emitter_ids,
                expression_ids,
                gradient_ids,
                render_route_ids,
                emitters: showcase_01_emitters::State::default(),
                expression: showcase_02_expression::State::default(),
                gradients: showcase_03_gradients::State::default(),
                render_routes: showcase_04_render_routes::State::default(),
                refraction: showcase_05_refraction::State::default(),
                interaction_id,
                ground_mesh,
                interaction: showcase_06_interaction::State::default(),
            }))
        }),
    )
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    camera: Camera,
    controller: OrbitCameraController,
    mode: ShowcaseMode,
    /// Registered effects, aligned with each showcase's `presets()`.
    emitter_ids: Vec<EffectId>,
    expression_ids: Vec<EffectId>,
    gradient_ids: Vec<EffectId>,
    render_route_ids: Vec<EffectId>,
    emitters: showcase_01_emitters::State,
    expression: showcase_02_expression::State,
    gradients: showcase_03_gradients::State,
    render_routes: showcase_04_render_routes::State,
    refraction: showcase_05_refraction::State,
    interaction_id: EffectId,
    ground_mesh: MeshId,
    interaction: showcase_06_interaction::State,
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
            });

        // ---- Central panel: viewport ----
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(BG_COLOUR))
            .show(ctx, |ui| {
                let (rect, response) =
                    ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
                let (w, h) = (rect.width(), rect.height());

                self.controller.begin_frame(ViewportContext {
                    hovered: response.hovered(),
                    focused: response.has_focus(),
                    viewport_size: [w, h],
                });
                self.feed_input(ui, rect);

                self.controller.apply_to_camera(&mut self.camera);
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

                let mut scene = SceneFrame::from_surface_items(Vec::new());
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
                    frame_data.effects.environment = Some(EnvironmentMap {
                        show_skybox: true,
                        ..Default::default()
                    });
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
            self.controller
                .push_event(ViewportEvent::ModifiersChanged(viewport_lib::Modifiers {
                    alt: i.modifiers.alt,
                    shift: i.modifiers.shift,
                    ctrl: i.modifiers.command,
                }));

            if let Some(p) = i.pointer.interact_pos() {
                let local = glam::Vec2::new(p.x - rect.left(), p.y - rect.top());
                self.controller
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
                        self.controller.push_event(ViewportEvent::MouseButton {
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
                        self.controller.push_event(ViewportEvent::Wheel {
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
