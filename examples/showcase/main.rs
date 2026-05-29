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
mod viewport_callback;

use eframe::egui;
use viewport_lib::{
    ButtonState, Camera, CameraFrame, FrameData, OrbitCameraController, SceneFrame, ScrollUnits,
    ViewportContext, ViewportEvent, ViewportRenderer,
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
}

impl ShowcaseMode {
    const ALL: &'static [ShowcaseMode] = &[
        ShowcaseMode::Emitters,
        ShowcaseMode::Expression,
        ShowcaseMode::Gradients,
        ShowcaseMode::RenderRoutes,
    ];

    fn label(self) -> &'static str {
        match self {
            ShowcaseMode::Emitters => "1: Emitters",
            ShowcaseMode::Expression => "2: Expression",
            ShowcaseMode::Gradients => "3: Gradients",
            ShowcaseMode::RenderRoutes => "4: Render routes",
        }
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
            let format = rs.target_format;

            let mut renderer = ViewportRenderer::new(device, format);

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
            renderer.with_item_type_plugin(device, Box::new(plugin));

            rs.renderer.write().callback_resources.insert(renderer);

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
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Particles animate every frame.
        ctx.request_repaint();

        // ---- Top bar: mode switch ----
        egui::TopBottomPanel::top("mode_panel").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label("Showcase:");
                for mode in ShowcaseMode::ALL.iter().copied() {
                    if ui.selectable_label(self.mode == mode, mode.label()).clicked() {
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
                ShowcaseMode::Gradients => {
                    showcase_03_gradients::controls(&mut self.gradients, ui)
                }
                ShowcaseMode::RenderRoutes => {
                    showcase_04_render_routes::controls(&mut self.render_routes, ui)
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
                };
                scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);

                let frame_data = FrameData::new(
                    CameraFrame::from_camera(&self.camera, [w, h])
                        .with_pixels_per_point(ui.ctx().pixels_per_point()),
                    scene,
                );

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
    }
}

impl App {
    /// Translate egui pointer / wheel events into `ViewportEvent`s for the orbit
    /// controller.
    fn feed_input(&mut self, ui: &egui::Ui, rect: egui::Rect) {
        ui.input(|i| {
            self.controller.push_event(ViewportEvent::ModifiersChanged(
                viewport_lib::Modifiers {
                    alt: i.modifiers.alt,
                    shift: i.modifiers.shift,
                    ctrl: i.modifiers.command,
                },
            ));

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
