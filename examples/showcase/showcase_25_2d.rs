//! Showcase 25: 2D effects.
//!
//! A 2D game's effects on viewport-lib's XY plane, seen top-down through an
//! orthographic camera so the page reads as a 2D game draws: painter sprites
//! for the backdrop, platforms and a character, and effects locked to the
//! plane. One demo at a time:
//!
//! - Plane lock: turbulent sparks that never leave the plane, and a firework
//!   whose shell breaks into sparks that stay on it.
//! - Blends: smoke that darkens the ground (multiply) and glows that lighten it
//!   without blowing out (screen).
//! - Painter order: hit sparks in front of a character and dust behind it.
//! - Pixel art: a pixel-art burst sheet drawn nearest, snapped and at integer
//!   scale, beside its linear twin.
//! - Hooks: a hit flash and a dissolve, the sprite effects' maths on particles.
//! - Rain: rain that dies on the platforms it hits and splashes where it lands.

use eframe::{egui, wgpu};
use viewport_lib::{Camera, Colour};
use viewport_lib_particles::{
    Collide2d, Collider2d, Collider2dSet, CollisionResponse, EffectAsset, EffectId, Emitter,
    EventCondition, Flipbook, FragmentHook, Modifier, Orient, PainterPlacement, ParticleBlend,
    ParticleItem, ParticleItems, ParticlePlane, ParticlePlugin, PlaneLock, SpawnRate, SpawnShape,
    SubEmitter, TextureFilter, VelocityDist,
};
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteOrder, SpriteSizeMode};

/// World units across the view's height.
const VIEW_HEIGHT: f32 = 12.0;

/// The platforms of the rain demo, `(centre, size)`, as sprites and colliders.
const PLATFORMS: [([f32; 2], [f32; 2]); 3] = [
    ([-3.5, -1.0], [3.0, 0.4]),
    ([2.0, 1.0], [3.5, 0.4]),
    ([0.0, -4.0], [8.0, 0.4]),
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Demo {
    PlaneLock,
    Blends,
    Painter,
    PixelArt,
    Hooks,
    Rain,
}

pub struct State {
    pub demo: Demo,
    t: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            demo: Demo::PlaneLock,
            t: 0.0,
        }
    }
}

pub struct Ids {
    sparks: EffectId,
    shell: EffectId,
    embers: EffectId,
    smoke: EffectId,
    glow: EffectId,
    hit: EffectId,
    dust: EffectId,
    pixel: EffectId,
    linear: EffectId,
    flash: EffectId,
    dissolve: EffectId,
    rain: EffectId,
    splash: EffectId,
}

fn on_plane() -> PlaneLock {
    PlaneLock::new(ParticlePlane::Xy, 0.0)
}

fn emitter(rate: SpawnRate, lifetime: (f32, f32), colour: [f32; 4], size: f32) -> Emitter {
    Emitter {
        rate,
        lifetime,
        spawn: SpawnShape::Point,
        velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
        colour,
        size,
    }
}

/// A 32x8 sheet of four 8x8 cells: a ring that grows and fades across them.
fn burst_sheet() -> (u32, u32, Vec<u8>) {
    let (w, h) = (32u32, 8u32);
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for cell in 0..4u32 {
        let r = 1.0 + cell as f32 * 0.9;
        for y in 0..8u32 {
            for x in 0..8u32 {
                let d = ((x as f32 - 3.5).powi(2) + (y as f32 - 3.5).powi(2)).sqrt();
                if (d - r).abs() < 0.8 {
                    let i = ((y * w + cell * 8 + x) * 4) as usize;
                    let fade = 255 - cell as u8 * 50;
                    rgba[i..i + 4].copy_from_slice(&[255, 220 - cell as u8 * 40, 80, fade]);
                }
            }
        }
    }
    (w, h, rgba)
}

pub fn register(plugin: &mut ParticlePlugin, device: &wgpu::Device, queue: &wgpu::Queue) -> Ids {
    let mut add = |asset: EffectAsset| plugin.add_effect(device, asset);

    let sparks = add(
        EffectAsset::new("2d_sparks")
            .with_capacity(4_000)
            .with_emitter(Emitter {
                spawn: SpawnShape::Ring {
                    radius: 1.5,
                    thickness: 0.2,
                },
                velocity: VelocityDist::UniformBox {
                    min: [-1.0, -1.0, -1.0],
                    max: [1.0, 1.0, 1.0],
                },
                ..emitter(SpawnRate::PerSecond(500.0), (1.0, 2.0), [1.0, 0.7, 0.3, 1.0], 0.12)
            })
            .modifier(Modifier::CurlNoise {
                scale: 0.6,
                strength: 3.0,
                speed: 0.4,
            })
            .with_orient(Orient::InPlane)
            .with_plane_lock(on_plane()),
    );
    let embers = add(
        EffectAsset::new("2d_embers")
            .with_capacity(6_000)
            .with_emitter(Emitter {
                velocity: VelocityDist::UniformBox {
                    min: [-2.5, -2.5, -2.5],
                    max: [2.5, 2.5, 2.5],
                },
                ..emitter(SpawnRate::PerSecond(0.0), (0.6, 1.0), [0.4, 0.8, 1.0, 1.0], 0.1)
            })
            .modifier(Modifier::Drag(1.5)),
    );
    let shell = add(
        EffectAsset::new("2d_shell")
            .with_capacity(32)
            .with_emitter(Emitter {
                velocity: VelocityDist::UniformBox {
                    min: [-1.0, 3.0, -1.0],
                    max: [1.0, 5.0, 1.0],
                },
                ..emitter(SpawnRate::PerSecond(1.5), (0.8, 1.1), [1.0, 1.0, 1.0, 1.0], 0.2)
            })
            .modifier(Modifier::Accel([0.0, -4.0, 0.0]))
            .with_plane_lock(on_plane())
            .with_sub_emitter(SubEmitter::new(embers, EventCondition::OnDeath, 60)),
    );

    let smoke = add(
        EffectAsset::new("2d_smoke")
            .with_capacity(2_000)
            .with_blend(ParticleBlend::Multiply)
            .with_emitter(Emitter {
                spawn: SpawnShape::Circle {
                    radius: 0.6,
                    volume: true,
                },
                velocity: VelocityDist::UniformBox {
                    min: [-0.3, 0.4, 0.0],
                    max: [0.3, 0.9, 0.0],
                },
                ..emitter(SpawnRate::PerSecond(40.0), (3.0, 4.0), [0.3, 0.3, 0.35, 0.6], 1.6)
            })
            .with_plane_lock(on_plane())
            .with_painter(PainterPlacement::new(2, 0.0)),
    );
    let glow = add(
        EffectAsset::new("2d_glow")
            .with_capacity(2_000)
            .with_blend(ParticleBlend::Screen)
            .with_emitter(Emitter {
                spawn: SpawnShape::Rectangle {
                    half_size: [1.2, 1.2],
                },
                ..emitter(SpawnRate::PerSecond(60.0), (1.5, 2.5), [0.6, 0.5, 0.2, 0.8], 1.4)
            })
            .with_plane_lock(on_plane())
            .with_painter(PainterPlacement::new(2, 0.0)),
    );

    let hit = add(
        EffectAsset::new("2d_hit_sparks")
            .with_capacity(1_000)
            .with_emitter(Emitter {
                velocity: VelocityDist::UniformBox {
                    min: [0.5, -2.0, 0.0],
                    max: [4.0, 2.0, 0.0],
                },
                ..emitter(SpawnRate::PerSecond(120.0), (0.3, 0.6), [1.0, 0.9, 0.4, 1.0], 0.15)
            })
            .with_plane_lock(on_plane())
            .with_painter(PainterPlacement::new(2, 1.0)),
    );
    let dust = add(
        EffectAsset::new("2d_dust")
            .with_capacity(1_000)
            .with_blend(ParticleBlend::Alpha)
            .with_emitter(Emitter {
                spawn: SpawnShape::Segment {
                    a: [-0.6, -0.9],
                    b: [0.6, -0.9],
                },
                velocity: VelocityDist::UniformBox {
                    min: [-0.6, 0.1, 0.0],
                    max: [0.6, 0.5, 0.0],
                },
                ..emitter(SpawnRate::PerSecond(30.0), (1.0, 1.6), [0.7, 0.6, 0.5, 0.7], 0.6)
            })
            .with_plane_lock(on_plane())
            .with_painter(PainterPlacement::new(2, -1.0)),
    );

    let (w, h, sheet) = burst_sheet();
    let burst = |name: &str| {
        EffectAsset::new(name)
            .with_capacity(64)
            .with_emitter(emitter(SpawnRate::PerSecond(3.0), (1.0, 1.0), [1.0; 4], 1.0))
            .with_texture_rgba(w, h, sheet.clone())
            .with_flipbook(Flipbook {
                columns: 4,
                rows: 1,
                fps: 0.0,
            })
            .with_integer_scale(4.0)
            .with_plane_lock(on_plane())
    };
    let pixel = add(
        burst("2d_pixel_burst")
            .with_texture_filter(TextureFilter::Nearest)
            .with_pixel_snap(true),
    );
    let linear = add(burst("2d_linear_burst"));

    let square = |hook: FragmentHook| {
        EffectAsset::new("2d_hooked")
            .with_capacity(8)
            .with_blend(ParticleBlend::Alpha)
            .with_emitter(emitter(SpawnRate::Burst { count: 1 }, (1e6, 1e6), [0.3, 0.6, 1.0, 1.0], 2.0))
            .with_texture_rgba(1, 1, vec![255; 4])
            .with_fragment_hook(hook)
            .with_plane_lock(on_plane())
            .with_painter(PainterPlacement::new(2, 0.0))
    };
    let flash = add(square(FragmentHook::flash([1.0, 1.0, 1.0], 0.0)));
    let dissolve = add(square(FragmentHook::dissolve(0.0, [1.0, 0.5, 0.1], 0.08, 2.0, None)));

    let platforms = Collider2dSet::new(
        PLATFORMS
            .iter()
            .map(|(c, s)| Collider2d::Rect {
                min: [c[0] - s[0] * 0.5, c[1] - s[1] * 0.5],
                max: [c[0] + s[0] * 0.5, c[1] + s[1] * 0.5],
            })
            .collect(),
    );
    let set = plugin.add_collider_set(device, queue, &platforms);
    let mut add = |asset: EffectAsset| plugin.add_effect(device, asset);
    let splash = add(
        EffectAsset::new("2d_splash")
            .with_capacity(4_000)
            .with_emitter(Emitter {
                velocity: VelocityDist::UniformBox {
                    min: [-1.5, 0.5, 0.0],
                    max: [1.5, 2.0, 0.0],
                },
                ..emitter(SpawnRate::PerSecond(0.0), (0.25, 0.4), [0.6, 0.8, 1.0, 1.0], 0.08)
            })
            .modifier(Modifier::Accel([0.0, -9.8, 0.0]))
            .with_plane_lock(on_plane())
            .with_painter(PainterPlacement::new(3, 0.0)),
    );
    let rain = add(
        EffectAsset::new("2d_rain")
            .with_capacity(6_000)
            .with_emitter(Emitter {
                spawn: SpawnShape::Segment {
                    a: [-7.0, 0.0],
                    b: [7.0, 0.0],
                },
                velocity: VelocityDist::Fixed([0.0, -9.0, 0.0]),
                ..emitter(SpawnRate::PerSecond(400.0), (2.0, 2.0), [0.5, 0.7, 1.0, 0.8], 0.06)
            })
            .with_render(viewport_lib_particles::ParticleRender::Billboard { stretch: 0.08 })
            .with_plane_lock(on_plane())
            .with_collide(Collide2d {
                set,
                response: CollisionResponse::Kill,
            })
            .with_sub_emitter(SubEmitter::new(splash, EventCondition::OnContact, 3))
            .with_painter(PainterPlacement::new(3, 0.0)),
    );

    Ids {
        sparks,
        shell,
        embers,
        smoke,
        glow,
        hit,
        dust,
        pixel,
        linear,
        flash,
        dissolve,
        rain,
        splash,
    }
}

pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.heading("2D effects");
    ui.label("Effects on the XY plane, top-down, among painter sprites.");
    ui.separator();
    for (demo, label) in [
        (Demo::PlaneLock, "Plane lock"),
        (Demo::Blends, "Multiply and screen"),
        (Demo::Painter, "Painter order"),
        (Demo::PixelArt, "Pixel art"),
        (Demo::Hooks, "Flash and dissolve"),
        (Demo::Rain, "Rain on platforms"),
    ] {
        ui.radio_value(&mut state.demo, demo, label);
    }
    ui.separator();
    ui.label(match state.demo {
        Demo::PlaneLock => "Curl noise swirls the sparks but cannot lift them off the plane; the firework's sparks inherit the lock.",
        Demo::Blends => "Smoke multiplies the ground darker; the glows screen it lighter and never pass white.",
        Demo::Painter => "Sparks are keyed above the character on its layer, dust below it.",
        Demo::PixelArt => "Left: nearest, snapped, 4 texels a unit. Right: the same sheet sampled linearly.",
        Demo::Hooks => "The left square flashes, the right one dissolves and returns, each from a per-instance hook param.",
        Demo::Rain => "Rain dies on the platforms it hits, and each drop splashes where it lands.",
    });
}

/// The orthographic top-down camera the page draws with.
pub fn camera(w: f32, h: f32) -> Camera {
    let mut cam = Camera::look_at([0.0, 0.0, 20.0], [0.0, 0.0, 0.0], glam::Vec3::Y)
        .orthographic()
        .with_aspect(w / h.max(1.0));
    cam.set_ortho_height(VIEW_HEIGHT);
    cam
}

fn sprite(layer: u32, at: [f32; 2], size: [f32; 2], colour: Colour) -> SpriteItem {
    let mut s = SpriteItem::default();
    s.positions = vec![[at[0], at[1], 0.0]];
    s.sizes = vec![size];
    s.default_colour = colour;
    s.size_mode = SpriteSizeMode::WorldSpace;
    s.order = SpriteOrder::Painter {
        layer,
        depth_test: false,
    };
    s
}

/// This frame's particles and the sprites they draw among.
pub fn scene(ids: &Ids, state: &mut State, dt: f32) -> (ParticleItems, Vec<SpriteItem>) {
    state.t += dt;
    let t = state.t;
    let mut items = ParticleItems::new().with_dt(dt);
    let ground = sprite(0, [0.0, 0.0], [40.0, 40.0], Colour::srgb_rgb(0.24, 0.3, 0.22));
    let mut sprites = vec![ground];
    let at = |id: EffectId, key: u64, x: f32, y: f32| ParticleItem::new(id).with_key(key).at([x, y, 0.0]);
    match state.demo {
        Demo::PlaneLock => {
            items.push(at(ids.sparks, 0, -3.0, 0.0));
            items.push(at(ids.shell, 0, 3.0, -2.0));
            items.push(at(ids.embers, 0, 0.0, 0.0));
        }
        Demo::Blends => {
            sprites.push(sprite(1, [-3.0, 0.0], [4.0, 4.0], Colour::srgb_rgb(0.9, 0.85, 0.6)));
            sprites.push(sprite(1, [3.0, 0.0], [4.0, 4.0], Colour::srgb_rgb(0.2, 0.25, 0.45)));
            items.push(at(ids.smoke, 0, -3.0, -1.5));
            items.push(at(ids.glow, 0, 3.0, 0.0));
        }
        Demo::Painter => {
            // The character, swaying so the sparks and dust follow it.
            let x = (t * 0.8).sin() * 2.0;
            sprites.push(sprite(2, [x, 0.0], [1.2, 2.0], Colour::srgb_rgb(0.8, 0.3, 0.3)));
            items.push(at(ids.hit, 0, x + 0.2, 0.3));
            items.push(at(ids.dust, 0, x, 0.0));
        }
        Demo::PixelArt => {
            items.push(at(ids.pixel, 0, -2.0, 0.0));
            items.push(at(ids.linear, 0, 2.0, 0.0));
        }
        Demo::Hooks => {
            let flash = ((t * 3.0).sin() * 0.5 + 0.5).powi(4);
            let dissolve = (t * 0.5).sin() * 0.5 + 0.5;
            items.push(at(ids.flash, 1, -2.0, 0.0).with_hook_param("flash", [flash, 0.0, 0.0, 0.0]));
            items.push(
                at(ids.dissolve, 2, 2.0, 0.0).with_hook_param("dissolve", [dissolve, 0.0, 0.0, 0.0]),
            );
        }
        Demo::Rain => {
            for (c, s) in PLATFORMS {
                sprites.push(sprite(2, c, s, Colour::srgb_rgb(0.45, 0.35, 0.25)));
            }
            items.push(at(ids.rain, 0, 0.0, 5.8));
            items.push(at(ids.splash, 0, 0.0, 0.0));
        }
    }
    (items, sprites)
}
