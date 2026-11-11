//! 2D collision with host collider sets.
//!
//! Every scene is a side view of an effect locked to the XZ plane: the camera
//! looks along +Y with +Z up, so plane x runs across the screen and plane y
//! (world Z) up it, at 16 pixels to a unit. Particles are small additive dots
//! on black.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::{Camera, Colour};
use viewport_lib_plugins::item_types::lighting_2d::{Occluder2d, occluders_from_tilemap};
use viewport_lib_plugins::item_types::tilemap::{Tile, TileOrientation, Tilemap, Tileset};

use viewport_lib_particles::{
    Collide2d, Collider2d, Collider2dSet, Collider2dSetId, ColliderEdge, CollisionResponse,
    EffectAsset, EffectId, Emitter, Modifier, ParticleBlend, ParticleItem, ParticleItems,
    ParticlePlane, ParticlePlugin, PlaneLock, SpawnRate, SpawnShape, VelocityDist,
};

const SIZE: u32 = 64;
const PX_PER_UNIT: f32 = 16.0;
/// World Z at the centre of the view.
const CENTRE_Z: f32 = 1.0;
const CENTRE_Z64: f64 = CENTRE_Z as f64;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("particles-collide-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

fn lock() -> PlaneLock {
    PlaneLock::new(ParticlePlane::Xz, 0.0)
}

fn floor(y: f32) -> Collider2d {
    Collider2d::Segment {
        a: [-10.0, y],
        b: [10.0, y],
    }
}

/// Plane coordinates of pixel `(x, y)`'s centre.
fn plane_of(x: u32, y: u32) -> [f32; 2] {
    let c = SIZE as f32 * 0.5;
    [
        (x as f32 + 0.5 - c) / PX_PER_UNIT,
        CENTRE_Z + (c - (y as f32 + 0.5)) / PX_PER_UNIT,
    ]
}

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: ViewportRenderer,
    target: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Scene {
    /// A scene with the effects `build` registers against the device and queue.
    fn new<T>(build: impl FnOnce(&mut ParticlePlugin, &wgpu::Device, &wgpu::Queue) -> T) -> Option<(Self, T)> {
        let (device, queue) = headless_device()?;
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let mut plugin = ParticlePlugin::new();
        let out = build(&mut plugin, &device, &queue);
        renderer
            .with_item_type_plugin(&device, Box::new(plugin))
            .expect("register item type plugin");
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("particles-collide-target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        Some((
            Self {
                device,
                queue,
                renderer,
                target,
                view,
            },
            out,
        ))
    }

    /// Render one frame of `ids` at `origin`, applying `edit` to the items
    /// first, and return the lit pixels in plane coordinates.
    fn frame(
        &mut self,
        ids: &[EffectId],
        origin: [f32; 3],
        edit: impl FnOnce(&mut ParticleItems),
    ) -> Vec<[f32; 2]> {
        let mut cam = Camera::look_at([0.0, -10.0, CENTRE_Z64], [0.0, 0.0, CENTRE_Z64], glam::Vec3::Z)
            .orthographic()
            .with_aspect(1.0);
        cam.set_ortho_height(SIZE as f32 / PX_PER_UNIT);
        let mut frame = FrameData::default();
        frame.camera.render_camera = RenderCamera::from_camera(&cam);
        frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some(Colour::linear_rgb(0.0, 0.0, 0.0));
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        for &id in ids {
            let mut item = ParticleItem::new(id).at(origin);
            item.settings.cast_shadows = false;
            items.push(item);
        }
        edit(&mut items);
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = self.renderer.owned().render(&self.device, &self.queue, &self.view, &frame);
        self.queue.submit(std::iter::once(cmd));

        let pixels = read_back(&self.device, &self.queue, &self.target);
        let mut lit = Vec::new();
        for y in 0..SIZE {
            for x in 0..SIZE {
                let i = ((y * SIZE + x) * 4) as usize;
                if pixels[i] as u32 + pixels[i + 1] as u32 + pixels[i + 2] as u32 > 60 {
                    lit.push(plane_of(x, y));
                }
            }
        }
        lit
    }

    fn frames(&mut self, ids: &[EffectId], origin: [f32; 3], n: u32) -> Vec<[f32; 2]> {
        let mut lit = Vec::new();
        for _ in 0..n {
            lit = self.frame(ids, origin, |_| {});
        }
        lit
    }
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let bytes = (SIZE * SIZE * 4) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("collide-readback"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 4),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    let out = slice.get_mapped_range().to_vec();
    buf.unmap();
    out
}

/// One dot dropped from rest under gravity.
fn ball(collide: Collide2d) -> EffectAsset {
    EffectAsset::new("ball")
        .with_capacity(4)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::Burst { count: 1 },
            lifetime: (100.0, 100.0),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.15,
        })
        .modifier(Modifier::Accel([0.0, 0.0, -9.8]))
        .with_plane_lock(lock())
        .with_collide(collide)
}

/// Rain falling straight down from a line across the top of the view.
fn rain(collide: Option<Collide2d>, lifetime: f32) -> EffectAsset {
    let mut asset = EffectAsset::new("rain")
        .with_capacity(4_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(600.0),
            lifetime: (lifetime, lifetime),
            spawn: SpawnShape::Segment {
                a: [-1.9, 0.0],
                b: [1.9, 0.0],
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, -4.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.15,
        })
        .with_plane_lock(lock());
    if let Some(c) = collide {
        asset = asset.with_collide(c);
    }
    asset
}

/// Where the rain spawns: the top of the view.
const RAIN_TOP: [f32; 3] = [0.0, 0.0, 2.9];

#[test]
fn a_dropped_particle_rebounds_to_its_restitution_height() {
    let Some((mut scene, ids)) = Scene::new(|p, d, q| {
        let set = p.add_collider_set(d, q, &Collider2dSet::new(vec![floor(0.0)]));
        vec![p.add_effect(
            d,
            ball(Collide2d {
                set,
                response: CollisionResponse::Bounce {
                    restitution: 0.5,
                    friction: 0.0,
                },
            }),
        )]
    }) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Dropped from 2 units, it lands after about 0.64 s at 6.3 units/s and
    // leaves at half that, rising a quarter of the drop: 0.5 units.
    let mut heights = Vec::new();
    for _ in 0..110 {
        let lit = scene.frame(&ids, [0.0, 0.0, 2.0], |_| {});
        if !lit.is_empty() {
            heights.push(lit.iter().map(|p| p[1]).sum::<f32>() / lit.len() as f32);
        }
    }
    let landed = heights
        .iter()
        .position(|&h| h < 0.2)
        .expect("the ball should reach the floor");
    let low = heights.iter().cloned().fold(f32::MAX, f32::min);
    assert!(low > -0.15, "the ball should never pass the floor, lowest {low}");
    let peak = heights[landed..].iter().cloned().fold(f32::MIN, f32::max);
    assert!(
        (0.35..=0.7).contains(&peak),
        "a restitution of 0.5 should rebound to about 0.5 units, got {peak}"
    );
}

#[test]
fn kill_and_stick_stop_rain_at_the_floor() {
    let lit_for = |response: Option<CollisionResponse>| {
        let (mut scene, ids) = Scene::new(|p, d, q| {
            let set = p.add_collider_set(d, q, &Collider2dSet::new(vec![floor(0.0)]));
            let collide = response.map(|response| Collide2d { set, response });
            vec![p.add_effect(d, rain(collide, 3.0))]
        })?;
        Some(scene.frames(&ids, RAIN_TOP, 90))
    };
    let Some(free) = lit_for(None) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let kill = lit_for(Some(CollisionResponse::Kill)).unwrap();
    let stick = lit_for(Some(CollisionResponse::Stick)).unwrap();
    let below = |lit: &[[f32; 2]]| lit.iter().filter(|p| p[1] < -0.15).count();
    let on_floor = |lit: &[[f32; 2]]| lit.iter().filter(|p| p[1].abs() < 0.12).count();

    assert!(below(&free) > 20, "without collision rain should fall past the floor");
    assert_eq!(below(&kill), 0, "killed rain should not pass the floor");
    assert_eq!(below(&stick), 0, "stuck rain should not pass the floor");
    assert!(
        on_floor(&stick) > on_floor(&kill) + 20,
        "stuck rain should gather on the floor: stick {}, kill {}",
        on_floor(&stick),
        on_floor(&kill)
    );
}

#[test]
fn a_fast_particle_does_not_pass_a_thin_wall() {
    // Spawned at x = -1.5 moving at 90 units/s, a particle covers 1.5 units a
    // 60 Hz step: -1.5, 0, 1.5, then off the view. The wall from 0.25 to 1.25
    // is crossed inside one step.
    let shot = |collide: bool| {
        let (mut scene, ids) = Scene::new(|p, d, q| {
            let set = p.add_collider_set(
                d,
                q,
                &Collider2dSet::new(vec![Collider2d::Rect {
                    min: [0.25, -5.0],
                    max: [1.25, 5.0],
                }]),
            );
            let mut asset = rain(
                collide.then_some(Collide2d {
                    set,
                    response: CollisionResponse::Kill,
                }),
                1.0,
            )
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(600.0),
                lifetime: (1.0, 1.0),
                spawn: SpawnShape::Segment {
                    a: [0.0, -1.0],
                    b: [0.0, 1.0],
                },
                velocity: VelocityDist::Fixed([90.0, 0.0, 0.0]),
                colour: [1.0, 1.0, 1.0, 1.0],
                size: 0.15,
            });
            asset.plane_lock = Some(lock());
            vec![p.add_effect(d, asset)]
        })?;
        Some(scene.frames(&ids, [-1.5, 0.0, CENTRE_Z], 20))
    };
    let Some(free) = shot(false) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let walled = shot(true).unwrap();
    let past = |lit: &[[f32; 2]]| lit.iter().filter(|p| p[0] > 1.3).count();
    assert!(past(&free) > 5, "without collision the particles should show past the wall");
    assert_eq!(past(&walled), 0, "no particle should pass the wall in one step");
}

/// A collider for each `occluders_from_tilemap` occluder.
fn colliders(occluders: Vec<Occluder2d>) -> Collider2dSet {
    Collider2dSet::new(
        occluders
            .into_iter()
            .map(|o| match o {
                Occluder2d::Rect { min, max } => Collider2d::Rect { min, max },
                Occluder2d::Polygon(points) => Collider2d::Polygon(points),
                Occluder2d::Segment { a, b } => Collider2d::Segment { a, b },
                Occluder2d::Edges(edges) => Collider2d::Edges(
                    edges.iter().map(|e| ColliderEdge { a: e.a, b: e.b }).collect(),
                ),
            })
            .collect(),
    )
}

/// The edges' bounds, `[min_x, min_y, max_x, max_y]`.
fn bounds(set: &Collider2dSet) -> [f32; 4] {
    let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for e in set.edges() {
        for p in [e.a, e.b] {
            b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
        }
    }
    b
}

#[test]
fn tilemap_platforms_stop_the_rain() {
    // Two platforms of solid cells, half a unit square: one on the left at
    // y = 0, one on the right at y = 1.
    let mut map = Tilemap::sparse(Tileset::new(None, [16, 16], 1), 8);
    for col in -3..=-1 {
        map.set(col, 0, Tile::new(0));
    }
    for col in 1..=3 {
        map.set(col, 2, Tile::new(0));
    }
    let set = colliders(occluders_from_tilemap(&map, [0.0, 0.0], [0.5, 0.5], |t| {
        t != Tile::EMPTY
    }));
    assert!(!set.edges().is_empty());
    let Some((mut scene, ids)) = Scene::new(|p, d, q| {
        let id = p.add_collider_set(d, q, &set);
        vec![p.add_effect(
            d,
            rain(
                Some(Collide2d {
                    set: id,
                    response: CollisionResponse::Kill,
                }),
                3.0,
            ),
        )]
    }) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let lit = scene.frames(&ids, RAIN_TOP, 90);
    let under = |x0: f32, x1: f32, y: f32| {
        lit.iter()
            .filter(|p| p[0] > x0 && p[0] < x1 && p[1] < y)
            .count()
    };
    assert_eq!(under(-1.4, -0.1, -0.1), 0, "rain should not fall under the left platform");
    assert_eq!(under(0.6, 1.4, 0.9), 0, "rain should not fall under the right platform");
    assert!(under(-0.05, 0.45, 0.0) > 10, "rain should fall in the gap between them");
}

#[test]
fn an_isometric_tile_stops_the_rain() {
    // One diamond cell; its occluder edges come one per open cell side.
    let mut tileset = Tileset::new(None, [32, 16], 1);
    tileset.orientation = TileOrientation::Isometric;
    let mut map = Tilemap::sparse(tileset, 8);
    map.set(0, 0, Tile::new(0));
    let set = colliders(occluders_from_tilemap(&map, [0.0, 0.0], [1.0, 1.0], |t| {
        t != Tile::EMPTY
    }));
    let [x0, y0, x1, _] = bounds(&set);
    let cx = (x0 + x1) * 0.5;
    let Some((mut scene, ids)) = Scene::new(|p, d, q| {
        let id = p.add_collider_set(d, q, &set);
        vec![p.add_effect(
            d,
            rain(
                Some(Collide2d {
                    set: id,
                    response: CollisionResponse::Kill,
                }),
                3.0,
            ),
        )]
    }) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let lit = scene.frames(&ids, RAIN_TOP, 90);
    let shadow = lit
        .iter()
        .filter(|p| (p[0] - cx).abs() < 0.1 && p[1] < y0 - 0.1)
        .count();
    assert_eq!(shadow, 0, "rain should not fall under the diamond's middle");
    let beside = lit.iter().filter(|p| p[0] > x1 + 0.2 && p[1] < y0).count();
    assert!(beside > 5, "rain should fall beside the diamond");
}

#[test]
fn replacing_a_set_takes_effect_on_the_next_frame() {
    let Some((mut scene, (ids, set))) = Scene::new(|p, d, q| {
        let set = p.add_collider_set(d, q, &Collider2dSet::default());
        let ids = vec![p.add_effect(
            d,
            rain(
                Some(Collide2d {
                    set,
                    response: CollisionResponse::Kill,
                }),
                1.0,
            ),
        )];
        (ids, set)
    }) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let set: Collider2dSetId = set;
    let below = |lit: &[[f32; 2]]| lit.iter().filter(|p| p[1] < 0.8).count();
    let lit = scene.frames(&ids, RAIN_TOP, 45);
    assert!(below(&lit) > 20, "with an empty set the rain should fall through");

    // A floor of many short segments outgrows the empty set's buffers, so the
    // replacement also rebinds the effect.
    let segments: Vec<Collider2d> = (0..64)
        .map(|i| {
            let x = -4.0 + i as f32 * 0.125;
            Collider2d::Segment {
                a: [x, 1.0],
                b: [x + 0.125, 1.0],
            }
        })
        .collect();
    let replacement = Collider2dSet::new(segments);
    scene.frame(&ids, RAIN_TOP, |items| {
        items.replace_collider_set(set, replacement.clone())
    });
    // Rain already under the floor has died within its one-second life.
    let lit = scene.frames(&ids, RAIN_TOP, 75);
    assert_eq!(below(&lit), 0, "the new floor should stop the rain");
}

#[test]
fn contact_events_spawn_splashes() {
    use viewport_lib_particles::{EventCondition, SubEmitter};
    // Invisible rain that dies on the floor, feeding a child of bright dots
    // that only spawns from contact events.
    let splash_pixels = |with_rain: bool| {
        let (mut scene, ids) = Scene::new(|p, d, q| {
            let set = p.add_collider_set(d, q, &Collider2dSet::new(vec![floor(0.0)]));
            let splash = p.add_effect(
                d,
                EffectAsset::new("splash")
                    .with_capacity(2_000)
                    .with_blend(ParticleBlend::Additive)
                    .with_emitter(Emitter {
                        rate: SpawnRate::PerSecond(0.0),
                        lifetime: (0.3, 0.3),
                        spawn: SpawnShape::Point,
                        velocity: VelocityDist::Fixed([0.0, 0.0, 0.5]),
                        colour: [1.0, 1.0, 1.0, 1.0],
                        size: 0.15,
                    })
                    .with_plane_lock(lock()),
            );
            let mut drops = rain(
                Some(Collide2d {
                    set,
                    response: CollisionResponse::Kill,
                }),
                3.0,
            )
            .with_sub_emitter(SubEmitter::new(splash, EventCondition::OnContact, 1));
            drops.emitter.size = 0.0;
            let rain_id = p.add_effect(d, drops);
            if with_rain { vec![splash, rain_id] } else { vec![splash] }
        })?;
        let lit = scene.frames(&ids, RAIN_TOP, 90);
        Some(lit.iter().filter(|p| p[1] > -0.1 && p[1] < 0.4).count())
    };
    let Some(with) = splash_pixels(true) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let without = splash_pixels(false).unwrap();
    assert_eq!(without, 0, "the splash has no spawner of its own");
    assert!(with > 10, "rain hitting the floor should splash, got {with} px");
}
