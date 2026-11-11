//! Plane-locked simulation, the flat spawn shapes and in-plane billboards.
//!
//! Each test renders with an orthographic camera chosen so a plane lock is easy
//! to see: a side view puts a locked XY effect on one horizontal line of pixels,
//! a top view shows the flat shapes' outlines. Positions are read back as lit
//! pixels against a black background.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, EventCondition, Modifier, Orient, ParticleBlend, ParticlePlane,
    ParticleItem, ParticleItems, ParticlePlugin, PlaneLock, SpawnRate, SpawnShape, SubEmitter,
    VelocityDist,
};

const SIZE: u32 = 64; // 64 * 4 bytes = 256-byte rows, no padding.
/// Orthographic view height in world units, so one unit is `SIZE / HEIGHT` px.
const HEIGHT: f32 = 4.0;
const PX_PER_UNIT: f32 = SIZE as f32 / HEIGHT;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("particles-plane-lock-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// Looking along +Y at the origin with +Z up: world z is screen up.
fn side_camera() -> Camera {
    ortho(Camera::look_at([0.0, -10.0, 0.0], [0.0, 0.0, 0.0], glam::Vec3::Z))
}

/// Looking down -Z at the origin with `up` as screen up.
fn top_camera(up: glam::Vec3) -> Camera {
    ortho(Camera::look_at([0.0, 0.0, 10.0], [0.0, 0.0, 0.0], up))
}

fn ortho(cam: Camera) -> Camera {
    let mut cam = cam.orthographic().with_aspect(1.0);
    cam.set_ortho_height(HEIGHT);
    cam
}

fn frame(cam: &Camera) -> FrameData {
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera::from_camera(cam);
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

/// A small additive dot that draws one or two pixels per particle.
fn dot(name: &str, rate: SpawnRate) -> EffectAsset {
    EffectAsset::new(name)
        .with_capacity(4_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate,
            lifetime: (1.5, 2.0),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.06,
        })
}

/// Lit pixels of the last frame, as `(x, y)` with y down, after rendering
/// `frames` frames of `ids` at the origin.
fn render_lit(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    plugin: ParticlePlugin,
    ids: &[EffectId],
    cam: &Camera,
    frames: u32,
) -> Vec<(i32, i32)> {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    renderer
        .with_item_type_plugin(device, Box::new(plugin))
        .expect("register item type plugin");
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-plane-lock-target"),
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
    for _ in 0..frames {
        let mut f = frame(cam);
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        for &id in ids {
            items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        }
        f.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = renderer.owned().render(device, queue, &view, &f);
        queue.submit(std::iter::once(cmd));
    }
    let pixels = read_back(device, queue, &target);
    let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    let background = luma(((SIZE + 1) * 4) as usize);
    let mut lit = Vec::new();
    for y in 0..SIZE {
        for x in 0..SIZE {
            if luma(((y * SIZE + x) * 4) as usize) > background + 60 {
                lit.push((x as i32, y as i32));
            }
        }
    }
    lit
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let bytes = (SIZE * SIZE * 4) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("plane-lock-readback"),
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

/// The largest distance, in pixels, of any lit pixel's row from `row`.
fn max_row_spread(lit: &[(i32, i32)], row: f32) -> f32 {
    lit.iter()
        .map(|&(_, y)| (y as f32 + 0.5 - row).abs())
        .fold(0.0, f32::max)
}

/// Screen row of world height `z` in the side view.
fn side_row(z: f32) -> f32 {
    SIZE as f32 * 0.5 - z * PX_PER_UNIT
}

/// A swirling cloud: random velocities with a vertical component, pushed
/// around by strong curl noise, so an unlocked cloud spreads in every axis. Its
/// dots are about two pixels across, so a locked line that falls on a pixel
/// boundary still covers pixel centres.
fn turbulent(name: &str) -> EffectAsset {
    dot(name, SpawnRate::PerSecond(600.0))
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(600.0),
            lifetime: (1.5, 2.0),
            spawn: SpawnShape::Sphere {
                radius: 0.3,
                volume: true,
            },
            velocity: VelocityDist::UniformBox {
                min: [-0.6, -0.6, -0.6],
                max: [0.6, 0.6, 0.6],
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.12,
        })
        .modifier(Modifier::CurlNoise {
            scale: 1.5,
            strength: 4.0,
            speed: 0.5,
        })
}

#[test]
fn locked_effect_stays_on_its_plane() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let spread = |lock: Option<PlaneLock>| {
        let mut plugin = ParticlePlugin::new();
        let mut asset = turbulent("cloud");
        if let Some(lock) = lock {
            asset = asset.with_plane_lock(lock);
        }
        let id = plugin.add_effect(&device, asset);
        let lit = render_lit(&device, &queue, plugin, &[id], &side_camera(), 60);
        assert!(!lit.is_empty(), "the cloud drew nothing");
        lit
    };

    let free = spread(None);
    let locked = spread(Some(PlaneLock::new(ParticlePlane::Xy, 0.5)));

    let free_spread = max_row_spread(&free, side_row(0.0));
    let locked_spread = max_row_spread(&locked, side_row(0.5));
    assert!(
        free_spread > 6.0,
        "the unlocked cloud should spread off the plane, spread {free_spread} px"
    );
    assert!(
        locked_spread <= 2.0,
        "a locked cloud should stay on the line at depth 0.5, spread {locked_spread} px"
    );
}

#[test]
fn xz_lock_uses_elevation_towards_the_camera() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // The XZ plane's elevation is world -Y: a lock at depth 1 sits at y = -1.
    // Seen from +X with +Z up, screen right is world +Y, so y = -1 is one unit
    // left of the centre, and a lock at y = +1 would draw on the right.
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(
        &device,
        turbulent("xz").with_plane_lock(PlaneLock::new(ParticlePlane::Xz, 1.0)),
    );
    let cam = ortho(Camera::look_at([10.0, 0.0, 0.0], [0.0, 0.0, 0.0], glam::Vec3::Z));
    let lit = render_lit(&device, &queue, plugin, &[id], &cam, 60);
    assert!(!lit.is_empty(), "the cloud drew nothing");
    let column = SIZE as f32 * 0.5 - PX_PER_UNIT;
    let spread = lit
        .iter()
        .map(|&(x, _)| (x as f32 + 0.5 - column).abs())
        .fold(0.0, f32::max);
    assert!(
        spread <= 2.0,
        "an XZ lock at depth 1 should draw on the column one unit left, spread {spread} px"
    );
}

#[test]
fn flat_shapes_spawn_inside_their_bounds() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let centre = SIZE as f32 * 0.5;
    // Plane coordinates of a lit pixel in the top view (screen up = world +Y).
    let to_plane = |(x, y): (i32, i32)| {
        (
            (x as f32 + 0.5 - centre) / PX_PER_UNIT,
            (centre - (y as f32 + 0.5)) / PX_PER_UNIT,
        )
    };
    // Half a pixel plus half a particle of slack on every bound.
    let slack = 0.5 / PX_PER_UNIT + 0.06;

    let shape_lit = |spawn: SpawnShape| {
        let mut plugin = ParticlePlugin::new();
        let id = plugin.add_effect(
            &device,
            dot("shape", SpawnRate::PerSecond(1_500.0))
                .with_emitter(Emitter {
                    rate: SpawnRate::PerSecond(1_500.0),
                    lifetime: (1.5, 2.0),
                    spawn,
                    velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
                    colour: [1.0, 1.0, 1.0, 1.0],
                    size: 0.06,
                })
                .with_plane_lock(PlaneLock::new(ParticlePlane::Xy, 0.0)),
        );
        let lit = render_lit(&device, &queue, plugin, &[id], &top_camera(glam::Vec3::Y), 30);
        assert!(lit.len() > 20, "{spawn:?} drew too little: {} px", lit.len());
        lit
    };

    for p in shape_lit(SpawnShape::Rectangle {
        half_size: [1.2, 0.6],
    }) {
        let (x, y) = to_plane(p);
        assert!(
            x.abs() <= 1.2 + slack && y.abs() <= 0.6 + slack,
            "rectangle pixel outside its bounds at ({x}, {y})"
        );
    }

    let ring = shape_lit(SpawnShape::Ring {
        radius: 1.2,
        thickness: 0.3,
    });
    for &p in &ring {
        let (x, y) = to_plane(p);
        let r = (x * x + y * y).sqrt();
        assert!(
            r >= 1.05 - slack && r <= 1.35 + slack,
            "ring pixel at radius {r}, outside 1.05..1.35"
        );
    }
    assert!(
        ring.iter().all(|&p| {
            let (x, y) = to_plane(p);
            x * x + y * y > 0.8
        }),
        "the ring's hole should stay empty"
    );

    for p in shape_lit(SpawnShape::Segment {
        a: [-1.0, -0.5],
        b: [1.0, 0.5],
    }) {
        let (x, y) = to_plane(p);
        // Distance from the line y = x / 2.
        let d = (y - 0.5 * x).abs() / (1.25f32).sqrt();
        assert!(
            d <= slack && x.abs() <= 1.0 + slack,
            "segment pixel off the segment at ({x}, {y})"
        );
    }
}

#[test]
fn in_plane_billboards_turn_with_the_world() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // One large square sprite at the origin, seen from above with the camera
    // rolled 45 degrees. A camera-plane billboard stays square on screen; an
    // in-plane one keeps the world's axes, so it shows as a diamond and the
    // screen square's corners stay dark.
    let lit_corner = |orient: Orient| {
        let mut plugin = ParticlePlugin::new();
        let id = plugin.add_effect(
            &device,
            dot("square", SpawnRate::Burst { count: 1 })
                .with_emitter(Emitter {
                    rate: SpawnRate::Burst { count: 1 },
                    lifetime: (10.0, 10.0),
                    spawn: SpawnShape::Point,
                    velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
                    colour: [1.0, 1.0, 1.0, 1.0],
                    size: 2.0,
                })
                .with_texture_rgba(1, 1, vec![255; 4])
                .with_orient(orient)
                .with_plane_lock(PlaneLock::new(ParticlePlane::Xy, 0.0)),
        );
        let up = glam::Vec3::new(1.0, 1.0, 0.0).normalize();
        let lit = render_lit(&device, &queue, plugin, &[id], &top_camera(up), 3);
        assert!(!lit.is_empty(), "{orient:?} drew nothing");
        // A point 0.85 units right and 0.85 up on screen: inside a screen-aligned
        // square of half size 1, outside a diamond of half diagonal 1.
        let c = SIZE as f32 * 0.5;
        let corner = ((c + 0.85 * PX_PER_UNIT) as i32, (c - 0.85 * PX_PER_UNIT) as i32);
        lit.contains(&corner)
    };

    assert!(
        lit_corner(Orient::ParallelCameraPlane),
        "a camera-plane billboard should cover the screen square's corner"
    );
    assert!(
        !lit_corner(Orient::InPlane),
        "an in-plane billboard should turn with the world and leave the corner dark"
    );
}

/// A locked parent that bursts sparks on the plane, and a child spawned where
/// each spark dies.
fn firework(plugin: &mut ParticlePlugin, device: &wgpu::Device, child: EffectAsset) -> [EffectId; 2] {
    let child = plugin.add_effect(device, child);
    let parent = plugin.add_effect(
        device,
        dot("shell", SpawnRate::PerSecond(40.0))
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(40.0),
                lifetime: (0.2, 0.3),
                spawn: SpawnShape::Point,
                velocity: VelocityDist::UniformBox {
                    min: [-1.0, -1.0, -1.0],
                    max: [1.0, 1.0, 1.0],
                },
                colour: [0.0, 0.0, 0.0, 0.0],
                size: 0.0,
            })
            .with_plane_lock(PlaneLock::new(ParticlePlane::Xy, 0.0))
            .with_sub_emitter(
                SubEmitter::new(child, EventCondition::OnDeath, 8).with_inherit_velocity(1.0),
            ),
    );
    [child, parent]
}

fn embers(lock: Option<PlaneLock>) -> EffectAsset {
    let mut asset = turbulent("embers").with_emitter(Emitter {
        rate: SpawnRate::PerSecond(0.0),
        lifetime: (1.0, 1.5),
        spawn: SpawnShape::Sphere {
            radius: 0.2,
            volume: true,
        },
        velocity: VelocityDist::UniformBox {
            min: [-0.6, -0.6, -0.6],
            max: [0.6, 0.6, 0.6],
        },
        colour: [1.0, 1.0, 1.0, 1.0],
        size: 0.12,
    });
    if let Some(lock) = lock {
        asset = asset.with_plane_lock(lock);
    }
    asset
}

#[test]
fn sub_emitter_children_inherit_the_lock() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // An unlocked child of a locked parent stays on the parent's plane.
    let mut plugin = ParticlePlugin::new();
    let ids = firework(&mut plugin, &device, embers(None));
    let lit = render_lit(&device, &queue, plugin, &ids, &side_camera(), 60);
    assert!(lit.len() > 4, "the children drew too little: {} px", lit.len());
    let spread = max_row_spread(&lit, side_row(0.0));
    assert!(
        spread <= 2.0,
        "unlocked children of a locked parent should stay on its plane, spread {spread} px"
    );

    // A child's own depth on the same plane wins.
    let mut plugin = ParticlePlugin::new();
    let ids = firework(
        &mut plugin,
        &device,
        embers(Some(PlaneLock::new(ParticlePlane::Xy, 1.0))),
    );
    let lit = render_lit(&device, &queue, plugin, &ids, &side_camera(), 60);
    assert!(lit.len() > 4, "the children drew too little: {} px", lit.len());
    let spread = max_row_spread(&lit, side_row(1.0));
    assert!(
        spread <= 2.0,
        "children with their own depth should draw at it, spread {spread} px"
    );
}

#[test]
fn mismatched_child_plane_is_rejected() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut plugin = ParticlePlugin::new();
    let child = plugin.add_effect(
        &device,
        embers(Some(PlaneLock::new(ParticlePlane::Xz, 0.0))),
    );
    let parent = dot("shell", SpawnRate::PerSecond(10.0))
        .with_plane_lock(PlaneLock::new(ParticlePlane::Xy, 0.0))
        .with_sub_emitter(SubEmitter::new(child, EventCondition::OnDeath, 4));
    let err = plugin
        .try_add_effect(&device, parent)
        .expect_err("a child on another plane must be refused");
    assert_eq!(err.parent, ParticlePlane::Xy);
    assert_eq!(err.child, ParticlePlane::Xz);
    assert_eq!(plugin.effect_count(), 1, "the refused parent must not register");
}
