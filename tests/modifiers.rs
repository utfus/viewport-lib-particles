//! Modifier library: radial / vortex / conform forces and kill regions.
//!
//! - `kill_sphere_culls`: a kill-sphere covering the emitter kills its particles,
//!   so almost nothing draws versus the same effect without it.
//! - `kill_aabb_culls`: the same for an axis-aligned kill box.
//! - `radial_pushes_outward`: an outward radial force drives particles off the
//!   emitter, darkening the centre versus a still control.
//! - `vortex_and_conform_render`: a vortex swirl and a conform-to-sphere pull both
//!   simulate and draw (exercises those `eval_modifier` arms end to end).
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, Modifier, ParticleBlend, ParticleItem, ParticleItems,
    ParticlePlugin, SpawnRate, SpawnShape, VelocityDist,
};

const SIZE: u32 = 64;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("particles-modifiers-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

fn frame_looking_at_origin() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    let render_camera = RenderCamera::from_camera(&cam);
    frame.camera.render_camera = render_camera;
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-modifiers-readback"),
        size: (row_bytes * SIZE) as u64,
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
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
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
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(5)),
    });
    rx.recv().unwrap().unwrap();
    staging.slice(..).get_mapped_range().to_vec()
}

fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    effect: EffectAsset,
    total: usize,
    build: impl Fn(usize, EffectId) -> ParticleItems,
) -> Vec<u8> {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(device, effect);
    renderer.with_item_type_plugin(device, Box::new(plugin)).expect("register item type plugin");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-modifiers-target"),
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

    for f in 0..total {
        let mut frame = frame_looking_at_origin();
        frame
            .scene
            .submit_plugin_items(ParticlePlugin::TYPE_NAME, build(f, id));
        let cmd = renderer.owned().render(device, queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }
    read_back(device, queue, &target)
}

fn lit_pixels(pixels: &[u8]) -> usize {
    let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    let corner = luma(((SIZE + 1) * 4) as usize);
    (0..(SIZE * SIZE) as usize)
        .filter(|p| luma(p * 4) > corner + 60)
        .count()
}

/// Summed luma over a central `r`-by-`r` block, so the measurement is not a
/// single (easily saturated) pixel.
fn center_region_luma(pixels: &[u8], r: u32) -> i64 {
    let lo = SIZE / 2 - r / 2;
    let mut sum = 0i64;
    for y in lo..lo + r {
        for x in lo..lo + r {
            let i = ((y * SIZE + x) * 4) as usize;
            sum += pixels[i] as i64 + pixels[i + 1] as i64 + pixels[i + 2] as i64;
        }
    }
    sum
}

/// A still cloud: particles spawn spread out with no initial velocity, so a force
/// modifier is the only thing that moves them.
fn still_cloud() -> EffectAsset {
    EffectAsset::new("cloud")
        .with_capacity(6_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(3_000.0),
            lifetime: (2.5, 3.0),
            spawn: SpawnShape::Sphere {
                radius: 0.25,
                volume: true,
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.1,
        })
}

fn one_at_origin(_f: usize, id: EffectId) -> ParticleItems {
    let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
    items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
    items
}

#[test]
fn kill_sphere_culls() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let control = lit_pixels(&render(&device, &queue, still_cloud(), 16, one_at_origin));
    assert!(control > 0, "control cloud drew nothing");

    // A kill-sphere covering the whole visible cloud kills its particles.
    let killed = lit_pixels(&render(
        &device,
        &queue,
        still_cloud().modifier(Modifier::KillSphere {
            center: [0.0, 0.0, 0.0],
            radius: 8.0,
            kill_inside: true,
        }),
        16,
        one_at_origin,
    ));
    assert!(
        killed * 4 < control,
        "kill-sphere left too many particles (control {control}, killed {killed})"
    );
}

#[test]
fn kill_aabb_culls() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let control = lit_pixels(&render(&device, &queue, still_cloud(), 16, one_at_origin));
    let killed = lit_pixels(&render(
        &device,
        &queue,
        still_cloud().modifier(Modifier::KillAabb {
            min: [-8.0, -8.0, -8.0],
            max: [8.0, 8.0, 8.0],
            kill_inside: true,
        }),
        16,
        one_at_origin,
    ));
    assert!(
        killed * 4 < control,
        "kill-box left too many particles (control {control}, killed {killed})"
    );
}

#[test]
fn radial_concentrates_inward() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // A one-shot burst spread across a wide disc with no velocity. Without a force
    // it stays spread; a negative (inward) radial pulls it to the centre, so the
    // centre brightens. Concentration is a robust signal (unlike an outward push,
    // whose screen effect depends on the viewing plane).
    let burst = || {
        EffectAsset::new("burst")
            .with_capacity(3_000)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::Burst { count: 1_500 },
                lifetime: (4.0, 4.0),
                // A shell at radius 2: sparse at screen centre until it collapses.
                spawn: SpawnShape::Sphere {
                    radius: 2.0,
                    volume: false,
                },
                velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
                colour: [0.25, 0.25, 0.25, 1.0],
                size: 0.05,
            })
    };

    let control = center_region_luma(&render(&device, &queue, burst(), 48, one_at_origin), 8);
    // Inward radial pulls the shell in; drag damps the collapse so particles
    // settle at the centre instead of overshooting and re-expanding.
    let imploded = center_region_luma(
        &render(
            &device,
            &queue,
            burst()
                .modifier(Modifier::Radial {
                    center: [0.0, 0.0, 0.0],
                    strength: -15.0,
                    falloff: 0.0,
                })
                .modifier(Modifier::Drag(5.0)),
            48,
            one_at_origin,
        ),
        8,
    );
    assert!(
        imploded > control * 5 / 4,
        "inward radial did not concentrate at the centre (control {control}, imploded {imploded})"
    );
}

#[test]
fn vortex_and_conform_render() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let vortex = still_cloud().modifier(Modifier::Vortex {
        center: [0.0, 0.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        strength: 6.0,
        falloff: 0.0,
    });
    assert!(
        lit_pixels(&render(&device, &queue, vortex, 16, one_at_origin)) > 0,
        "vortex effect drew nothing"
    );

    let conform = still_cloud().modifier(Modifier::Conform {
        center: [0.0, 0.0, 0.0],
        radius: 1.0,
        attraction: 6.0,
        influence: 3.0,
        sticky: 0.5,
    });
    assert!(
        lit_pixels(&render(&device, &queue, conform, 16, one_at_origin)) > 0,
        "conform effect drew nothing"
    );
}
