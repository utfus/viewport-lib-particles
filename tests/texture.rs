//! Phase 12 verification: a billboard texture modulates the particle colour.
//!
//! Draws the same white-emitting effect twice with `TextureMode::Modulate`, once
//! through a solid-red texture and once through a solid-green one. Both blend
//! additively over the same background, so comparing the two runs isolates the
//! sampled texture colour: the red run must read higher in R than the green run,
//! and the green run higher in G. (A single run can't assert "R-dominant"
//! because the non-black background floods the other channels under additive
//! blending.)
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, Emitter, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin, SpawnRate,
    SpawnShape, TextureMode, VelocityDist,
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
        label: Some("particles-texture-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

fn frame_looking_at_origin() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera {
        view: cam.view_matrix(),
        projection: cam.proj_matrix(),
        eye_position: cam.eye_position().to_array(),
        forward: [0.0, 0.0, -1.0],
        orientation: cam.orientation,
        near: cam.effective_znear(),
        far: cam.zfar,
        distance: cam.distance,
        fov: cam.fov_y,
        aspect: 1.0,
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

/// A sparse, low-alpha white-emitting billboard cloud held at the origin. Kept
/// deliberately dim so additive accumulation stays well below saturation — the
/// texture's colour then shows as a clear channel gap rather than everything
/// clamping to ~255.
fn white_effect() -> EffectAsset {
    EffectAsset::new("tex")
        .with_capacity(4_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(400.0),
            lifetime: (1.0, 1.2),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 0.35],
            size: 1.2,
        })
}

/// Render for a few frames through a solid `colour` texture and return
/// `(center_r, center_g, center_b)`.
fn run(device: &wgpu::Device, queue: &wgpu::Queue, colour: [u8; 4]) -> (i32, i32, i32) {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    // 2x2 solid texture (colour is uniform across the cells).
    let pixels: Vec<u8> = std::iter::repeat(colour).take(4).flatten().collect();
    let tex = plugin.upload_texture(device, queue, 2, 2, &pixels);
    let asset = white_effect()
        .with_texture(tex)
        .with_texture_mode(TextureMode::Modulate);
    let id = plugin.add_effect(device, asset);
    renderer.with_item_type_plugin(device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-texture-target"),
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

    for _ in 0..3 {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = renderer.owned().render(device, queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(device, queue, &target);
    let i = (((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize;
    (pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32)
}

#[test]
fn texture_modulates_colour() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let (rr, rg, _rb) = run(&device, &queue, [255, 0, 0, 255]);
    let (gr, gg, _gb) = run(&device, &queue, [0, 255, 0, 255]);

    // The red-textured run must show more red than the green-textured run, and
    // vice versa for green — proving the sampled texel drives the colour.
    assert!(
        rr > gr + 40,
        "red texture not redder than green run: red-run r{rr}, green-run r{gr}"
    );
    assert!(
        gg > rg + 40,
        "green texture not greener than red run: green-run g{gg}, red-run g{rg}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-texture-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-texture-copy"),
    });
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
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();

    let mapped = staging.slice(..).get_mapped_range();
    mapped.to_vec()
}
