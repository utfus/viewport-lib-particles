//! Inline-bytes texture authored on the asset (issue 017).
//!
//! Mirrors `tests/texture.rs`, but instead of `ParticlePlugin::upload_texture`
//! (which needs a `device`/`queue` and a plugin handle the host lacks after
//! install), the texture is carried as CPU pixels on the `EffectAsset` via
//! `with_texture_rgba`. The plugin uploads it during its next `prepare`. Drawing
//! the same white emitter through a solid-red vs solid-green inline texture and
//! comparing the two runs isolates the sampled colour, proving the bytes were
//! uploaded and bound — through both the `add_effect` and `replace_effect`
//! delivery paths.
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
        label: Some("particles-texture-bytes-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

fn frame_looking_at_origin() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    let mut render_camera = RenderCamera::from_camera(&cam);
    render_camera.forward = [0.0, 0.0, -1.0];
    render_camera.far = cam.zfar;
    render_camera.aspect = 1.0;
    frame.camera.render_camera = render_camera;
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

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

/// A 2x2 solid-`colour` texture as raw RGBA8 bytes.
fn solid(colour: [u8; 4]) -> Vec<u8> {
    std::iter::repeat_n(colour, 4).flatten().collect()
}

/// Render the white emitter through a solid inline texture and return the centre
/// `(r, g, b)`. `via_replace` chooses the delivery path: `false` registers the
/// textured asset directly (`add_effect`); `true` registers a textureless asset,
/// then swaps in the inline-texture asset with `ParticleItems::replace_effect`.
fn run(device: &wgpu::Device, queue: &wgpu::Queue, colour: [u8; 4], via_replace: bool) -> (i32, i32, i32) {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();

    let textured = white_effect()
        .with_texture_rgba(2, 2, solid(colour))
        .with_texture_mode(TextureMode::Modulate);

    let id = if via_replace {
        // Start textureless; the texture arrives later through replace_effect.
        plugin.add_effect(device, white_effect())
    } else {
        plugin.add_effect(device, textured.clone())
    };
    renderer.with_item_type_plugin(device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-texture-bytes-target"),
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

    for f in 0..4 {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        // Deliver the inline texture via replace on the first frame.
        if via_replace && f == 0 {
            items.replace_effect(id, textured.clone());
        }
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
fn inline_texture_modulates_colour_via_add() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let (rr, rg, _rb) = run(&device, &queue, [255, 0, 0, 255], false);
    let (gr, gg, _gb) = run(&device, &queue, [0, 255, 0, 255], false);

    assert!(
        rr > gr + 40,
        "red inline texture not redder than green run: red-run r{rr}, green-run r{gr}"
    );
    assert!(
        gg > rg + 40,
        "green inline texture not greener than red run: green-run g{gg}, red-run g{rg}"
    );
}

#[test]
fn inline_texture_modulates_colour_via_replace() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let (rr, rg, _rb) = run(&device, &queue, [255, 0, 0, 255], true);
    let (gr, gg, _gb) = run(&device, &queue, [0, 255, 0, 255], true);

    assert!(
        rr > gr + 40,
        "red inline texture (via replace) not redder than green run: red-run r{rr}, green-run r{gr}"
    );
    assert!(
        gg > rg + 40,
        "green inline texture (via replace) not greener than red run: green-run g{gg}, red-run g{rg}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-texture-bytes-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-texture-bytes-copy"),
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
