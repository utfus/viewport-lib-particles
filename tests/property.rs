//! A runtime property read in the expression graph, and
//! the per-frame value override channel on `ParticleItem`.
//!
//! Builds a program whose per-particle colour is a declared `vec3` property
//! `tint`, then renders the same effect twice: once with the property left at its
//! (black) default, once overridden to white per frame. The white run must read
//! back a clearly brighter center, proving the property lowers to a uniform read
//! and the host-set value drives the shader.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    Attribute, EffectAsset, EffectProgram, Module, ParticleBlend, ParticleItem, ParticleItems,
    ParticlePlugin, PropertyValue, SpawnRate,
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
        label: Some("particles-property-test"),
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

/// An effect whose per-particle colour is the `tint` property (default black),
/// held near the origin by a small downward acceleration.
fn property_effect() -> EffectAsset {
    let mut m = Module::new();
    let tint = m.property("tint");
    let size = m.lit(1.5);
    let gravity = m.lit_vec3([0.0, 0.0, -1.0]);

    let mut program = EffectProgram::new()
        .with_rate(SpawnRate::PerSecond(6_000.0))
        .with_lifetime(2.0, 3.0)
        .property("tint", PropertyValue::Vec3([0.0, 0.0, 0.0]));
    program.module = m;
    let program = program
        .set(Attribute::Colour, tint)
        .set(Attribute::Size, size)
        .update(viewport_lib_particles::UpdateOp::Accelerate(gravity));

    EffectAsset::new("prop")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_program(program)
}

/// Render the effect for a few frames, optionally overriding `tint` each frame,
/// and return `(center_sum, corner_sum)` of the read-back RGB.
fn run(device: &wgpu::Device, queue: &wgpu::Queue, tint: Option<[f32; 3]>) -> (i32, i32) {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(device, property_effect());
    renderer.with_item_type_plugin(device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-property-target"),
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
        let mut item = ParticleItem::new(id).at([0.0, 0.0, 0.0]);
        if let Some(t) = tint {
            item = item.with_property("tint", PropertyValue::Vec3(t));
        }
        items.push(item);
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);

        let cmd = renderer.owned().render(device, queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(device, queue, &target);
    let sum = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
    };
    (sum(SIZE / 2, SIZE / 2), sum(1, 1))
}

#[test]
fn property_override_drives_colour() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // Default property (black tint): additive black adds nothing, center stays
    // near the background.
    let (default_center, _) = run(&device, &queue, None);
    // Overridden to white: the property read now yields a bright colour.
    let (white_center, white_corner) = run(&device, &queue, Some([1.0, 1.0, 1.0]));

    assert!(
        white_center > white_corner + 100,
        "white run center {white_center} not clearly brighter than corner {white_corner}"
    );
    assert!(
        white_center > default_center + 100,
        "property override had no effect: default center {default_center}, white center {white_center}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-property-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-property-copy"),
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
