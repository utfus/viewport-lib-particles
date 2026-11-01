//! Custom fragment shading hook for particle colour (Phase 33 / issue 018).
//!
//! `validate_*` are pure CPU checks of `FragmentHook::validate`. The GPU tests
//! render a billboard effect whose colour comes from a custom WGSL hook and
//! confirm the hook actually runs: a hook that samples a mask texture and tints it
//! through a param produces red vs green depending on the param, across both the
//! additive and OIT (alpha) blend routes. Comparing two runs isolates the hook's
//! output (a non-black background floods channels under additive blending).
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, Emitter, FragmentHook, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin,
    SpawnRate, SpawnShape, VelocityDist,
};

// ---------------------------------------------------------------------------
// Validation (CPU)
// ---------------------------------------------------------------------------

#[test]
fn validate_accepts_well_formed_hook() {
    let hook = FragmentHook::new(
        "fire",
        "fn shade_particle(f: ParticleFragment) -> vec4<f32> { return f.colour; }",
    )
    .param([1.0, 0.5, 0.2, 0.0]);
    assert!(hook.validate().is_ok(), "{:?}", hook.validate());
}

#[test]
fn validate_rejects_missing_function() {
    let hook = FragmentHook::new("fire", "fn helper() -> f32 { return 1.0; }");
    let err = hook.validate().expect_err("must require shade_particle");
    assert!(err.contains("shade_particle"), "{err}");
}

#[test]
fn validate_rejects_bad_name() {
    let hook = FragmentHook::new(
        "1bad",
        "fn shade_particle(f: ParticleFragment) -> vec4<f32> { return f.colour; }",
    );
    assert!(hook.validate().is_err(), "leading digit is not a valid ident");
}

#[test]
fn validate_rejects_too_many_textures() {
    let mut hook = FragmentHook::new(
        "fire",
        "fn shade_particle(f: ParticleFragment) -> vec4<f32> { return f.colour; }",
    );
    for _ in 0..5 {
        hook = hook.texture(1, 1, vec![255, 255, 255, 255]);
    }
    assert!(hook.validate().is_err(), "5 textures exceeds the cap of 4");
}

// ---------------------------------------------------------------------------
// End-to-end render
// ---------------------------------------------------------------------------

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
        label: Some("particles-fragment-hook-test"),
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

fn white_emitter() -> Emitter {
    Emitter {
        rate: SpawnRate::PerSecond(400.0),
        lifetime: (1.0, 1.2),
        spawn: SpawnShape::Point,
        velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
        colour: [1.0, 1.0, 1.0, 0.35],
        size: 1.2,
    }
}

/// A hook that samples its mask texture (htex0) and tints it by param lane 1
/// (`hook_params.lanes[1]`), so the sampled colour is the param's rgb.
const TINT_HOOK: &str = "\
fn shade_particle(f: ParticleFragment) -> vec4<f32> {
    let mask = textureSample(htex0, sprite_samp, f.uv).r;
    let tint = hook_params.lanes[1].rgb;
    return vec4<f32>(tint * mask, f.colour.a * mask);
}";

/// Render the tint hook with `tint` (param lane 1) under `blend`, returning the
/// centre `(r, g, b)`.
fn run(device: &wgpu::Device, queue: &wgpu::Queue, tint: [f32; 4], blend: ParticleBlend) -> (i32, i32, i32) {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();

    // A solid-white 2x2 mask so `mask == 1` everywhere the sprite covers.
    let mask: Vec<u8> = std::iter::repeat_n([255u8, 255, 255, 255], 4).flatten().collect();
    let hook = FragmentHook::new("tint", TINT_HOOK)
        .texture(2, 2, mask)
        .param(tint);
    let asset = EffectAsset::new("hooked")
        .with_capacity(4_000)
        .with_blend(blend)
        .with_emitter(white_emitter())
        .with_fragment_hook(hook);
    let id = plugin.add_effect(device, asset);
    renderer.with_item_type_plugin(device, Box::new(plugin)).expect("register item type plugin");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-fragment-hook-target"),
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
fn hook_drives_colour_additive() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (rr, rg, _) = run(&device, &queue, [1.0, 0.0, 0.0, 1.0], ParticleBlend::Additive);
    let (gr, gg, _) = run(&device, &queue, [0.0, 1.0, 0.0, 1.0], ParticleBlend::Additive);
    assert!(rr > gr + 40, "red hook not redder: red-run r{rr}, green-run r{gr}");
    assert!(gg > rg + 40, "green hook not greener: green-run g{gg}, red-run g{rg}");
}

#[test]
fn hook_drives_colour_oit() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Alpha blend routes through the OIT pass, exercising the hooked OIT pipeline.
    let (rr, rg, _) = run(&device, &queue, [1.0, 0.0, 0.0, 1.0], ParticleBlend::Alpha);
    let (gr, gg, _) = run(&device, &queue, [0.0, 1.0, 0.0, 1.0], ParticleBlend::Alpha);
    assert!(rr > gr + 20, "red hook (OIT) not redder: red-run r{rr}, green-run r{gr}");
    assert!(gg > rg + 20, "green hook (OIT) not greener: green-run g{gg}, red-run g{rg}");
}

/// A hook that defines a helper function and reads the group-0 `camera` uniform
/// (view-dependent colour, the iridescent-bubbles showcase shape) must compose,
/// compile, and draw — on the alpha/OIT route.
#[test]
fn hook_reads_camera_and_helpers() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let body = "\
fn tint(v: vec3<f32>) -> vec3<f32> { return v * 2.0; }
fn shade_particle(f: ParticleFragment) -> vec4<f32> {
    let view = normalize(camera.eye_pos - f.world);
    let k = clamp(abs(view.z), 0.2, 1.0);
    return vec4<f32>(tint(vec3<f32>(k, k, k)), f.colour.a);
}";
    let asset = EffectAsset::new("cam")
        .with_capacity(2_000)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(white_emitter())
        .with_fragment_hook(FragmentHook::new("cam", body));
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, asset);
    renderer.with_item_type_plugin(&device, Box::new(plugin)).expect("register item type plugin");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-hook-cam-target"),
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
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }
    let pixels = read_back(&device, &queue, &target);
    let lit = (0..(SIZE * SIZE) as usize)
        .filter(|p| pixels[p * 4] as i32 + pixels[p * 4 + 1] as i32 + pixels[p * 4 + 2] as i32 > 30)
        .count();
    assert!(lit > 0, "camera-reading hook drew nothing");
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-fragment-hook-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-fragment-hook-copy"),
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
