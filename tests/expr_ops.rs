//! Expression op set, built-ins, and typed validation.
//!
//! The `validate_*` tests are pure CPU (no GPU needed): they exercise
//! `EffectProgram::validate` / `Module::infer_types` accepting a well-typed graph
//! and rejecting arity/type errors with node-level messages.
//!
//! `time_driven_program_runs` compiles a program using the time and maths ops (time, sin,
//! saturate, mix, vec constructors, and a per-particle `rand` in the *update*
//! kernel) to WGSL and renders it, proving the generated shaders compile and draw.
//! It skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    Attribute, EffectAsset, EffectProgram, Module, ParticleBlend, ParticleItem, ParticleItems,
    ParticlePlugin, SpawnRate, UpdateOp,
};

/// Build a program that pulses colour with `time`, varies size per particle, and
/// applies a small random-direction acceleration (rand in the *update* kernel).
/// Each builder call is bound to its own `let` because the builders take
/// `&mut self` and so cannot be nested.
fn wide_program() -> EffectProgram {
    let mut m = Module::new();
    let t = m.time();
    let s = m.sin(t);
    let half = m.lit(0.5);
    let scaled = m.mul(s, half);
    let shifted = m.add(scaled, half);
    // brightness in [0,1]: saturate(0.5 + 0.5*sin(time))
    let bright = m.saturate(shifted);
    let lo = m.lit(0.3);
    let hi = m.lit(1.0);
    let g = m.mix(lo, hi, bright);
    let one = m.lit(1.0);
    let colour = m.vec3(bright, g, one);
    let smin = m.lit(0.15);
    let smax = m.lit(0.3);
    let r = m.rand();
    let size = m.mix(smin, smax, r);
    let dir = m.rand_unit();
    let strength = m.lit(0.05);
    let accel = m.mul(dir, strength);

    EffectProgram::new()
        .with_module(m)
        .with_rate(SpawnRate::PerSecond(2000.0))
        .with_lifetime(1.5, 2.0)
        .set(Attribute::Colour, colour)
        .set(Attribute::Size, size)
        .update(UpdateOp::Accelerate(accel))
}

// ---------------------------------------------------------------------------
// Typed validation (CPU only)
// ---------------------------------------------------------------------------

/// A program exercising the wider op set, all well-typed, should validate.
#[test]
fn validate_accepts_wide_op_set() {
    let program = wide_program();

    assert!(
        program.validate().is_ok(),
        "well-typed program rejected: {:?}",
        program.validate()
    );
}

/// Writing a scalar to a vec3 attribute is a type error surfaced by validate.
#[test]
fn validate_rejects_attribute_type_mismatch() {
    let mut m = Module::new();
    let scalar = m.lit(1.0);
    let program = EffectProgram::new()
        .with_module(m)
        .set(Attribute::Position, scalar);

    let errs = program.validate().expect_err("scalar into Position must fail");
    assert!(
        errs.iter().any(|e| e.message.contains("Position")),
        "expected a Position type error, got {errs:?}"
    );
}

/// An op with the wrong operand shape (dot of two scalars) is caught by
/// inference, before any attribute check.
#[test]
fn validate_rejects_bad_op_arity() {
    let mut m = Module::new();
    let a = m.lit(1.0);
    let b = m.lit(2.0);
    let bad = m.dot(a, b); // dot needs vectors
    let accel = m.splat3(bad);
    let program = EffectProgram::new()
        .with_module(m)
        .update(UpdateOp::Accelerate(accel));

    let errs = program.validate().expect_err("dot of scalars must fail");
    assert!(
        errs.iter().any(|e| e.message.contains("dot")),
        "expected a dot type error, got {errs:?}"
    );
}

/// Swizzling a component the source vector does not have is a type error.
#[test]
fn validate_rejects_out_of_range_swizzle() {
    let mut m = Module::new();
    let v = m.lit_vec3([1.0, 2.0, 3.0]);
    let w = m.w(v); // vec3 has no .w
    let size = w;
    let program = EffectProgram::new()
        .with_module(m)
        .set(Attribute::Size, size);

    let errs = program.validate().expect_err("vec3 .w must fail");
    assert!(
        errs.iter().any(|e| e.message.contains("swizzle")),
        "expected a swizzle type error, got {errs:?}"
    );
}

// ---------------------------------------------------------------------------
// End-to-end codegen + render
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
        label: Some("particles-expr-ops-test"),
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

#[test]
fn time_driven_program_runs() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let program = wide_program();
    program.validate().expect("program should type-check");

    let asset = EffectAsset::new("expr-ops")
        .with_capacity(8_000)
        .with_blend(ParticleBlend::Additive)
        .with_program(program);

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, asset);
    renderer.with_item_type_plugin(&device, Box::new(plugin)).expect("register item type plugin");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-expr-ops-target"),
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

    for _ in 0..8 {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(&device, &queue, &target);
    let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    let corner = luma(((SIZE + 1) * 4) as usize);
    let center = luma((((SIZE / 2) * SIZE + SIZE / 2) * 4) as usize);
    assert!(
        center > corner + 40,
        "expected an additive glow at centre (centre {center}, corner {corner})"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-expr-ops-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-expr-ops-copy"),
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
