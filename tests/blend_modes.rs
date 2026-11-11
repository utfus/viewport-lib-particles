//! Multiply and screen blends.
//!
//! One large square particle sits in front of an opaque grey quad. Each run's
//! centre pixel is compared with the quad alone: multiply scales the colour
//! behind it (a coloured particle darkens the channels it lacks; a transparent
//! texel leaves the quad unchanged), screen lightens it (a grey particle
//! brightens; a black texel leaves it unchanged).
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SceneRenderItem, SurfaceSubmission, ViewportRenderer};
use viewport_lib::resources::MeshData;
use viewport_lib::scene::material::ShadingModel;
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, Emitter, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin,
    SpawnRate, SpawnShape, VelocityDist,
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
        label: Some("particles-blend-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// A large double-sided quad in the XY plane.
fn quad_mesh() -> MeshData {
    let s = 20.0;
    let mut mesh = MeshData::default();
    mesh.positions = vec![[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    mesh.indices = vec![0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2];
    mesh
}

/// One square particle, `colour` tinted, sampling a 1x1 `texel`.
fn square(blend: ParticleBlend, colour: [f32; 4], texel: [u8; 4]) -> EffectAsset {
    EffectAsset::new("square")
        .with_capacity(16)
        .with_blend(blend)
        .with_emitter(Emitter {
            rate: SpawnRate::Burst { count: 1 },
            lifetime: (1000.0, 1000.0),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour,
            size: 3.0,
        })
        .with_texture_rgba(1, 1, texel.to_vec())
}

/// The centre pixel's RGB over the grey quad, with `effect` in front of it (or
/// the quad alone for `None`).
fn centre(effect: Option<EffectAsset>) -> Option<[i32; 3]> {
    let (device, queue) = headless_device()?;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &quad_mesh())
        .expect("upload backdrop mesh");
    let mut material = viewport_lib::Material::default();
    material.shading_model = ShadingModel::Pbr;
    material.base_colour = viewport_lib::Colour::linear_rgb(0.5, 0.5, 0.5);
    let material = renderer
        .resources_mut()
        .upload_material(material)
        .expect("upload backdrop material");
    let mut backdrop = SceneRenderItem::new(mesh_id, material);
    backdrop.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.5)).into();

    let mut plugin = ParticlePlugin::new();
    let id = effect.map(|e| plugin.add_effect(&device, e));
    renderer
        .with_item_type_plugin(&device, Box::new(plugin))
        .expect("register item type plugin");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-blend-target"),
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

    // Three frames: the inline texture uploads on the first.
    for _ in 0..3 {
        let mut frame = FrameData::default();
        frame.camera.render_camera = RenderCamera::from_camera(&Camera::default());
        frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![backdrop.clone()].into());
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        if let Some(id) = id {
            // No shadow, so the quad behind changes only by the blend.
            let mut item = ParticleItem::new(id).at([0.0, 0.0, 0.0]);
            item.settings.cast_shadows = false;
            items.push(item);
        }
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(&device, &queue, &target);
    let i = (((SIZE / 2) * SIZE + (SIZE / 2)) * 4) as usize;
    Some([pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32])
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let bytes = (SIZE * SIZE * 4) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blend-readback"),
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

fn close(a: [i32; 3], b: [i32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 3)
}

#[test]
fn multiply_darkens_and_transparent_texels_leave_the_scene() {
    let Some(base) = centre(None) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(base[0] > 40, "the backdrop should be visible, got {base:?}");

    // A blue tint keeps blue and darkens red and green.
    let tinted = centre(Some(square(
        ParticleBlend::Multiply,
        [0.1, 0.1, 1.0, 1.0],
        [255; 4],
    )))
    .unwrap();
    assert!(
        tinted[0] < base[0] - 20 && tinted[1] < base[1] - 20,
        "multiply should darken red and green: base {base:?}, tinted {tinted:?}"
    );
    assert!(
        (tinted[2] - base[2]).abs() <= 3,
        "multiply by a full blue channel should leave blue: base {base:?}, tinted {tinted:?}"
    );

    // A transparent texel multiplies by white.
    let clear = centre(Some(square(
        ParticleBlend::Multiply,
        [0.1, 0.1, 1.0, 1.0],
        [0, 0, 0, 0],
    )))
    .unwrap();
    assert!(
        close(clear, base),
        "a transparent multiply texel should leave the scene: base {base:?}, got {clear:?}"
    );
}

#[test]
fn screen_lightens_and_black_texels_leave_the_scene() {
    let Some(base) = centre(None) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let lit = centre(Some(square(
        ParticleBlend::Screen,
        [0.5, 0.5, 0.5, 1.0],
        [255; 4],
    )))
    .unwrap();
    assert!(
        lit.iter().zip(base).all(|(l, b)| *l > b + 20),
        "screen should lighten every channel: base {base:?}, lit {lit:?}"
    );
    assert!(
        lit.iter().all(|&c| c < 255),
        "a grey screen over grey should not reach white: {lit:?}"
    );

    let black = centre(Some(square(
        ParticleBlend::Screen,
        [1.0, 1.0, 1.0, 1.0],
        [0, 0, 0, 255],
    )))
    .unwrap();
    assert!(
        close(black, base),
        "a black screen texel should leave the scene: base {base:?}, got {black:?}"
    );
}
