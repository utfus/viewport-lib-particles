//! Pixel-art presentation: nearest sampling, integer-scale sizing, pixel
//! snapping and the sprite row order.
//!
//! Every scene is a top-down orthographic view with one additive billboard on
//! an opaque black background, sized from its texture with `with_integer_scale` so each texel covers
//! a known number of pixels.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::{Camera, Colour};

use viewport_lib_particles::{
    EffectAsset, Emitter, Flipbook, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin,
    SpawnRate, SpawnShape, TextureFilter, VelocityDist,
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
        label: Some("particles-pixel-art-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// One still particle sampling a `w` x `h` texture of `rgba` bytes, sized at
/// `texels_per_unit`.
fn sheet(w: u32, h: u32, rgba: Vec<u8>, texels_per_unit: f32) -> EffectAsset {
    EffectAsset::new("sheet")
        .with_capacity(4)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::Burst { count: 1 },
            lifetime: (1000.0, 1000.0),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.1,
        })
        .with_texture_rgba(w, h, rgba)
        .with_integer_scale(texels_per_unit)
}

/// The image after drawing `asset` at world `x` under a camera of
/// `px_per_unit` pixels to a world unit.
fn render(asset: EffectAsset, x: f32, px_per_unit: f32) -> Option<Vec<u8>> {
    let (device, queue) = headless_device()?;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, asset);
    renderer
        .with_item_type_plugin(&device, Box::new(plugin))
        .expect("register item type plugin");
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-pixel-art-target"),
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
    let mut cam = Camera::look_at([0.0, 0.0, 10.0], [0.0, 0.0, 0.0], glam::Vec3::Y)
        .orthographic()
        .with_aspect(1.0);
    cam.set_ortho_height(SIZE as f32 / px_per_unit);
    // Three frames: the inline texture uploads on the first.
    for _ in 0..3 {
        let mut frame = FrameData::default();
        frame.camera.render_camera = RenderCamera::from_camera(&cam);
        frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some(Colour::linear_rgb(0.0, 0.0, 0.0));
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        let mut item = ParticleItem::new(id).at([x, 0.0, 0.0]);
        item.settings.cast_shadows = false;
        items.push(item);
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }
    Some(read_back(&device, &queue, &target))
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let bytes = (SIZE * SIZE * 4) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel-art-readback"),
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

/// RGB of pixel `(x, y)`, y down.
fn rgb(pixels: &[u8], x: u32, y: u32) -> [i32; 3] {
    let i = ((y * SIZE + x) * 4) as usize;
    [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32]
}

/// The lit pixels' bounding box, `[min_x, min_y, max_x, max_y]`.
fn lit_box(pixels: &[u8]) -> Option<[u32; 4]> {
    let mut b: Option<[u32; 4]> = None;
    for y in 0..SIZE {
        for x in 0..SIZE {
            if rgb(pixels, x, y).iter().sum::<i32>() > 60 {
                let e = b.get_or_insert([x, y, x, y]);
                e[0] = e[0].min(x);
                e[1] = e[1].min(y);
                e[2] = e[2].max(x);
                e[3] = e[3].max(y);
            }
        }
    }
    b
}

/// Whether channel `ch` is bright and well above the other two. Tone mapping
/// desaturates bright colours a little, so a pure texel is not pure on screen.
fn dominant(c: [i32; 3], ch: usize) -> bool {
    c[ch] > 150 && (0..3).all(|i| i == ch || c[i] < c[ch] - 150)
}

/// Red, green on the top row; blue, white on the bottom (row 0 first).
fn quadrants() -> Vec<u8> {
    vec![
        255, 0, 0, 255, 0, 255, 0, 255, //
        0, 0, 255, 255, 255, 255, 255, 255,
    ]
}

#[test]
fn snapped_billboards_land_on_whole_pixels() {
    // An odd-width quad's edges move with its centre's sub-pixel offset. Snapped,
    // the centre lands on the same whole pixel for offsets either side of it,
    // so both draw the same pixels; unsnapped they do not.
    for px_per_unit in [5.0f32, 7.0, 9.0] {
        let offset = 0.3 / px_per_unit;
        let draw = |snap: bool, x: f32| {
            render(sheet(1, 1, vec![255; 4], 1.0).with_pixel_snap(snap), x, px_per_unit)
        };
        let Some(a) = draw(true, offset) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let b = draw(true, -offset).unwrap();
        let (ba, bb) = (lit_box(&a).unwrap(), lit_box(&b).unwrap());
        assert_eq!(ba, bb, "snapped quads at {px_per_unit} px/unit should match");
        let width = ba[2] - ba[0] + 1;
        assert_eq!(width, px_per_unit as u32, "the quad should be one unit wide");

        let ua = lit_box(&draw(false, offset).unwrap()).unwrap();
        let ub = lit_box(&draw(false, -offset).unwrap()).unwrap();
        assert_ne!(ua, ub, "without the snap the sub-pixel offset should show");
    }
}

#[test]
fn nearest_sampling_reproduces_texels_with_row_zero_on_top() {
    // A 2x2 sheet at 2 texels per unit is one unit across; at 16 px per unit
    // each texel covers 8 x 8 pixels.
    let Some(nearest) = render(
        sheet(2, 2, quadrants(), 2.0).with_texture_filter(TextureFilter::Nearest),
        0.0,
        16.0,
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert_eq!(lit_box(&nearest), Some([24, 24, 39, 39]));
    let (r, g, b, w) = (
        rgb(&nearest, 27, 27),
        rgb(&nearest, 36, 27),
        rgb(&nearest, 27, 36),
        rgb(&nearest, 36, 36),
    );
    assert!(dominant(r, 0), "top-left should be red: {r:?}");
    assert!(dominant(g, 1), "top-right should be green: {g:?}");
    assert!(dominant(b, 2), "bottom-left should be blue: {b:?}");
    assert!(w.iter().all(|&c| c > 150), "bottom-right should be white: {w:?}");
    // Every pixel of a texel is that texel exactly, up to its edge.
    assert_eq!(rgb(&nearest, 31, 24), r, "red should run to the centre line");
    assert_eq!(rgb(&nearest, 32, 24), g, "green should start at the centre line");

    // Linear sampling blends across the centre line.
    let linear = render(sheet(2, 2, quadrants(), 2.0), 0.0, 16.0).unwrap();
    let edge = rgb(&linear, 31, 27);
    assert!(
        edge[1] > r[1] + 20,
        "linear sampling should blend green into red at the edge: {edge:?}, nearest {r:?}"
    );
}

#[test]
fn a_flipbook_starts_at_the_sheets_top_left_cell() {
    // Four one-texel cells; a stretched flipbook on a long life holds the first.
    let Some(pixels) = render(
        sheet(2, 2, quadrants(), 1.0)
            .with_texture_filter(TextureFilter::Nearest)
            .with_flipbook(Flipbook {
                columns: 2,
                rows: 2,
                fps: 0.0,
            }),
        0.0,
        16.0,
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // One cell at one texel per unit is one unit: 16 px.
    assert_eq!(lit_box(&pixels), Some([24, 24, 39, 39]));
    let c = rgb(&pixels, 32, 32);
    assert!(dominant(c, 0), "the first cell is the red one: {c:?}");
}
