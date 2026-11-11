//! Hook parity with viewport-lib's sprite materials and effects.
//!
//! Sprite material bodies from viewport-lib's own tests draw particles through
//! `FragmentHook::from_material`; the ready-made flash, dissolve, palette and
//! outline hooks behave as the sprite effects do; named params take per-
//! instance values; hooks draw on the soft route; and a vertex body moves the
//! quad.
//!
//! Every scene is a top-down orthographic view at 16 pixels to a unit with
//! square additive particles on black. Skips cleanly when no GPU adapter is
//! available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::{Camera, Colour};
use viewport_lib_types::plugin::material::{MaterialPlugin, ParamDesc};

use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, FragmentHook, InlineTexture, ParticleBlend, ParticleItem,
    ParticleItems, ParticlePlugin, SpawnRate, SpawnShape, TextureFilter, VelocityDist,
};

const SIZE: u32 = 64;
const PX_PER_UNIT: f32 = 16.0;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("particles-hook-parity-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// viewport-lib's test sprite material: paints in its `tint`, keeping alpha.
struct Solid;

impl MaterialPlugin for Solid {
    fn name(&self) -> &str {
        "solid"
    }

    fn wgsl_body(&self) -> String {
        "fn shade_sprite(f: SpriteFragment) -> vec4<f32> {\n    \
             return vec4<f32>(params.tint, f.texel.a * f.colour.a);\n}\n"
            .to_string()
    }

    fn param_layout(&self) -> Vec<ParamDesc> {
        vec![ParamDesc::vec3("tint", [1.0, 0.0, 1.0])]
    }
}

/// viewport-lib's test vertex material: moves the top corners right by `shift`.
struct Lean;

impl MaterialPlugin for Lean {
    fn name(&self) -> &str {
        "golden_lean"
    }

    fn wgsl_body(&self) -> String {
        "fn displace_sprite(v: SpriteVertex) -> vec3<f32> {\n    \
             return v.world + vec3<f32>(select(0.0, params.shift, v.corner >= 2u), 0.0, 0.0);\n}\n"
            .to_string()
    }

    fn param_layout(&self) -> Vec<ParamDesc> {
        vec![ParamDesc::f32("shift", 0.0)]
    }
}

/// One still square particle, `size` units across, sampling `texture` (white
/// when `None`).
fn square(hook: FragmentHook, size: f32, texture: Option<InlineTexture>) -> EffectAsset {
    let texture = texture.unwrap_or(InlineTexture {
        width: 1,
        height: 1,
        rgba: vec![255; 4],
    });
    EffectAsset::new("hooked")
        .with_capacity(4)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::Burst { count: 1 },
            lifetime: (1000.0, 1000.0),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size,
        })
        .with_texture_rgba(texture.width, texture.height, texture.rgba)
        .with_texture_filter(TextureFilter::Nearest)
        .with_fragment_hook(hook)
}

/// One instance to draw: the effect's index, its position and its hook param
/// overrides.
type Instance = (usize, [f32; 3], Vec<(&'static str, [f32; 4])>);

/// Render `items` and return the image.
fn render(effects: Vec<EffectAsset>, items: &[Instance]) -> Option<Vec<u8>> {
    let (device, queue) = headless_device()?;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let ids: Vec<EffectId> = effects
        .into_iter()
        .map(|e| plugin.add_effect(&device, e))
        .collect();
    renderer
        .with_item_type_plugin(&device, Box::new(plugin))
        .expect("register item type plugin");
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-hook-parity-target"),
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
    cam.set_ortho_height(SIZE as f32 / PX_PER_UNIT);
    for _ in 0..3 {
        let mut frame = FrameData::default();
        frame.camera.render_camera = RenderCamera::from_camera(&cam);
        frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
        frame.viewport.show_grid = false;
        frame.viewport.show_axes_indicator = false;
        frame.viewport.background_colour = Some(Colour::linear_rgb(0.0, 0.0, 0.0));
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
        let mut list = ParticleItems::new().with_dt(1.0 / 60.0);
        for (k, (effect, at, params)) in items.iter().enumerate() {
            let mut item = ParticleItem::new(ids[*effect]).with_key(k as u64).at(*at);
            item.settings.cast_shadows = false;
            for (name, value) in params {
                item = item.with_hook_param(name, *value);
            }
            list.push(item);
        }
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, list);
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }
    Some(read_back(&device, &queue, &target))
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let bytes = (SIZE * SIZE * 4) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hook-parity-readback"),
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

/// RGB at world `(x, y)`.
fn at(pixels: &[u8], x: f32, y: f32) -> [i32; 3] {
    let px = (SIZE as f32 / 2.0 + x * PX_PER_UNIT) as u32;
    let py = (SIZE as f32 / 2.0 - y * PX_PER_UNIT) as u32;
    let i = ((py * SIZE + px) * 4) as usize;
    [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32]
}

fn dark(c: [i32; 3]) -> bool {
    c.iter().all(|&v| v < 20)
}

#[test]
fn a_sprite_material_body_draws_a_particle() {
    let Some(pixels) = render(
        vec![square(FragmentHook::from_material(&Solid), 1.0, None)],
        &[(0, [0.0, 0.0, 0.0], vec![])],
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let c = at(&pixels, 0.0, 0.0);
    assert!(c[0] > 150 && c[2] > 150 && c[1] < 40, "the default tint is magenta, got {c:?}");

    // The same body with the tint set per instance.
    let green = render(
        vec![square(FragmentHook::from_material(&Solid), 1.0, None)],
        &[(0, [0.0, 0.0, 0.0], vec![("tint", [0.0, 1.0, 0.0, 0.0])])],
    )
    .unwrap();
    let g = at(&green, 0.0, 0.0);
    assert!(g[1] > 150 && g[0] < 40 && g[2] < 40, "the instance tint is green, got {g:?}");
}

#[test]
fn a_sprite_vertex_body_moves_the_quad() {
    // A one-unit square; the top corners lean 1 unit right, so the top edge's
    // left end moves off the quad's original left side.
    let Some(still) = render(
        vec![square(FragmentHook::from_material(&Lean), 1.0, None)],
        &[(0, [0.0, 0.0, 0.0], vec![])],
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let leaned = render(
        vec![square(FragmentHook::from_material(&Lean), 1.0, None)],
        &[(0, [0.0, 0.0, 0.0], vec![("shift", [1.0, 0.0, 0.0, 0.0])])],
    )
    .unwrap();
    // Near the top-left corner.
    assert!(!dark(at(&still, -0.4, 0.4)), "the unmoved quad covers its top-left");
    assert!(dark(at(&leaned, -0.4, 0.4)), "the leaned quad leaves its top-left");
    assert!(!dark(at(&leaned, 0.9, 0.4)), "the leaned quad reaches right at the top");
    assert!(dark(at(&still, 0.9, 0.4)));
}

#[test]
fn flash_mixes_towards_its_colour_per_instance() {
    let red = InlineTexture {
        width: 1,
        height: 1,
        rgba: vec![255, 0, 0, 255],
    };
    let Some(pixels) = render(
        vec![square(FragmentHook::flash([1.0, 1.0, 1.0], 0.0), 1.0, Some(red))],
        &[
            (0, [-1.0, 0.0, 0.0], vec![]),
            (0, [1.0, 0.0, 0.0], vec![("flash", [1.0, 0.0, 0.0, 0.0])]),
        ],
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let plain = at(&pixels, -1.0, 0.0);
    let flashed = at(&pixels, 1.0, 0.0);
    assert!(plain[0] > 150 && plain[1] < 40, "no flash keeps the red texel: {plain:?}");
    assert!(flashed.iter().all(|&c| c > 150), "a full flash is white: {flashed:?}");
}

#[test]
fn dissolve_cuts_below_its_threshold() {
    let hook = || FragmentHook::dissolve(0.0, [1.0, 0.5, 0.0], 0.0, 1.0, None);
    let Some(pixels) = render(
        vec![square(hook(), 1.0, None)],
        &[
            (0, [-1.0, 0.0, 0.0], vec![]),
            (0, [1.0, 0.0, 0.0], vec![("dissolve", [1.0, 0.0, 0.0, 0.0])]),
        ],
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    assert!(!dark(at(&pixels, -1.0, 0.0)), "a threshold of 0 keeps the particle");
    assert!(dark(at(&pixels, 1.0, 0.0)), "a threshold of 1 cuts all of it");

    // A noise texture of one grey value, 0.5 once its sRGB bytes are decoded:
    // inside the band from the threshold 0.4 to 0.6, so it takes the edge
    // colour.
    let grey = InlineTexture {
        width: 1,
        height: 1,
        rgba: vec![188, 188, 188, 255],
    };
    let edged = render(
        vec![square(
            FragmentHook::dissolve(0.4, [1.0, 0.0, 0.0], 0.2, 1.0, Some(grey)),
            1.0,
            None,
        )],
        &[(0, [0.0, 0.0, 0.0], vec![])],
    )
    .unwrap();
    let c = at(&edged, 0.0, 0.0);
    assert!(c[0] > 150 && c[1] < 40, "noise inside the edge band takes the edge colour: {c:?}");
}

#[test]
fn palette_swap_reads_the_row_per_instance() {
    // A two-column, two-row palette: row 0 red then green, row 1 blue then
    // white. A black texel picks column 0.
    let palette = InlineTexture {
        width: 2,
        height: 2,
        rgba: vec![
            255, 0, 0, 255, 0, 255, 0, 255, //
            0, 0, 255, 255, 255, 255, 255, 255,
        ],
    };
    let black = InlineTexture {
        width: 1,
        height: 1,
        rgba: vec![0, 0, 0, 255],
    };
    let Some(pixels) = render(
        vec![square(
            FragmentHook::palette_swap(palette, 0, false),
            1.0,
            Some(black),
        )],
        &[
            (0, [-1.0, 0.0, 0.0], vec![]),
            (0, [1.0, 0.0, 0.0], vec![("palette_row", [1.0, 0.0, 0.0, 0.0])]),
        ],
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let row0 = at(&pixels, -1.0, 0.0);
    let row1 = at(&pixels, 1.0, 0.0);
    assert!(row0[0] > 150 && row0[2] < 40, "row 0, column 0 is red: {row0:?}");
    assert!(row1[2] > 150 && row1[0] < 40, "row 1, column 0 is blue: {row1:?}");
}

#[test]
fn outline_rings_the_opaque_texels() {
    // An 8x8 texture whose middle 4x4 is opaque, on a 2-unit quad: the opaque
    // part is the middle unit, and a 4-pixel outline rings it inside the quad.
    let mut rgba = vec![0u8; 8 * 8 * 4];
    for y in 2..6 {
        for x in 2..6 {
            let i = (y * 8 + x) * 4;
            rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
        }
    }
    let Some(pixels) = render(
        vec![square(
            FragmentHook::outline([1.0, 0.0, 0.0, 1.0], 4.0),
            2.0,
            Some(InlineTexture {
                width: 8,
                height: 8,
                rgba,
            }),
        )],
        &[(0, [0.0, 0.0, 0.0], vec![])],
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let inside = at(&pixels, 0.0, 0.0);
    let ring = at(&pixels, 0.62, 0.0);
    let beyond = at(&pixels, 0.95, 0.0);
    assert!(inside.iter().all(|&c| c > 150), "the opaque middle stays white: {inside:?}");
    assert!(ring[0] > 150 && ring[1] < 40, "the outline is red just outside it: {ring:?}");
    assert!(dark(beyond), "past the outline width it is clear: {beyond:?}");
}

#[test]
fn hooks_draw_on_the_soft_route() {
    let Some(pixels) = render(
        vec![square(FragmentHook::from_material(&Solid), 1.0, None).with_soft(0.5)],
        &[(0, [0.0, 0.0, 0.0], vec![("tint", [0.0, 1.0, 0.0, 0.0])])],
    ) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let c = at(&pixels, 0.0, 0.0);
    assert!(c[1] > 150 && c[0] < 40, "a soft hooked billboard draws its hook's colour: {c:?}");
}
