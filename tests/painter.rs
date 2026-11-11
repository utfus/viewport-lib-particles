//! Painter's-order particles.
//!
//! An effect with a `PainterPlacement` reports batches into viewport-lib's
//! painter sequence and draws there, sorted with painter sprites by layer and
//! key. The scenes here are a top-down orthographic view at one world unit per
//! pixel with a red painter sprite and a large green particle on the centre, so
//! the centre pixel says which drew last.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{
    FrameData, PickBackend, PickId, PickMask, RenderCamera, SurfaceSubmission, ViewportRenderer,
};
use viewport_lib::wgpu;
use viewport_lib::{Camera, Colour};
use viewport_lib_plugins::item_types::lighting_2d::Lighting2dItem;
use viewport_lib_plugins::item_types::sprite::{SpriteItem, SpriteOrder, SpriteSizeMode};

use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, PainterPlacement, ParticleBlend, ParticleItem, ParticleItems,
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
        label: Some("particles-painter-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// One opaque green square particle, `size` world units across, at a placement.
fn square(placement: PainterPlacement, size: f32) -> EffectAsset {
    EffectAsset::new("painter-square")
        .with_capacity(16)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(Emitter {
            rate: SpawnRate::Burst { count: 1 },
            lifetime: (1000.0, 1000.0),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [0.0, 1.0, 0.0, 1.0],
            size,
        })
        .with_texture_rgba(1, 1, vec![255; 4])
        .with_painter(placement)
}

/// A red painter sprite, 30 units across, at `x`.
fn sprite(layer: u32, x: f32) -> SpriteItem {
    let mut s = SpriteItem::default();
    s.positions = vec![[x, 0.0, 0.0]];
    s.default_size = 30.0;
    s.default_colour = Colour::linear_rgb(1.0, 0.0, 0.0);
    s.size_mode = SpriteSizeMode::WorldSpace;
    s.order = SpriteOrder::Painter {
        layer,
        depth_test: false,
    };
    s
}

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: ViewportRenderer,
    target: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Scene {
    /// A scene with the effects `build` registers, and their ids.
    fn new(
        build: impl FnOnce(&mut ParticlePlugin, &wgpu::Device) -> Vec<EffectId>,
    ) -> Option<(Self, Vec<EffectId>)> {
        let (device, queue) = headless_device()?;
        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        viewport_lib_plugins::item_types::install(&mut renderer, &device);
        let mut plugin = ParticlePlugin::new();
        let ids = build(&mut plugin, &device);
        renderer
            .with_item_type_plugin(&device, Box::new(plugin))
            .expect("register item type plugin");
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("particles-painter-target"),
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
        Some((
            Self {
                device,
                queue,
                renderer,
                target,
                view,
            },
            ids,
        ))
    }

    /// Render `frames` frames of `particles` over `sprites` (and a lighting
    /// item, when given), returning the last frame.
    fn render(
        &mut self,
        particles: &[ParticleItem],
        sprites: &[SpriteItem],
        lighting: Option<Lighting2dItem>,
        frames: u32,
    ) -> FrameData {
        let mut last = None;
        for _ in 0..frames {
            let mut cam = Camera::look_at([0.0, 0.0, 10.0], [0.0, 0.0, 0.0], glam::Vec3::Y)
                .orthographic()
                .with_aspect(1.0);
            cam.set_ortho_height(SIZE as f32);
            let mut frame = FrameData::default();
            frame.camera.render_camera = RenderCamera::from_camera(&cam);
            frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
            frame.viewport.show_grid = false;
            frame.viewport.show_axes_indicator = false;
            frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
            *frame.scene.items_mut::<SpriteItem>() = sprites.to_vec();
            if let Some(l) = lighting.clone() {
                *frame.scene.items_mut::<Lighting2dItem>() = vec![l];
            }
            let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
            for p in particles {
                let mut p = p.clone();
                p.settings.cast_shadows = false;
                items.push(p);
            }
            frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
            let cmd = self.renderer.owned().render(&self.device, &self.queue, &self.view, &frame);
            self.queue.submit(std::iter::once(cmd));
            last = Some(frame);
        }
        last.unwrap()
    }

    /// RGB at world `(x, y)`.
    fn pixel(&self, x: f32, y: f32) -> [i32; 3] {
        let pixels = read_back(&self.device, &self.queue, &self.target);
        let px = (SIZE as f32 / 2.0 + x) as u32;
        let py = (SIZE as f32 / 2.0 - y) as u32;
        let i = ((py * SIZE + px) * 4) as usize;
        [pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32]
    }
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let bytes = (SIZE * SIZE * 4) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("painter-readback"),
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

fn green(c: [i32; 3]) -> bool {
    c[1] > 150 && c[0] < 60
}

fn red(c: [i32; 3]) -> bool {
    c[0] > 150 && c[1] < 60
}

#[test]
fn painter_particles_sort_with_sprites_by_layer() {
    let Some((mut scene, ids)) = Scene::new(|p, d| vec![p.add_effect(d, square(PainterPlacement::new(2, 0.0), 20.0))]) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let particle = ParticleItem::new(ids[0]).at([0.0, 0.0, 0.0]);

    // Sprite on layer 1, particle on 2: the particle draws over it.
    let frame = scene.render(&[particle.clone()], &[sprite(1, 0.0)], None, 3);
    let c = scene.pixel(0.0, 0.0);
    assert!(green(c), "a layer-2 particle should cover a layer-1 sprite, got {c:?}");
    let ran = scene.renderer.debug_viewport_passes_ran(0);
    assert!(
        !ran.iter().any(|p| p.contains("oit")),
        "painter particles must not open the OIT pass: {ran:?}"
    );
    let order = scene.renderer.debug_painter_sequence(&frame, true);
    assert_eq!(
        order.iter().map(|(n, ..)| *n).collect::<Vec<_>>(),
        vec!["vpl.sprite", ParticlePlugin::TYPE_NAME],
        "{order:?}"
    );

    // Sprite on layer 3: it covers the particle. The order holds frame to frame.
    scene.render(&[particle.clone()], &[sprite(3, 0.0)], None, 3);
    let c = scene.pixel(0.0, 0.0);
    assert!(red(c), "a layer-3 sprite should cover a layer-2 particle, got {c:?}");
    scene.render(&[particle], &[sprite(3, 0.0)], None, 2);
    assert_eq!(scene.pixel(0.0, 0.0), c, "the order should be stable across frames");
}

#[test]
fn an_instance_placement_moves_it_between_sprites() {
    let Some((mut scene, ids)) = Scene::new(|p, d| vec![p.add_effect(d, square(PainterPlacement::new(2, 0.0), 12.0))]) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Two instances of one effect over two sprites on the effect's layer at
    // key 0: one placed below the sprite, one above.
    let id = ids[0];
    let below = ParticleItem::new(id)
        .with_key(1)
        .at([-14.0, 0.0, 0.0])
        .with_painter(PainterPlacement::new(2, -1.0));
    let above = ParticleItem::new(id)
        .with_key(2)
        .at([14.0, 0.0, 0.0])
        .with_painter(PainterPlacement::new(2, 1.0));
    let mut left = sprite(2, -14.0);
    left.default_size = 20.0;
    let mut right = sprite(2, 14.0);
    right.default_size = 20.0;
    scene.render(&[below, above], &[left, right], None, 3);
    let l = scene.pixel(-14.0, 0.0);
    let r = scene.pixel(14.0, 0.0);
    assert!(red(l), "the instance keyed below should be covered, got {l:?}");
    assert!(green(r), "the instance keyed above should cover, got {r:?}");
}

#[test]
fn lit_placements_take_the_2d_light_and_unlit_ones_keep_their_colour() {
    let Some((mut scene, ids)) = Scene::new(|p, d| {
        vec![
            p.add_effect(d, square(PainterPlacement::new(1, 0.0), 14.0)),
            p.add_effect(d, square(PainterPlacement::new(1, 0.0).with_unlit(true), 14.0)),
        ]
    }) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut dark = Lighting2dItem::default();
    dark.ambient = Colour::linear_rgb(0.1, 0.1, 0.1);
    dark.use_frame_lights = false;
    let particles = [
        ParticleItem::new(ids[0]).with_key(1).at([-14.0, 0.0, 0.0]),
        ParticleItem::new(ids[1]).with_key(2).at([14.0, 0.0, 0.0]),
    ];
    scene.render(&particles, &[], Some(dark), 3);
    let lit = scene.pixel(-14.0, 0.0);
    let unlit = scene.pixel(14.0, 0.0);
    assert!(green(unlit), "an unlit placement should keep its colour, got {unlit:?}");
    assert!(
        lit[1] < unlit[1] - 80,
        "a lit placement should darken under a dim ambient: lit {lit:?}, unlit {unlit:?}"
    );

    // With no lighting item the lit placement draws at full colour too.
    scene.render(&particles, &[], None, 2);
    let bare = scene.pixel(-14.0, 0.0);
    assert!(green(bare), "with no light layer a lit placement keeps its colour, got {bare:?}");
}

#[test]
fn a_painter_particle_over_a_sprite_wins_the_pick() {
    let Some((mut scene, ids)) = Scene::new(|p, d| vec![p.add_effect(d, square(PainterPlacement::new(2, 0.0), 20.0))]) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut particle = ParticleItem::new(ids[0]).at([0.0, 0.0, 0.0]);
    particle.settings.pick_id = PickId(7);
    let mut under = sprite(1, 0.0);
    under.settings.pick_id = PickId(3);
    let frame = scene.render(&[particle], &[under], None, 3);
    let hit = scene
        .renderer
        .pick_object(
            PickBackend::Gpu,
            glam::Vec2::new(SIZE as f32 / 2.0, SIZE as f32 / 2.0),
            &frame,
            &scene.device,
            &scene.queue,
            PickMask::OBJECT,
        )
        .expect("the centre should hit something");
    assert_eq!(hit.id, 7, "the particle drawn over the sprite should win the pick");
}

#[test]
fn trails_and_hooked_billboards_draw_in_painter_order() {
    use viewport_lib_particles::{FragmentHook, ParticleRender};
    let Some((mut scene, ids)) = Scene::new(|p, d| {
        let hook = FragmentHook::new(
            "flat_green",
            "fn shade_particle(f: ParticleFragment) -> vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }",
        );
        vec![
            p.add_effect(
                d,
                square(PainterPlacement::new(2, 0.0), 14.0)
                    .with_fragment_hook(hook),
            ),
            p.add_effect(
                d,
                square(PainterPlacement::new(2, 0.0), 2.0)
                    .with_emitter(Emitter {
                        rate: SpawnRate::PerSecond(60.0),
                        lifetime: (2.0, 2.0),
                        spawn: SpawnShape::Point,
                        velocity: VelocityDist::Fixed([0.0, 20.0, 0.0]),
                        colour: [0.0, 1.0, 0.0, 1.0],
                        size: 2.0,
                    })
                    .with_render(ParticleRender::Trail {
                        width: 3.0,
                        segments: 8,
                    }),
            ),
        ]
    }) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let particles = [
        ParticleItem::new(ids[0]).with_key(1).at([-14.0, 0.0, 0.0]),
        ParticleItem::new(ids[1]).with_key(2).at([14.0, -20.0, 0.0]),
    ];
    // Both over a layer-1 sprite.
    let mut back = sprite(1, 0.0);
    back.default_size = 60.0;
    scene.render(&particles, &[back], None, 30);
    let hooked = scene.pixel(-14.0, 0.0);
    assert!(green(hooked), "a hooked painter billboard should draw over the sprite, got {hooked:?}");
    let trail = (-18..0).any(|y| green(scene.pixel(14.0, y as f32)));
    assert!(trail, "a painter trail should draw over the sprite");
}

#[test]
fn hooked_painter_billboards_take_the_2d_light() {
    use viewport_lib_particles::FragmentHook;
    let hook = || FragmentHook::flash([1.0, 1.0, 1.0], 0.0);
    let Some((mut scene, ids)) = Scene::new(|p, d| {
        vec![
            p.add_effect(d, square(PainterPlacement::new(1, 0.0), 14.0).with_fragment_hook(hook())),
            p.add_effect(
                d,
                square(PainterPlacement::new(1, 0.0).with_unlit(true), 14.0)
                    .with_fragment_hook(hook()),
            ),
        ]
    }) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut dark = Lighting2dItem::default();
    dark.ambient = Colour::linear_rgb(0.1, 0.1, 0.1);
    dark.use_frame_lights = false;
    let particles = [
        ParticleItem::new(ids[0]).with_key(1).at([-14.0, 0.0, 0.0]),
        ParticleItem::new(ids[1]).with_key(2).at([14.0, 0.0, 0.0]),
    ];
    scene.render(&particles, &[], Some(dark), 3);
    let lit = scene.pixel(-14.0, 0.0);
    let unlit = scene.pixel(14.0, 0.0);
    assert!(green(unlit), "an unlit hooked placement keeps its colour, got {unlit:?}");
    assert!(
        lit[1] < unlit[1] - 80,
        "a lit hooked placement darkens under a dim ambient: lit {lit:?}, unlit {unlit:?}"
    );
}
