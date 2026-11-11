# Changelog

## [Unreleased]

### Breaking

- **Billboard textures put row 0 at the top** - a billboard's texture, and each flipbook cell, now maps texture row 0 (the first row of `upload_texture` / `with_texture_rgba` bytes, and the top row as images load) to the quad's top edge, as viewport-lib's sprites and tilemaps read them; it was mapped to the bottom. A flipbook's first row of cells is the top row of the sheet. Remove any vertical flip added to compensate.
- **`EffectAsset`, `ParticleItem` and `FragmentHook` gain fields** - `EffectAsset::{plane_lock, painter, painter_depth_test, texture_filter, pixel_snap, texels_per_unit, collide}`, `ParticleItem::{painter, hook_params}` and `FragmentHook::param_names`. `EffectAsset` is `#[non_exhaustive]`; a `ParticleItem` or `FragmentHook` struct literal needs the new fields, or build it with `new`.

### Features

- **Plane-locked effects** - `EffectAsset::with_plane_lock(PlaneLock { plane, depth })` keeps an effect on the XY or XZ plane at a fixed elevation, with the axes of viewport-lib's `WorldPlane`: spawn positions are projected onto it and velocity and forces lose their normal component, so turbulence and attractors cannot drift particles off it. Works on the fixed-function and expression-graph paths.
- **Sub-emitters keep the plane** - a child with no lock of its own runs under its parent's, and inherited velocity loses its normal component. `ParticlePlugin::try_add_effect` and `try_replace_effect` refuse a child locked to a different plane from its parent with a `PlaneLockMismatch`; `add_effect` panics on it, and `check_sub_emitter_lock` runs the same check without a plugin.
- **Flat spawn shapes** - `SpawnShape::Ring`, `Segment` and `Rectangle`, in plane coordinates; they and `Circle` lie in the lock plane, or XY without a lock. Sub-emitter children now take every spawn shape but the mesh ones, where they took only boxes and spheres.
- **`Orient::InPlane`** - billboards lie in the lock plane, so a rotated 2D camera turns them with the world.
- **`ParticleBlend::Multiply` and `Screen`** - multiply darkens what is behind a particle and screen lightens it without passing white, both weighted by alpha and with the same blend states as viewport-lib's sprites. They draw on the billboard, soft, mesh and trail routes and with fragment hooks. Like additive they are order-independent: they draw in the main pass with no depth sort and no OIT, and `ParticleBlend::is_order_independent` says which blends do. A `match` over `ParticleBlend` needs arms for them.
- **Painter's-order particles** - `EffectAsset::with_painter(PainterPlacement)` and `ParticleItem::with_painter` draw an effect in viewport-lib's painter sequence, sorted with sprites, tile layers and painter meshes by layer and key, after the opaque scene and its OIT, with depth writes off and no depth sort. `with_painter_depth_test(true)` tests against scene depth. `PainterPlacement` and `painter_key` are re-exported from viewport-lib-types. Lit billboards take their range's 2D light and an `unlit` placement keeps its colour; mesh and trail routes and hooked billboards draw in order unlit. Picks follow the painter order. Painter effects draw on the HDR path only, and leave the shadow pass.
- **`draws_transparent`** - the plugin now opens the OIT pass only on frames with an over-blended billboard outside the painter sequence and the soft pass.
- **`requires` names what the plugin binds** - four bind groups and up to five storage buffers a stage, on top of compute, so a renderer capped at the compatibility tier declines the plugin at registration with that reason.
- **Showcase 2D page** - tab 25 shows the 2D features on the XY plane among painter sprites, top-down: plane lock, multiply and screen, painter order, pixel art, flash and dissolve hooks, and rain on platforms.
- **Pixel-art billboards** - `EffectAsset::with_texture_filter(TextureFilter::Nearest)` samples the billboard texture without blending, `with_pixel_snap(true)` puts each billboard's centre on a whole device pixel as sprites snap, and `with_integer_scale(texels_per_unit)` sizes billboards from their texture or flipbook cell, ignoring the particle size and size ramp, so a camera at a whole number of pixels per texel draws each texel as a block of whole pixels. Integer-scale billboards can be non-square. `TextureFilter` is re-exported from viewport-lib-types.
- **2D colliders** - `ParticlePlugin::add_collider_set` registers a `Collider2dSet` of `Collider2d` shapes (segments, rectangles, polygons and loose edges, the same shapes as viewport-lib's `Occluder2d`) in plane coordinates, and `EffectAsset::with_collide(Collide2d { set, response })` makes a plane-locked effect bounce off them (`CollisionResponse::Bounce { restitution, friction }`), die on them (`Kill`) or stick to them (`Stick`). Each step tests the particle's path, so fast particles do not pass thin walls. Replace a set with `replace_collider_set` or the queued `ParticleItems::replace_collider_set`; `collider_stats` reports a set's edge and cell counts. Fixed-function only.
- **`EventCondition::OnContact`** - a sub-emitter fires when a parent particle hits a collider, for splashes. A `match` over `EventCondition` needs an arm for it.
- **One hook contract with viewport-lib's sprite materials** - a `FragmentHook` body may define `shade_sprite(f: SpriteFragment)` as well as `shade_particle`, and `displace_particle` or `displace_sprite` to move the quad's corners, so one body draws a sprite and a particle. `FragmentHook::from_material` takes a `MaterialPlugin` written for sprites as it is. The contract is written out on `FragmentHook`.
- **Named hook params** - `FragmentHook::named_param` declares a lane the body reads as `params.<name>`, and `ParticleItem::with_hook_param` sets it per instance. Hook params are rewritten every frame, so the per-instance value changes as it is set.
- **Ready-made hooks** - `FragmentHook::flash`, `dissolve`, `palette_swap` and `outline` do what viewport-lib's sprite effects do, with the same maths. A particle's outline stays inside its quad.
- **Hooks on the soft and painter routes** - hooked billboards draw soft, and in painter order they take the 2D light of their range like plain billboards.

## [0.1.0]

Initial release. A GPU particle and VFX system for `viewport-lib`, built as an `ItemTypePlugin` so its codegen and modifiers live in this crate rather than in `viewport-lib` itself.

### Features

- **Expression-graph effects** - an `EffectProgram` built from `Module` / `Expr` compiles to per-effect emit and simulate WGSL; a fixed-function `Emitter` plus modifiers covers simple effects.
- **Render routes** - billboards, instanced meshes and ribbon trails.
- **Gradients and textures** - colour, size and alpha ramps baked to a LUT, sprite textures and flipbook animation.
- **Forces** - drag, point attractors and curl-noise turbulence.
- **Transparency and interaction** - order-independent transparency, GPU depth sorting, picking, highlighting and screen-space refraction.
- **Sub-emitters** - particles that spawn particles on the GPU.
- **Runtime property overrides** - tune an effect live from the host each frame.
- **`showcase` example** - an eframe gallery with a live, tweakable demo for every feature.
