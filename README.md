# viewport-lib-particles

A GPU particle and VFX system for [`viewport-lib`](https://github.com/grimandgreedy/viewport-lib), built as an
`ItemTypePlugin`. Particles are a scene item category with their own emit and
simulate compute passes; they draw inside the lib's HDR scene pass through the
shared group-0 bindings, so they are lit, shadowed, clipped, pickable, and
outline-selectable alongside the rest of the scene.

Effects are authored from an expression graph (`Module` / `Expr`) and a stack of
modifiers that run at spawn (init), each simulation step (force / update), and
at draw (render). The modifier stack of an `EffectAsset` compiles to WGSL emit
and simulate kernels; all runtime state (particle buffers, compiled pipelines)
lives in this crate, not in `viewport-lib` core.

## Features

- Expression-graph codegen to per-effect emit/simulate WGSL, plus a fixed-function
  emitter + force path for simple effects
- Render routes: billboards, instanced meshes, and ribbon trails
- Gradient LUTs, textures and flipbook animation
- Forces including drag, point attractors, and curl-noise turbulence
- Order-independent transparency, GPU depth sorting, picking, highlighting, and
  screen-space refraction
- Sub-emitters — particles that spawn particles on the GPU
- Runtime property overrides for tuning effects live

## Showcase

The `showcase` example is an eframe gallery with a live, tweakable demo for
every feature — emitters, expression graphs, gradients, render routes,
refraction, interaction, textures, turbulence, and sub-emitters:

```sh
cargo run --example showcase
```

For an authoring-API-only example that opens no window:

```sh
cargo run --example particles-minimal
```

## Usage

```rust,ignore
use viewport_lib_particles::{EffectAsset, ParticleItem, ParticleItems, ParticlePlugin};

let mut plugin = ParticlePlugin::new();
let id = plugin.add_effect(&device, EffectAsset::new("fountain") /* ...modifiers... */);
renderer.with_item_type_plugin(&device, Box::new(plugin));

// Each frame:
let mut items = ParticleItems::new();
items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, Box::new(items));
```

## License

GPL-3.0-only.
