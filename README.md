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

## Status

Skeleton. The public API is in place and the plugin registers and runs against
`viewport-lib`, but codegen and simulation are not implemented yet. See
[`docs/plans/particle-system-plan.md`](docs/plans/particle-system-plan.md) for
the phased build-out.

## Why a separate crate

The heavy, fast-moving surface of a full VFX system (expression codegen, the
modifier library, gradient assets, GPU sorting) does not belong in
`viewport-lib`'s compatibility-frozen core API. Keeping it in a sibling crate
lets it version and iterate independently while plugging into the renderer at
the documented `ItemTypePlugin` seam. This mirrors how `viewport-lib-wind` and
the skinning plugin live outside core.

## Target usage

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

## Run the example

```sh
cargo run --example particles-minimal
```

The skeleton example only exercises the authoring API; it does not open a window
or draw yet.

## Coordinate system

`viewport-lib` is Z-up. Gravity pulls along `-Z`; spawn shapes and forces follow
the same convention.

## License

GPL-3.0-only.
