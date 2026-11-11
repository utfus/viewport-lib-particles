# Changelog

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
