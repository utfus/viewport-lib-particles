# Changelog

## [0.1.0]

Initial release. A GPU particle and VFX system for `viewport-lib`, built as an
`ItemTypePlugin` so none of the codegen or modifier surface lands in
`viewport-lib`'s compatibility-frozen core.

### Features

- Expression-graph codegen to per-effect emit/simulate WGSL, plus a
  fixed-function emitter + force path for simple effects
- Render routes: billboards, instanced meshes, and ribbon trails
- Gradient LUTs, textures, and flipbook animation
- Forces including drag, point attractors, and curl-noise turbulence
- Order-independent transparency, GPU depth sorting, picking, highlighting, and
  screen-space refraction
- Sub-emitters — particles that spawn particles on the GPU
- Runtime property overrides for tuning effects live
- `showcase` example: an eframe gallery with a live, tweakable demo for every
  feature
