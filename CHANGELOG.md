# Changelog

## [Unreleased]

### Features

- **Plane-locked effects** - `EffectAsset::with_plane_lock(PlaneLock { plane, depth })` keeps an effect on the XY or XZ plane at a fixed elevation, with the axes of viewport-lib's `WorldPlane`: spawn positions are projected onto it and velocity and forces lose their normal component, so turbulence and attractors cannot drift particles off it. Works on the fixed-function and expression-graph paths.
- **Sub-emitters keep the plane** - a child with no lock of its own runs under its parent's, and inherited velocity loses its normal component. `ParticlePlugin::try_add_effect` and `try_replace_effect` refuse a child locked to a different plane from its parent with a `PlaneLockMismatch`; `add_effect` panics on it, and `check_sub_emitter_lock` runs the same check without a plugin.
- **Flat spawn shapes** - `SpawnShape::Ring`, `Segment` and `Rectangle`, in plane coordinates; they and `Circle` lie in the lock plane, or XY without a lock. Sub-emitter children now take every spawn shape but the mesh ones, where they took only boxes and spheres.
- **`Orient::InPlane`** - billboards lie in the lock plane, so a rotated 2D camera turns them with the world.

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
