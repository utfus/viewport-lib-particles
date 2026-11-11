use super::*;
use viewport_lib_types::material::TextureFilter;
use viewport_lib_types::plugin::material::{MaterialPlugin, ParamKind, ParamValue};
use viewport_lib_types::render_item::PainterPlacement;

/// Fixed number of texture slots a custom fragment hook can bind. The group-3
/// hook layout always declares this many (unused slots bind the shared 1x1
/// white default), so every hooked billboard pipeline shares one layout and
/// pipelines cache purely by composed-source hash.
pub(crate) const HOOK_TEXTURES: usize = 4;

/// Number of `vec4` lanes in a fragment hook's params uniform. Lane 0 is engine-
/// reserved (`.w` carries `time`); lanes 1.. are the effect's own params.
pub(crate) const HOOK_PARAM_LANES: usize = 8;

/// An effect's texture carried as CPU pixels on the asset, for hosts that cannot
/// call [`ParticlePlugin::upload_texture`](crate::ParticlePlugin) because they
/// hold no `device`/`queue` and no plugin handle after the renderer takes
/// ownership. When [`EffectAsset::texture_data`] is set (via
/// [`EffectAsset::with_texture_rgba`]) and [`EffectAsset::texture`] is not, the
/// plugin uploads these bytes to a [`ParticleTextureId`] on its next frame, binds
/// it, and drops the CPU copy, so a texture goes through the normal
/// `descriptor -> asset -> replace_effect` flow with no out-of-band upload.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InlineTexture {
    /// Texture width in texels.
    pub width: u32,
    /// Texture height in texels.
    pub height: u32,
    /// `width * height * 4` bytes, row-major RGBA8 (sRGB), no row padding: the
    /// same layout [`ParticlePlugin::upload_texture`](crate::ParticlePlugin) takes.
    pub rgba: Vec<u8>,
}

/// Custom WGSL that colours a billboard particle and can move its corners.
///
/// The contract is viewport-lib's sprite material contract (its
/// `SpriteMaterials` and `sprite_contract.wgsl`), so one body draws a sprite
/// and a particle, and [`FragmentHook::from_material`] takes a sprite material
/// plugin as it is. This is the description both sides follow.
///
/// # Functions
///
/// [`body`](Self::body) defines at least one of these, plus any helpers:
///
/// ```text
/// fn shade_particle(f: ParticleFragment) -> vec4<f32>
/// fn shade_sprite(f: SpriteFragment) -> vec4<f32>
/// fn displace_particle(v: ParticleVertex) -> vec3<f32>
/// fn displace_sprite(v: SpriteVertex) -> vec3<f32>
/// ```
///
/// A shading function returns the particle's straight-alpha linear colour in
/// place of the built-in tint and texture; the plugin premultiplies it, and
/// fades and lights it on the routes that do. A displacing function returns
/// where one corner of the quad goes, in world space. A body with no shading
/// function keeps the built-in colour; one with no displacing function keeps
/// the corners where they are.
///
/// # Inputs
///
/// ```text
/// struct ParticleFragment {
///     uv:      vec2<f32>, // texture UV, the flipbook cell applied
///     texel:   vec4<f32>, // the texture at uv; white with no texture
///     colour:  vec4<f32>, // the particle's tint, with its ramps
///     age:     f32,       // life, 0 at birth to 1 at death
///     age_sec: f32,       // seconds since birth
///     world:   vec3<f32>,
///     time:    f32,       // seconds, from the frame time
/// };
/// ```
///
/// `SpriteFragment` starts with the same seven fields and adds `local_uv` (0
/// to 1 across the quad from its bottom-left), `uv_rect` (the flipbook cell),
/// `flash`, `dissolve` and `palette_row` (always 0 on a particle), and `uv_ddx`
/// / `uv_ddy` (the derivatives of `uv`, for `textureSampleGrad`).
/// `ParticleVertex` and `SpriteVertex` are the same struct: `corner` (0 bottom
/// left, 1 bottom right, 2 top left, 3 top right), `local_uv`, `world` (the
/// corner), `anchor` (the particle position) and `time`.
///
/// # Bindings
///
/// Up to `HOOK_TEXTURES` (4) textures read as `htex0..htex3` through
/// `sprite_samp`, given as CPU bytes on the hook (see [`InlineTexture`]); the
/// particle's own texture is `sprite_tex`. Params are `vec4` lanes in
/// `hook_params.lanes`: lane 0 is reserved, with the time in `.w`, and param
/// `i` is lane `1 + i`, at most 7. Named params are also read as
/// `params.<name>`, swizzled to their type (see [`named_param`](Self::named_param)),
/// and set per instance with
/// [`ParticleItem::with_hook_param`](crate::ParticleItem::with_hook_param).
/// Textures are read in the fragment stage only; params in both.
///
/// # Routes
///
/// Billboards on every billboard pass: additive and the other order-independent
/// blends, OIT, soft, and painter, where a lit placement takes 2D light. Not
/// the mesh or trail routes. Picks use the unmoved, camera-facing quad.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentHook {
    /// A WGSL identifier prefix for the hook's symbols, so a body is portable
    /// across host shaders without colliding.
    pub name: String,
    /// The WGSL body: the functions above plus any helpers. Must not redefine
    /// the template's symbols (`vs`, `fs`, `shade`, `base_colour`, `flip_uv`,
    /// `billboard_world`, `camera`, `particles`, `params`, ...).
    pub body: String,
    /// Extra textures the hook samples (`htex0..`), as inline RGBA8 bytes uploaded
    /// by the plugin. At most `HOOK_TEXTURES`.
    pub textures: Vec<InlineTexture>,
    /// Custom params, one `vec4` lane each, read as `hook_params.lanes[1 + i]`.
    /// At most 7.
    pub params: Vec<[f32; 4]>,
    /// Names for the params, read in the body as fields of a `params` struct
    /// (`params.tint`), as viewport-lib's material bodies read theirs. Entry
    /// `i` names lane `1 + i`. Either empty or one per param. A named param can
    /// be set per instance with
    /// [`ParticleItem::with_hook_param`](crate::ParticleItem::with_hook_param).
    #[cfg_attr(feature = "serde", serde(default))]
    pub param_names: Vec<HookParamDecl>,
}

/// The name and type of one [`FragmentHook`] param lane. The body reads it as
/// `params.<name>`, the lane swizzled to the type.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HookParamDecl {
    /// The field name, a WGSL identifier.
    pub name: String,
    /// The field's WGSL type.
    pub kind: ParamKind,
}

impl FragmentHook {
    /// A hook with the given symbol `name` and WGSL `body`.
    pub fn new(name: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            body: body.into(),
            textures: Vec::new(),
            params: Vec::new(),
            param_names: Vec::new(),
        }
    }

    /// A hook from a viewport-lib material plugin's body and parameters, so
    /// the body written for a sprite material draws a particle too. Its
    /// parameters become named lanes at their defaults. Textures are not part
    /// of a material plugin; add them with [`texture`](Self::texture).
    pub fn from_material(plugin: &dyn MaterialPlugin) -> Self {
        let mut hook = Self::new(plugin.name(), plugin.wgsl_body());
        for p in plugin.param_layout() {
            let lane = match p.default {
                ParamValue::F32(v) => [v, 0.0, 0.0, 0.0],
                ParamValue::Vec2(v) => [v[0], v[1], 0.0, 0.0],
                ParamValue::Vec3(v) => [v[0], v[1], v[2], 0.0],
                ParamValue::Vec4(v) => v,
                ParamValue::Colour(c) => c.into(),
            };
            hook = hook.named_param(p.name, p.kind, lane);
        }
        hook
    }

    /// Add a texture input (`htex0`, `htex1`, ...), as inline RGBA8 bytes.
    pub fn texture(mut self, width: u32, height: u32, rgba: Vec<u8>) -> Self {
        self.textures.push(InlineTexture {
            width,
            height,
            rgba,
        });
        self
    }

    /// Add a params lane (`hook_params.lanes[1 + i]`).
    pub fn param(mut self, lane: [f32; 4]) -> Self {
        self.params.push(lane);
        self
    }

    /// Add a named params lane, read in the body as `params.<name>` and set
    /// per instance by name. Mixing with unnamed [`param`](Self::param) lanes
    /// is refused by [`validate`](Self::validate).
    pub fn named_param(mut self, name: impl Into<String>, kind: ParamKind, lane: [f32; 4]) -> Self {
        self.params.push(lane);
        self.param_names.push(HookParamDecl {
            name: name.into(),
            kind,
        });
        self
    }

    /// The lane of the param called `name`, counting from 1.
    pub fn param_lane(&self, name: &str) -> Option<usize> {
        self.param_names.iter().position(|d| d.name == name).map(|i| i + 1)
    }

    /// Structural check before compilation: the `name` is a valid WGSL
    /// identifier, the body defines one of the hook functions, the params are
    /// all named or none, and the texture / param counts are within the fixed
    /// caps. A WGSL *compile* error in the body still
    /// shows up at pipeline build, as with the codegen path.
    pub fn validate(&self) -> Result<(), String> {
        if !is_identifier(&self.name) {
            return Err(format!("hook name {:?} is not a valid WGSL identifier", self.name));
        }
        let defines = |f: &str| self.body.contains(&format!("fn {f}("));
        if !["shade_particle", "shade_sprite", "displace_particle", "displace_sprite"]
            .iter()
            .any(|f| defines(f))
        {
            return Err("hook body must define `shade_particle`, `shade_sprite`, \
                 `displace_particle` or `displace_sprite`"
                .to_string());
        }
        if !self.param_names.is_empty() && self.param_names.len() != self.params.len() {
            return Err(format!(
                "hook names {} of its {} params; name all of them or none",
                self.param_names.len(),
                self.params.len()
            ));
        }
        for (i, d) in self.param_names.iter().enumerate() {
            if !is_identifier(&d.name) || self.param_names[..i].iter().any(|e| e.name == d.name) {
                return Err(format!("hook param name {:?} is not a unique WGSL identifier", d.name));
            }
        }
        if self.textures.len() > HOOK_TEXTURES {
            return Err(format!(
                "hook has {} textures, max {}",
                self.textures.len(),
                HOOK_TEXTURES
            ));
        }
        // Lane 0 is engine-reserved, so effect params occupy lanes 1..LANES.
        if self.params.len() > HOOK_PARAM_LANES - 1 {
            return Err(format!(
                "hook has {} param lanes, max {}",
                self.params.len(),
                HOOK_PARAM_LANES - 1
            ));
        }
        Ok(())
    }
}

fn is_identifier(s: &str) -> bool {
    s.chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// When a parent effect produces spawn events for its child.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EventCondition {
    /// One trigger per live parent particle each simulation step. Suited to
    /// continuous streams: a rocket laying down a smoke trail.
    EveryStep,
    /// One trigger when a parent particle dies (its lifetime crosses zero).
    /// Suited to bursts: a firework shell breaking into sparks at its apex.
    /// A particle killed by [`CollisionResponse::Kill`] triggers it too.
    OnDeath,
    /// One trigger the step a parent particle hits a collider (see
    /// [`Collide2d`]), whatever the response: a splash where rain lands.
    OnContact,
}

/// Links a parent effect to a child effect so parent particles spawn child
/// particles on the GPU.
///
/// The parent's simulate pass appends spawn events (the parent particle's
/// position and velocity) into a shared buffer; the child's emit pass consumes
/// them, seeding new particles at the event position. The child keeps its own
/// spawn shape, velocity distribution, lifetime, colour, and size, so it shapes
/// the burst; set the child emitter's rate to zero for a purely event-driven
/// child.
///
/// Both effects must be fixed-function (no [`EffectProgram`]). Register the child
/// before the parent so the parent can name it. The plugin orders the passes
/// (parent simulate before child emit) automatically.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubEmitter {
    /// The child effect that consumes this parent's events.
    pub child: EffectId,
    /// When the parent produces events.
    pub condition: EventCondition,
    /// Child particles spawned per trigger. Bounded by the child's capacity.
    pub count: u32,
    /// Fraction of the parent particle's velocity handed to each child particle,
    /// added on top of the child's own velocity distribution. `0` ignores the
    /// parent velocity; `1` matches it.
    pub inherit_velocity: f32,
}

impl SubEmitter {
    /// A sub-emitter that spawns `count` child particles per trigger under
    /// `condition`, inheriting none of the parent velocity.
    pub fn new(child: EffectId, condition: EventCondition, count: u32) -> Self {
        Self {
            child,
            condition,
            count,
            inherit_velocity: 0.0,
        }
    }

    /// Hand each child particle `fraction` of the parent particle's velocity.
    pub fn with_inherit_velocity(mut self, fraction: f32) -> Self {
        self.inherit_velocity = fraction;
        self
    }
}

/// The authored description of one effect.
///
/// Carries no GPU state. Register it with a
/// [`ParticlePlugin`](crate::ParticlePlugin) to get an [`EffectId`].
#[derive(Clone, Debug)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EffectAsset {
    /// Human-readable name, used in debug labels.
    pub name: String,
    /// Maximum simultaneously-live particles. Memory scales linearly.
    pub capacity: u32,
    /// Blend mode for the draw.
    pub blend: ParticleBlend,
    /// Fixed-function emitter. Ignored when [`program`](Self::program) is set.
    pub emitter: Emitter,
    /// Modifiers applied each simulation step: forces summed into acceleration,
    /// plus drag and kill-region rules. Ignored when [`program`](Self::program)
    /// is set. (Was named `forces`; the serde alias keeps old `.effect` files
    /// loading.)
    #[cfg_attr(feature = "serde", serde(alias = "forces"))]
    pub modifiers: Vec<Modifier>,
    /// Optional codegen program. When present, the effect compiles to per-effect
    /// emit/simulate WGSL from the expression graph instead of using the
    /// fixed-function emitter and forces above.
    pub program: Option<EffectProgram>,
    /// Optional colour/size ramp over lifetime, applied at draw. `None` uses the
    /// shared identity ramp (no change). Works for both the fixed and codegen
    /// paths.
    pub gradient: Option<Gradient>,
    /// How live particles are drawn. Defaults to a round camera-facing
    /// billboard.
    pub render: ParticleRender,
    /// Optional texture sampled by the billboard fragment. `None` draws the
    /// procedural soft round dot. Uploaded via
    /// `ParticlePlugin::upload_texture`. Applies to the billboard routes.
    pub texture: Option<ParticleTextureId>,
    /// Optional inline texture pixels: an alternative to a
    /// pre-uploaded [`texture`](Self::texture) id for hosts that have no
    /// `device`/`queue` at asset-build time. When set and [`texture`](Self::texture)
    /// is `None`, the plugin uploads these bytes on its next frame, fills
    /// [`texture`](Self::texture) with the resulting id, and drops the CPU copy.
    /// Set via [`with_texture_rgba`](Self::with_texture_rgba). Ignored once
    /// [`texture`](Self::texture) is set. See [`InlineTexture`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub texture_data: Option<InlineTexture>,
    /// Optional custom WGSL fragment hook computing the particle's colour, for recovering a shader-graph material's true look.
    /// `None` uses the built-in fragment (`texture_mode` against the single
    /// texture). Billboard route only (additive / OIT). See [`FragmentHook`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub fragment_hook: Option<FragmentHook>,
    /// How [`texture`](Self::texture) modulates the particle colour. Ignored
    /// when no texture is set.
    pub texture_mode: TextureMode,
    /// Optional flipbook animation over [`texture`](Self::texture) treated as an
    /// atlas. `None` samples the whole texture.
    pub flipbook: Option<Flipbook>,
    /// Optional sub-emitter: this effect's particles spawn particles in a child
    /// effect on the GPU. `None` for a leaf effect. Fixed-function only.
    pub sub_emitter: Option<SubEmitter>,
    /// The frame particles simulate in. [`SimulationSpace::Global`] (world space)
    /// is the default; [`SimulationSpace::Local`] makes live particles follow the
    /// emitting instance's translation.
    pub sim_space: SimulationSpace,
    /// Billboard orientation mode. Applies to the
    /// [`ParticleRender::Billboard`] route; ignored by the mesh and trail routes.
    #[cfg_attr(feature = "serde", serde(default))]
    pub orient: Orient,
    /// Size the billboard in screen space: when `true`, a particle
    /// keeps a constant apparent size regardless of distance (the `size` becomes
    /// its world size at unit camera distance). `false` is world-space size.
    /// Billboard route only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub screen_space_size: bool,
    /// Soft-particle fade distance in world units. `Some(d)` fades the
    /// billboard's alpha to zero as it comes within `d` of the opaque scene
    /// surface, removing the hard clip line where a particle intersects geometry.
    /// Soft effects draw in the renderer's read-only-depth pass (via
    /// `ItemTypePlugin::paint_depth_read`), composited over the scene with
    /// alpha blending, so a soft effect is over-blended regardless of its
    /// [`blend`](Self::blend). Billboard route only; `None` disables it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub soft: Option<f32>,
    /// Per-particle billboard roll at spawn, in radians, plus a symmetric random
    /// spread. Each particle spawns at `rotation` plus a uniform value in
    /// `[-rotation_variance, rotation_variance]`. Billboard route only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotation: f32,
    /// Symmetric random spread on the spawn roll, in radians. See
    /// [`rotation`](Self::rotation).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotation_variance: f32,
    /// Constant billboard spin, in radians per second, advanced by particle age.
    /// Billboard route only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotation_rate: f32,
    /// Timed bursts layered on top of the emitter's continuous rate. Each fires
    /// once, `count` particles, at its `time` after the instance starts. Empty by
    /// default. Fixed-function only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub spawn_schedule: Vec<TimedBurst>,
    /// World-space offset added to every spawn position, on top of the spawn
    /// shape and the emitting instance's position. Lets a spawn region sit off the
    /// anchor. Fixed-function only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub spawn_offset: [f32; 3],
    /// Symmetric random spread on each particle's spawn size, in the same world
    /// units as [`Emitter::size`]. Each particle spawns at `size` plus a uniform
    /// value in `[-size_variance, size_variance]`, clamped at zero. Fixed-function
    /// only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub size_variance: f32,
    /// Global simulation-time multiplier for every instance of this effect (maps
    /// Unity's `simulationSpeed`). `1.0` runs at real time; `2.0` runs
    /// twice as fast, `0.5` half. Applied as a `dt` multiplier, so it composes
    /// with the spawn rate, particle ageing, forces, burst-schedule times, and
    /// flipbook fps, unlike pre-scaling `rate`/`lifetime`. A per-instance
    /// [`ParticleItem::time_scale`] multiplies on top of this.
    #[cfg_attr(feature = "serde", serde(default = "one"))]
    pub simulation_speed: f32,
    /// Start a looping effect at steady state instead of empty (maps Unity's
    /// `prewarm`). When `true`, an instance's simulation is fast-forwarded
    /// by roughly one particle lifetime on its first activation (before its first
    /// draw), so ambient set dressing is full the instant it appears rather than
    /// visibly filling in. The warmup duration is derived from the effect's max
    /// particle lifetime; [`ParticleItem::warmup`] overrides it per instance.
    #[cfg_attr(feature = "serde", serde(default))]
    pub prewarm: bool,
    /// Keep the particles on a world plane, for a 2D game. `None` simulates in
    /// 3D. A sub-emitter child with no lock of its own runs under its parent's.
    /// See [`PlaneLock`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub plane_lock: Option<PlaneLock>,
    /// Draw the effect in viewport-lib's painter sequence at this placement,
    /// sorted with sprites, tile layers and other painter content by layer and
    /// key instead of by depth. `None` draws it in the 3D passes. A
    /// [`ParticleItem::painter`](crate::ParticleItem::painter) overrides it per
    /// instance. Compose the placement with
    /// [`painter_key`](crate::painter_key). See [`with_painter`](Self::with_painter).
    #[cfg_attr(feature = "serde", serde(default))]
    pub painter: Option<PainterPlacement>,
    /// Test painter draws against the scene depth, so opaque 3D geometry in
    /// front hides them. `false`, the default and what a 2D game wants, draws
    /// them over whatever is there in painter order.
    #[cfg_attr(feature = "serde", serde(default))]
    pub painter_depth_test: bool,
    /// How the billboard texture is sampled. [`TextureFilter::Nearest`] keeps
    /// pixel art crisp. Billboard routes only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub texture_filter: TextureFilter,
    /// Snap each billboard's centre to a whole device pixel, as viewport-lib's
    /// sprites do, so pixel art does not shimmer as it moves. Billboard routes
    /// only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub pixel_snap: bool,
    /// Size each billboard from its texture instead of the particle size:
    /// `texels_per_unit` texels to a world unit, so the quad is the texture (or
    /// flipbook cell) width and height divided by it. The particle size, size
    /// variance and size ramp are then ignored, so the texel grid stays whole.
    /// With a camera at a whole number of pixels per texel (`Camera2d`'s
    /// integer-scale snap) each texel covers a whole number of pixels.
    /// Billboard routes only; `None` sizes by the particle size.
    #[cfg_attr(feature = "serde", serde(default))]
    pub texels_per_unit: Option<f32>,
    /// Collide the particles with a 2D collider set in the lock plane. Needs
    /// [`plane_lock`](Self::plane_lock). Fixed-function only. See [`Collide2d`].
    #[cfg_attr(feature = "serde", serde(default))]
    pub collide: Option<Collide2d>,
}

/// Serde default for [`EffectAsset::simulation_speed`] (`1.0`).
#[cfg(feature = "serde")]
fn one() -> f32 {
    1.0
}

impl Default for EffectAsset {
    fn default() -> Self {
        Self {
            name: "effect".to_string(),
            capacity: 10_000,
            blend: ParticleBlend::default(),
            emitter: Emitter::default(),
            modifiers: Vec::new(),
            program: None,
            gradient: None,
            render: ParticleRender::default(),
            texture: None,
            texture_data: None,
            fragment_hook: None,
            texture_mode: TextureMode::default(),
            flipbook: None,
            sub_emitter: None,
            sim_space: SimulationSpace::Global,
            orient: Orient::ParallelCameraPlane,
            screen_space_size: false,
            soft: None,
            rotation: 0.0,
            rotation_variance: 0.0,
            rotation_rate: 0.0,
            spawn_schedule: Vec::new(),
            spawn_offset: [0.0, 0.0, 0.0],
            size_variance: 0.0,
            simulation_speed: 1.0,
            prewarm: false,
            plane_lock: None,
            painter: None,
            painter_depth_test: false,
            texture_filter: TextureFilter::Linear,
            pixel_snap: false,
            texels_per_unit: None,
            collide: None,
        }
    }
}

impl EffectAsset {
    /// A named, empty effect with default capacity and emitter.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Set the live-particle capacity.
    pub fn with_capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// Set the blend mode.
    pub fn with_blend(mut self, blend: ParticleBlend) -> Self {
        self.blend = blend;
        self
    }

    /// Replace the emitter.
    pub fn with_emitter(mut self, emitter: Emitter) -> Self {
        self.emitter = emitter;
        self
    }

    /// Append a modifier (a force, drag, or kill region).
    pub fn modifier(mut self, modifier: Modifier) -> Self {
        self.modifiers.push(modifier);
        self
    }

    /// Append a modifier. Kept for the pre-existing name; identical to
    /// [`modifier`](Self::modifier).
    pub fn force(self, force: Modifier) -> Self {
        self.modifier(force)
    }

    /// Replace the whole modifier list.
    pub fn with_modifiers(mut self, modifiers: Vec<Modifier>) -> Self {
        self.modifiers = modifiers;
        self
    }

    /// Attach a codegen program. The effect then compiles its emit/simulate
    /// kernels from the expression graph, ignoring the fixed-function emitter
    /// and forces.
    pub fn with_program(mut self, program: EffectProgram) -> Self {
        self.program = Some(program);
        self
    }

    /// Attach a colour/size ramp over lifetime.
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Set the render route (billboard with optional stretch, or mesh instances).
    pub fn with_render(mut self, render: ParticleRender) -> Self {
        self.render = render;
        self
    }

    /// Sample `texture` in the billboard fragment, modulating colour per
    /// [`TextureMode::Modulate`]. Combine with [`with_texture_mode`](Self::with_texture_mode)
    /// or [`with_flipbook`](Self::with_flipbook).
    pub fn with_texture(mut self, texture: ParticleTextureId) -> Self {
        self.texture = Some(texture);
        self
    }

    /// Sample a texture given as CPU pixels, uploaded by the plugin on its next
    /// frame, for hosts that cannot call
    /// [`ParticlePlugin::upload_texture`](crate::ParticlePlugin) (no `device`/
    /// `queue`, no plugin handle). `rgba` is `width * height * 4` bytes, row-major
    /// RGBA8 (sRGB). Combine with [`with_texture_mode`](Self::with_texture_mode) /
    /// [`with_flipbook`](Self::with_flipbook) as with a pre-uploaded texture.
    /// Ignored if a [`texture`](Self::texture) id is already set. See
    /// [`InlineTexture`].
    pub fn with_texture_rgba(mut self, width: u32, height: u32, rgba: Vec<u8>) -> Self {
        self.texture_data = Some(InlineTexture {
            width,
            height,
            rgba,
        });
        self
    }

    /// Compute the particle's colour from a custom WGSL fragment hook instead of
    /// the built-in `texture_mode`. Billboard route only.
    /// See [`FragmentHook`].
    pub fn with_fragment_hook(mut self, hook: FragmentHook) -> Self {
        self.fragment_hook = Some(hook);
        self
    }

    /// Set how the texture modulates the particle colour.
    pub fn with_texture_mode(mut self, mode: TextureMode) -> Self {
        self.texture_mode = mode;
        self
    }

    /// Animate the texture as a flipbook atlas.
    pub fn with_flipbook(mut self, flipbook: Flipbook) -> Self {
        self.flipbook = Some(flipbook);
        self
    }

    /// Spawn particles in a child effect from this effect's particles. See
    /// [`SubEmitter`]. Fixed-function only; ignored if either effect carries a
    /// program.
    pub fn with_sub_emitter(mut self, sub_emitter: SubEmitter) -> Self {
        self.sub_emitter = Some(sub_emitter);
        self
    }

    /// Set the simulation space. [`SimulationSpace::Local`] makes live particles
    /// follow the emitting instance's translation each frame.
    pub fn with_simulation_space(mut self, space: SimulationSpace) -> Self {
        self.sim_space = space;
        self
    }

    /// Set the billboard orientation mode. Billboard route only.
    pub fn with_orient(mut self, orient: Orient) -> Self {
        self.orient = orient;
        self
    }

    /// Size billboards in screen space so distant particles keep a constant
    /// apparent size. Billboard route only.
    pub fn with_screen_size(mut self, enable: bool) -> Self {
        self.screen_space_size = enable;
        self
    }

    /// Make this a soft-particle effect: fade the billboard alpha to
    /// zero within `distance` world units of the opaque scene surface, so it
    /// dissolves into geometry instead of showing a hard intersection edge.
    /// Draws through the read-only-depth pass with alpha blending. Billboard
    /// route only.
    pub fn with_soft(mut self, distance: f32) -> Self {
        self.soft = Some(distance);
        self
    }

    /// Set per-particle billboard rotation: a spawn roll `angle` (radians) with a
    /// symmetric random `variance`, and a constant spin `rate` (radians per
    /// second) advanced by age. Billboard route only.
    pub fn with_rotation(mut self, angle: f32, variance: f32, rate: f32) -> Self {
        self.rotation = angle;
        self.rotation_variance = variance;
        self.rotation_rate = rate;
        self
    }

    /// Append a timed burst: `count` particles firing once at `time` seconds after
    /// an instance starts, on top of the emitter's continuous rate. Fixed-function
    /// only.
    pub fn add_burst(mut self, count: u32, time: f32) -> Self {
        self.spawn_schedule.push(TimedBurst { count, time });
        self
    }

    /// Replace the timed-burst schedule. See [`add_burst`](Self::add_burst).
    pub fn with_spawn_schedule(mut self, schedule: Vec<TimedBurst>) -> Self {
        self.spawn_schedule = schedule;
        self
    }

    /// Offset every spawn position by `offset` world units, on top of the spawn
    /// shape and the instance position. Fixed-function only.
    pub fn with_spawn_offset(mut self, offset: [f32; 3]) -> Self {
        self.spawn_offset = offset;
        self
    }

    /// Set a symmetric random spread on each particle's spawn size, in the same
    /// units as [`Emitter::size`]. Fixed-function only.
    pub fn with_size_variance(mut self, variance: f32) -> Self {
        self.size_variance = variance;
        self
    }

    /// Set the global simulation-time multiplier for every instance of this
    /// effect. `1.0` is real time; `2.0` runs twice as fast. A
    /// per-instance [`ParticleItem::time_scale`] multiplies on top. Negative
    /// values are clamped to `0` at simulation time.
    pub fn with_simulation_speed(mut self, speed: f32) -> Self {
        self.simulation_speed = speed;
        self
    }

    /// Start instances of this effect at steady state on first activation by
    /// fast-forwarding one particle lifetime, so the effect is full when it
    /// first appears. See [`prewarm`](Self::prewarm).
    pub fn with_prewarm(mut self, prewarm: bool) -> Self {
        self.prewarm = prewarm;
        self
    }

    /// Draw the effect in the painter sequence at `placement`.
    ///
    /// Painter effects draw after the opaque scene and its order-independent
    /// transparency, with depth writes off, in the order the renderer sorts
    /// every plugin's painter content: by layer, then key. Within an effect
    /// particles draw in slot order, with no depth sort. A lit placement takes
    /// the 2D light of its layer's range on the billboard route; an
    /// [`unlit`](PainterPlacement::unlit) one keeps its colour and draws after
    /// the range's lit content. The mesh and trail routes, and billboards with
    /// a [`FragmentHook`], draw in painter order unlit. Painter effects draw on
    /// the HDR path only.
    pub fn with_painter(mut self, placement: PainterPlacement) -> Self {
        self.painter = Some(placement);
        self
    }

    /// Test painter draws against the scene depth. See
    /// [`painter_depth_test`](Self::painter_depth_test).
    pub fn with_painter_depth_test(mut self, depth_test: bool) -> Self {
        self.painter_depth_test = depth_test;
        self
    }

    /// Sample the billboard texture with `filter`. See
    /// [`texture_filter`](Self::texture_filter).
    pub fn with_texture_filter(mut self, filter: TextureFilter) -> Self {
        self.texture_filter = filter;
        self
    }

    /// Snap billboard centres to device pixels. See
    /// [`pixel_snap`](Self::pixel_snap).
    pub fn with_pixel_snap(mut self, snap: bool) -> Self {
        self.pixel_snap = snap;
        self
    }

    /// Size billboards from their texture at `texels_per_unit` texels to a
    /// world unit. See [`texels_per_unit`](Self::texels_per_unit).
    pub fn with_integer_scale(mut self, texels_per_unit: f32) -> Self {
        self.texels_per_unit = Some(texels_per_unit);
        self
    }

    /// Collide the particles with a 2D collider set. See [`Collide2d`].
    pub fn with_collide(mut self, collide: Collide2d) -> Self {
        self.collide = Some(collide);
        self
    }

    /// Keep the particles on `lock`'s plane at its depth. See [`PlaneLock`].
    pub fn with_plane_lock(mut self, lock: PlaneLock) -> Self {
        self.plane_lock = Some(lock);
        self
    }
}

/// RON `.effect` serialisation, behind the `serde` feature. Round-trips the whole
/// authored asset model (emitter, forces, expression [`Module`], gradient, render
/// route, textures, sub-emitter, simulation space).
///
/// Note the GPU-handle fields ([`EffectAsset::texture`],
/// [`ParticleRender::Mesh`], [`SubEmitter::child`]) serialise as their raw ids.
/// Those ids are session-scoped registration handles: a loaded asset only refers
/// to the right resource if the host re-registers meshes/textures/child effects
/// in the same order (or remaps the ids after loading).
#[cfg(feature = "serde")]
impl EffectAsset {
    /// Serialise this asset to a pretty RON string.
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }

    /// Parse an asset from a RON string.
    pub fn from_ron(s: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(s)
    }

    /// Write this asset to a `.effect` (RON) file.
    pub fn save_ron(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let s = self.to_ron().map_err(std::io::Error::other)?;
        std::fs::write(path, s)
    }

    /// Read an asset from a `.effect` (RON) file.
    pub fn load_ron(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_ron(&s).map_err(std::io::Error::other)
    }
}
