use super::*;

/// How an effect's texture modulates the per-particle colour in the billboard
/// fragment shader. Mirrors Hanabi's `ImageSampleMapping`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextureMode {
    /// Multiply both colour (rgb) and alpha by the sampled texel.
    #[default]
    Modulate,
    /// Multiply only the colour (rgb); keep the particle's own alpha.
    ModulateRgb,
    /// Take alpha from the texel's red channel (a mask); keep the colour. The
    /// masked-spark idiom.
    ModulateAlphaFromR,
}

/// Flipbook (sprite-sheet) animation over a texture atlas. The texture is a grid
/// of `columns` x `rows` animation frames; the billboard fragment picks a cell
/// per particle and offsets its UVs into it.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Flipbook {
    /// Atlas columns.
    pub columns: u32,
    /// Atlas rows.
    pub rows: u32,
    /// Playback rate. `0` stretches the whole sheet once across each particle's
    /// life (first frame at spawn, last near death); `> 0` loops the sheet at
    /// this many frames per second driven by the particle's elapsed age.
    pub fps: f32,
}

impl Flipbook {
    /// A `columns` x `rows` sheet stretched once across particle life.
    pub fn new(columns: u32, rows: u32) -> Self {
        Self {
            columns,
            rows,
            fps: 0.0,
        }
    }

    /// Loop the sheet at `fps` frames per second instead of stretching it across
    /// life.
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.fps = fps;
        self
    }
}

/// How a mesh-route particle is oriented each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MeshAlign {
    /// Align the mesh's local +Z to the particle's velocity direction.
    Velocity,
    /// A stable per-particle random tumble that advances with age.
    Random,
}

/// How an effect's live particles are drawn.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParticleRender {
    /// Camera-facing billboard. `stretch` (> 0) elongates it along the
    /// screen-projected velocity, scaled by speed, for motion streaks.
    Billboard {
        /// Stretch factor. `0` is a round billboard.
        stretch: f32,
    },
    /// One instance of an uploaded mesh per particle.
    Mesh {
        /// The uploaded mesh to instance.
        mesh: ParticleMeshId,
        /// Per-particle orientation.
        align: MeshAlign,
    },
    /// A camera-facing ribbon swept through each particle's recent position
    /// history, tapering to a point at the tail. A comet / streak that follows
    /// the actual simulated path, not just the instantaneous velocity.
    Trail {
        /// Ribbon half-width at the head, in world units. Tapers to zero.
        width: f32,
        /// History segments to sweep. Clamped to the trail history length.
        segments: u32,
    },
}

impl Default for ParticleRender {
    fn default() -> Self {
        ParticleRender::Billboard { stretch: 0.0 }
    }
}

/// Colour and size ramps applied over a particle's lifetime.
///
/// Baked into a small 1D lookup texture the draw shader samples by normalized
/// age (0 at spawn, 1 at death). The sampled RGB multiplies the particle colour
/// and the sampled scale multiplies its size, so an identity gradient (the
/// default) leaves both unchanged. Keys are `(t in 0..1, value)` pairs in
/// ascending `t`; values are linearly interpolated.
///
/// When using a colour ramp, set the emitter/program colour to white so the
/// multiply yields the ramp colour directly.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Gradient {
    /// Colour keys `(t, rgb)`. Multiplies the particle colour.
    pub colour: Vec<(f32, [f32; 3])>,
    /// Size-scale keys `(t, scale)`. Multiplies the particle size.
    pub size: Vec<(f32, f32)>,
}

impl Default for Gradient {
    fn default() -> Self {
        // Identity: no colour or size change over life.
        Self {
            colour: vec![(0.0, [1.0, 1.0, 1.0]), (1.0, [1.0, 1.0, 1.0])],
            size: vec![(0.0, 1.0), (1.0, 1.0)],
        }
    }
}

impl Gradient {
    /// An identity gradient (no change over life).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the colour ramp keys (ascending `t`).
    pub fn with_colour(mut self, keys: Vec<(f32, [f32; 3])>) -> Self {
        if !keys.is_empty() {
            self.colour = keys;
        }
        self
    }

    /// Set the size-scale ramp keys (ascending `t`).
    pub fn with_size(mut self, keys: Vec<(f32, f32)>) -> Self {
        if !keys.is_empty() {
            self.size = keys;
        }
        self
    }

    /// Sample the colour ramp at normalized age `t`.
    pub(crate) fn sample_colour(&self, t: f32) -> [f32; 3] {
        sample_keys3(&self.colour, t)
    }

    /// Sample the size ramp at normalized age `t`.
    pub(crate) fn sample_size(&self, t: f32) -> f32 {
        sample_keys1(&self.size, t)
    }
}

/// Piecewise-linear sample of ascending `(t, vec3)` keys.
fn sample_keys3(keys: &[(f32, [f32; 3])], t: f32) -> [f32; 3] {
    if keys.is_empty() {
        return [1.0, 1.0, 1.0];
    }
    if t <= keys[0].0 {
        return keys[0].1;
    }
    for w in keys.windows(2) {
        let (t0, a) = w[0];
        let (t1, b) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return [
                a[0] + (b[0] - a[0]) * f,
                a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f,
            ];
        }
    }
    keys[keys.len() - 1].1
}

/// Piecewise-linear sample of ascending `(t, f32)` keys.
fn sample_keys1(keys: &[(f32, f32)], t: f32) -> f32 {
    if keys.is_empty() {
        return 1.0;
    }
    if t <= keys[0].0 {
        return keys[0].1;
    }
    for w in keys.windows(2) {
        let (t0, a) = w[0];
        let (t1, b) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return a + (b - a) * f;
        }
    }
    keys[keys.len() - 1].1
}

/// The frame an effect's particles simulate in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SimulationSpace {
    /// World space: a particle keeps its world position after spawn, so when the
    /// emitting instance moves the already-spawned particles stay put (they trail
    /// behind a moving emitter). The default, and the behaviour of every effect
    /// before this option existed.
    #[default]
    Global,
    /// The emitting instance's frame: live particles rigidly follow the
    /// instance's translation each frame, so the whole effect moves with the
    /// emitter. Only the instance translation is tracked — a
    /// [`ParticleItem`](crate::ParticleItem) carries no rotation or scale — so
    /// this is a translation-only local space, applied each simulation step.
    Local,
}

/// GPU blend mode for the particle draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParticleBlend {
    /// Standard over-blend (premultiplied): particles occlude the background by
    /// their alpha.
    Alpha,
    /// Additive: particle colour adds to the background. Emissive glow look.
    Additive,
    /// Premultiplied over-blend. Same blend state as `Alpha` given the shader
    /// already outputs premultiplied colour; kept distinct for intent.
    Premultiplied,
}

impl Default for ParticleBlend {
    fn default() -> Self {
        ParticleBlend::Additive
    }
}
