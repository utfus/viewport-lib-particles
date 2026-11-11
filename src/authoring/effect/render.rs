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

/// How a camera-facing billboard is oriented each frame. Applies to
/// the [`ParticleRender::Billboard`] route, set on the effect via
/// [`EffectAsset::with_orient`](crate::EffectAsset::with_orient). Mirrors
/// Hanabi's `OrientMode`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Orient {
    /// Parallel to the camera's near plane (the cheap default): every billboard
    /// shares the camera's right/up axes, so they stay axis-aligned on screen.
    #[default]
    ParallelCameraPlane,
    /// Face the camera *position*: each billboard builds its own basis pointing
    /// at the eye, a true spherical billboard (correct at wide field of view and
    /// screen edges, slightly costlier).
    FaceCamera,
    /// Orient the quad's local X along the particle's screen-projected velocity
    /// (without the length change [`Billboard::stretch`](ParticleRender::Billboard)
    /// adds), so a textured sprite reads as travelling along its motion.
    AlongVelocity,
    /// A fixed screen-space roll: rotate the camera-plane basis by `radians`
    /// about the view direction. A constant tilt for every particle.
    Fixed {
        /// Roll angle in radians.
        radians: f32,
    },
    /// Lie in the effect's [`PlaneLock`](crate::PlaneLock) plane (XY without a
    /// lock): the quad's axes are the plane's x and y, so a rotated `Camera2d`
    /// turns the particles with the world instead of keeping them upright on
    /// screen. Velocity stretch and [`AlongVelocity`](Self::AlongVelocity)-style
    /// turning stay within the plane.
    InPlane,
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
/// Baked into a small 1D lookup texture the draw shader samples by normalised
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
    /// Colour keys `(t, rgb)`. Multiplies the particle colour. Values may exceed
    /// `1.0` for HDR emissive output that blooms on a bloom-enabled renderer.
    pub colour: Vec<(f32, [f32; 3])>,
    /// Size-scale keys `(t, scale)`. Multiplies the particle size.
    pub size: Vec<(f32, f32)>,
    /// Alpha keys `(t, a)` over lifetime. Multiplies the particle alpha, so an
    /// effect can fade *in* as well as out. The default `[(0, 1), (1, 0)]` is a
    /// linear fade to zero. Billboard route only; the mesh and trail
    /// routes keep the plain lifetime fade.
    #[cfg_attr(feature = "serde", serde(default = "default_alpha_keys"))]
    pub alpha: Vec<(f32, f32)>,
}

/// The default alpha ramp: a linear fade from full at spawn to zero at death.
fn default_alpha_keys() -> Vec<(f32, f32)> {
    vec![(0.0, 1.0), (1.0, 0.0)]
}

impl Default for Gradient {
    fn default() -> Self {
        // Identity colour/size (no change over life); alpha fades out linearly.
        Self {
            colour: vec![(0.0, [1.0, 1.0, 1.0]), (1.0, [1.0, 1.0, 1.0])],
            size: vec![(0.0, 1.0), (1.0, 1.0)],
            alpha: default_alpha_keys(),
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

    /// Set the alpha ramp keys (ascending `t`). Lets the effect fade
    /// in and out over life instead of only fading out. Billboard route only.
    pub fn with_alpha(mut self, keys: Vec<(f32, f32)>) -> Self {
        if !keys.is_empty() {
            self.alpha = keys;
        }
        self
    }

    /// Sample the colour ramp at normalised age `t`.
    pub(crate) fn sample_colour(&self, t: f32) -> [f32; 3] {
        sample_keys3(&self.colour, t)
    }

    /// Sample the size ramp at normalised age `t`.
    pub(crate) fn sample_size(&self, t: f32) -> f32 {
        sample_keys1(&self.size, t)
    }

    /// Sample the alpha ramp at normalised age `t`.
    pub(crate) fn sample_alpha(&self, t: f32) -> f32 {
        sample_keys1(&self.alpha, t)
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
    /// behind a moving emitter). The default.
    #[default]
    Global,
    /// The emitting instance's frame: live particles rigidly follow the
    /// instance's translation each frame, so the whole effect moves with the
    /// emitter. Only the instance translation is tracked (a
    /// [`ParticleItem`](crate::ParticleItem) carries no rotation or scale), so
    /// this is a translation-only local space, applied each simulation step.
    Local,
}

/// GPU blend mode for the particle draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// Multiply: the background is scaled by the particle colour, weighted by
    /// alpha, so a transparent fragment leaves it unchanged. Darkens: smoke,
    /// shadows, stains. Order-independent, so it draws with the additive
    /// effects and needs no sort.
    Multiply,
    /// Screen: the inverse of multiplying the inverses, weighted by alpha.
    /// Lightens without passing white, so a stack of glows does not blow out.
    /// Order-independent, like multiply.
    Screen,
}

impl ParticleBlend {
    /// Whether the blend gives the same result in any draw order: additive,
    /// multiply and screen. These draw in the main pass with no depth sort;
    /// the over blends go through order-independent transparency.
    pub fn is_order_independent(self) -> bool {
        matches!(self, Self::Additive | Self::Multiply | Self::Screen)
    }
}

impl Default for ParticleBlend {
    fn default() -> Self {
        ParticleBlend::Additive
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_independent_blends_skip_oit_and_sorting() {
        for blend in [ParticleBlend::Additive, ParticleBlend::Multiply, ParticleBlend::Screen] {
            assert!(blend.is_order_independent(), "{blend:?}");
        }
        for blend in [ParticleBlend::Alpha, ParticleBlend::Premultiplied] {
            assert!(!blend.is_order_independent(), "{blend:?}");
        }
    }

    #[test]
    fn default_alpha_reproduces_lifetime_fade() {
        // The default alpha ramp is a linear fade from 1 at spawn to 0 at death.
        let g = Gradient::default();
        assert!((g.sample_alpha(0.0) - 1.0).abs() < 1e-6);
        assert!((g.sample_alpha(0.5) - 0.5).abs() < 1e-6);
        assert!((g.sample_alpha(1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn custom_alpha_ramp_fades_in_and_out() {
        // A fade-in-then-out curve peaks in the middle of life.
        let g = Gradient::new().with_alpha(vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]);
        assert!(g.sample_alpha(0.0) < 0.01, "starts transparent");
        assert!((g.sample_alpha(0.5) - 1.0).abs() < 1e-6, "opaque mid-life");
        assert!(g.sample_alpha(1.0) < 0.01, "ends transparent");
    }

    #[test]
    fn colour_ramp_carries_hdr_values() {
        // Colour keys above 1.0 pass through unclamped for HDR bloom.
        let g = Gradient::new().with_colour(vec![(0.0, [4.0, 2.0, 0.5]), (1.0, [0.0, 0.0, 0.0])]);
        let c = g.sample_colour(0.0);
        assert!(c[0] > 3.9 && c[1] > 1.9, "HDR colour is not clamped to 1.0");
    }
}
