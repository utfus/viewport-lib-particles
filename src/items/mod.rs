use viewport_lib_types::material::ItemSettings;
use viewport_lib_types::render_item::PainterPlacement;

use crate::effect::{
    Collider2dSet, Collider2dSetId, EffectAsset, EffectId, Emitter, Gradient, Modifier,
    PropertyValue,
};


/// One live instance of an effect in the scene this frame.
///
/// Carries the emitter origin and per-item settings; the authored behaviour
/// lives in the [`EffectAsset`] the [`EffectId`] points at.
///
/// Each submitted item with a distinct [`key`](Self::key) simulates as its own
/// independent particle set at its own position (many braziers from one brazier
/// effect), so a host holding a stable key can move / hide / retire one instance
/// across frames without disturbing the others. Items sharing an `(effect, key)`
/// collapse to one instance driven by the first; the default key is `0`, so a
/// single instance needs no key. (Sub-emitter parent/child effects stay
/// single-instance: their key is forced to `0`.)
///
/// Per-frame overrides tune the driving instance without re-registering the
/// effect: [`with_emitter`](Self::with_emitter) / [`with_forces`](Self::with_forces)
/// replace the fixed-function emitter and forces for live sliders,
/// [`with_gradient`](Self::with_gradient) re-bakes the lifetime ramp, and
/// [`with_property`](Self::with_property) sets a codegen property's value. They
/// are read from the driving item of each `(effect, key)` instance.
#[derive(Clone, Debug)]
pub struct ParticleItem {
    /// Which registered effect to simulate and draw.
    pub effect: EffectId,
    /// Stable per-instance identity within an effect. Items of the same effect
    /// with different keys simulate independently; the same key across frames is
    /// the same live particle set. Defaults to `0`.
    pub key: u64,
    /// World-space emitter origin, added to the effect's spawn shape.
    pub position: [f32; 3],
    /// Shared per-item flags (hidden, selected, pick id, opacity, ...).
    pub settings: ItemSettings,
    /// Per-frame emitter override for the fixed-function path. `None` uses the
    /// effect's registered emitter. Ignored by effects carrying a program.
    pub emitter: Option<Emitter>,
    /// Per-frame modifier override for the fixed-function path. `None` uses the
    /// effect's registered modifiers. Ignored by effects carrying a program.
    pub modifiers: Option<Vec<Modifier>>,
    /// Per-frame lifetime-ramp override. When set, the effect's ramp LUT is
    /// re-baked from these keys this frame. `None` keeps the registered gradient.
    pub gradient: Option<Gradient>,
    /// Per-frame property values, applied over the program's declared defaults by
    /// name. Empty leaves every property at its default.
    pub properties: Vec<(&'static str, PropertyValue)>,
    /// When `true`, this is a fire-and-forget transient burst: submit it once at
    /// a world point and the plugin plays it in a pooled transient slot, then
    /// self-retires it when its particles have aged out, so there is no need to
    /// keep submitting it. See [`burst`](Self::burst). Ignored for sub-emitter effects.
    pub one_shot: bool,
    /// Optional host RNG seed for deterministic spawning. When set,
    /// this instance's per-frame emit seed is a fixed function of the seed, the
    /// `(effect, key)`, and the instance's own simulation-frame counter, so a
    /// fixed seed + `dt` sequence reproduces the same seed stream run to run
    /// (preview vs runtime, captures, golden tests). `None` derives the seed from
    /// the global frame counter (not reproducible across runs). See
    /// [`with_seed`](Self::with_seed). GPU atomic spawn order still limits exact
    /// particle-set reproducibility.
    pub seed: Option<u64>,
    /// Per-instance simulation-time multiplier, applied on top of the
    /// effect's [`EffectAsset::simulation_speed`]. `1.0` runs at real time; `2.0`
    /// runs this placement twice as fast, `0.0` freezes it (like
    /// [`ItemSettings::hidden`] pausing, but still drawn). Scales the `dt` every
    /// particle of this instance sees, so it composes with spawn rate, ageing,
    /// forces, and burst times. See [`with_time_scale`](Self::with_time_scale).
    pub time_scale: f32,
    /// Per-instance prewarm override in seconds. `Some(secs)`
    /// fast-forwards this instance by `secs` on its first activation so it
    /// presents at steady state; `None` uses the effect's
    /// [`EffectAsset::prewarm`] (one particle lifetime when set, no warmup
    /// otherwise). See [`with_warmup`](Self::with_warmup).
    pub warmup: Option<f32>,
    /// This instance's place in the painter sequence, overriding the effect's
    /// [`EffectAsset::painter`]. A y-sorted effect sets it every frame from the
    /// instance's own sort value. `None` keeps the effect's.
    pub painter: Option<PainterPlacement>,
}

impl ParticleItem {
    /// A visible instance of `effect` at the world origin.
    pub fn new(effect: EffectId) -> Self {
        Self {
            effect,
            key: 0,
            position: [0.0, 0.0, 0.0],
            settings: ItemSettings::default(),
            emitter: None,
            modifiers: None,
            gradient: None,
            properties: Vec::new(),
            one_shot: false,
            seed: None,
            time_scale: 1.0,
            warmup: None,
            painter: None,
        }
    }

    /// Place the emitter at `position`.
    pub fn at(mut self, position: [f32; 3]) -> Self {
        self.position = position;
        self
    }

    /// Mark this as a fire-and-forget transient burst: submit it once (e.g. a
    /// muzzle flash on a shot) and the plugin plays it in a pooled transient
    /// slot and self-retires it when its particles die. No need to keep
    /// submitting it or hold a key. Best paired with `SpawnRate::Burst`. Ignored
    /// for sub-emitter effects.
    pub fn burst(mut self) -> Self {
        self.one_shot = true;
        self
    }

    /// Give this instance a stable per-effect identity. Different keys simulate
    /// as independent particle sets; the same key across frames is the same
    /// live set (so the host can move / hide / retire one of many). No effect on
    /// sub-emitter parent/child effects, which stay single-instance.
    pub fn with_key(mut self, key: u64) -> Self {
        self.key = key;
        self
    }

    /// Place this instance in the painter sequence, overriding the effect's
    /// placement. See [`EffectAsset::with_painter`].
    pub fn with_painter(mut self, placement: PainterPlacement) -> Self {
        self.painter = Some(placement);
        self
    }

    /// Override the fixed-function emitter for this frame (live tuning of rate,
    /// lifetime, shape, velocity, colour, size). No effect on program effects.
    pub fn with_emitter(mut self, emitter: Emitter) -> Self {
        self.emitter = Some(emitter);
        self
    }

    /// Override the fixed-function modifiers for this frame. No effect on program
    /// effects.
    pub fn with_modifiers(mut self, modifiers: Vec<Modifier>) -> Self {
        self.modifiers = Some(modifiers);
        self
    }

    /// Override the fixed-function modifiers for this frame. Kept for the
    /// pre-existing name; identical to [`with_modifiers`](Self::with_modifiers).
    pub fn with_forces(self, forces: Vec<Modifier>) -> Self {
        self.with_modifiers(forces)
    }

    /// Re-bake the lifetime ramp from `gradient` this frame (live gradient-key
    /// editing).
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Set a codegen property's value for this frame.
    pub fn with_property(mut self, name: &'static str, value: PropertyValue) -> Self {
        self.properties.push((name, value));
        self
    }

    /// Set a deterministic host RNG seed for this instance. A fixed
    /// seed + `dt` sequence reproduces the same spawn-seed stream run to run.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set this instance's simulation-time multiplier. Multiplies the
    /// effect's [`EffectAsset::simulation_speed`]. `2.0` runs it twice as fast,
    /// `0.0` freezes it. See [`time_scale`](Self::time_scale).
    pub fn with_time_scale(mut self, scale: f32) -> Self {
        self.time_scale = scale;
        self
    }

    /// Fast-forward this instance by `seconds` on its first activation so it
    /// presents at steady state, overriding the effect's
    /// [`EffectAsset::prewarm`]. See [`warmup`](Self::warmup).
    pub fn with_warmup(mut self, seconds: f32) -> Self {
        self.warmup = Some(seconds);
        self
    }
}

/// A runtime effect-lifecycle command, queued on [`ParticleItems`] and
/// applied by the plugin in `prepare` before it simulates. This is the delivery
/// path for hot-reload since the renderer owns the plugin after installation, so
/// the host cannot call [`ParticlePlugin::replace_effect`] / `remove_effect`
/// directly. (Registering a *new* effect still needs
/// [`ParticlePlugin::add_effect`] at startup, which returns the [`EffectId`].)
#[derive(Clone, Debug)]
pub(crate) enum EffectCommand {
    Replace(EffectId, Box<EffectAsset>),
    Remove(EffectId),
    ReplaceColliders(Collider2dSetId, Box<Collider2dSet>),
}

/// The per-frame collection of live effect instances.
///
/// Submit via
/// [`SceneFrame::submit_plugin_items`](viewport_lib::renderer::SceneFrame::submit_plugin_items)
/// under [`ParticlePlugin::TYPE_NAME`]. Set [`dt`](Self::dt) to the frame delta
/// so the simulation advances at the right rate.
#[derive(Clone, Debug)]
pub struct ParticleItems {
    pub(crate) items: Vec<ParticleItem>,
    /// Queued lifecycle commands applied at the start of `prepare`.
    pub(crate) commands: Vec<EffectCommand>,
    /// Simulation time step in seconds for this frame. Defaults to 1/60.
    pub dt: f32,
    /// Whether to depth-sort non-additive (alpha / premultiplied) effects
    /// back-to-front this frame. Additive effects are order-independent and are
    /// never sorted. Defaults to `true`; set `false` to see the ordering
    /// artefact.
    pub sort_transparent: bool,
    /// Distance cull: instances whose emitter origin is farther than
    /// this from the camera are not emitted, simulated, or drawn, and drop their
    /// live particles (resuming fresh when back in range). `None` disables the
    /// distance cull. Sub-emitter effects are never distance-culled.
    pub cull_distance: Option<f32>,
    /// Frustum cull: when `true`, instances whose (approximate)
    /// bounding sphere is fully outside the camera frustum are culled like the
    /// distance cull. The bound is estimated from the effect (spawn extent +
    /// particle travel), so it is conservative. `false` disables it.
    pub frustum_cull: bool,
    /// Simulation budget: cap on the number of instances simulated
    /// this frame. When more are visible, the nearest are kept and the rest are
    /// frozen (not simulated or drawn; particles retained). Sub-emitter instances
    /// are always kept. `None` is unlimited.
    pub max_sim_instances: Option<usize>,
    /// Per-effect cap on concurrent transient (burst) instances. When
    /// a new burst would exceed it, the oldest transient of that effect is
    /// recycled. `None` uses the default (64).
    pub max_transients: Option<usize>,
    /// Live particle-count readback: when `true`, each simulated instance runs a
    /// small count pass and its live count is read back (a couple of frames late)
    /// for [`ParticlePlugin::instance_stats`] /
    /// [`effect_stats`](ParticlePlugin::effect_stats). Off by default (it adds a
    /// compute pass and a copy per instance); enable it only while an editor is
    /// showing counts. Non-stalling either way.
    pub stats_readback: bool,
}

impl Default for ParticleItems {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            commands: Vec::new(),
            dt: 1.0 / 60.0,
            sort_transparent: true,
            cull_distance: None,
            frustum_cull: false,
            max_sim_instances: None,
            max_transients: None,
            stats_readback: false,
        }
    }
}

impl ParticleItems {
    /// An empty collection with the default 1/60 s time step.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the simulation time step for this frame.
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Enable or disable back-to-front depth sorting of non-additive effects for
    /// this frame.
    pub fn with_sort_transparent(mut self, sort: bool) -> Self {
        self.sort_transparent = sort;
        self
    }

    /// Set the distance cull: instances farther than `distance` from the camera
    /// are not simulated or drawn.
    pub fn with_cull_distance(mut self, distance: f32) -> Self {
        self.cull_distance = Some(distance);
        self
    }

    /// Enable frustum culling of off-screen instances this frame.
    pub fn with_frustum_cull(mut self, enable: bool) -> Self {
        self.frustum_cull = enable;
        self
    }

    /// Cap the number of instances simulated this frame; the nearest are kept.
    pub fn with_max_sim_instances(mut self, max: usize) -> Self {
        self.max_sim_instances = Some(max);
        self
    }

    /// Cap concurrent transient (burst) instances per effect.
    pub fn with_max_transients(mut self, max: usize) -> Self {
        self.max_transients = Some(max);
        self
    }

    /// Enable live particle-count readback this frame. Read the result with
    /// [`ParticlePlugin::instance_stats`] /
    /// [`effect_stats`](ParticlePlugin::effect_stats); it lags by about two frames
    /// and never stalls. Enable only while something needs the count.
    pub fn with_stats_readback(mut self, enable: bool) -> Self {
        self.stats_readback = enable;
        self
    }

    /// Append an instance.
    pub fn push(&mut self, item: ParticleItem) {
        self.items.push(item);
    }

    /// The same collection with `items` appended, for folding in instances that arrived
    /// through the plain `Vec<ParticleItem>` form.
    pub fn with_items(mut self, items: &[ParticleItem]) -> Self {
        self.items.extend_from_slice(items);
        self
    }

    /// Queue a hot-reload: replace `id`'s asset when the plugin next runs
    /// `prepare`. Preserves the live particle set for a uniform-only
    /// change; a structural change rebuilds the effect. See
    /// [`ParticlePlugin::replace_effect`].
    pub fn replace_effect(&mut self, id: EffectId, asset: EffectAsset) {
        self.commands
            .push(EffectCommand::Replace(id, Box::new(asset)));
    }

    /// Queue a collider set replacement, applied when the plugin next runs
    /// `prepare`, so effects collide with the new shapes from that frame. See
    /// `ParticlePlugin::replace_collider_set`.
    pub fn replace_collider_set(&mut self, id: Collider2dSetId, set: Collider2dSet) {
        self.commands
            .push(EffectCommand::ReplaceColliders(id, Box::new(set)));
    }

    /// Queue a removal: tombstone `id` and reclaim its GPU state when the plugin
    /// next runs `prepare`. See [`ParticlePlugin::remove_effect`].
    pub fn remove_effect(&mut self, id: EffectId) {
        self.commands.push(EffectCommand::Remove(id));
    }

    /// The submitted instances.
    pub fn items(&self) -> &[ParticleItem] {
        &self.items
    }
}

#[cfg(feature = "gpu")]
mod plugin_item;
