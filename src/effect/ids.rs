
/// Identifies an [`EffectAsset`] registered with a
/// [`ParticlePlugin`](crate::ParticlePlugin).
///
/// Returned by `ParticlePlugin::add_effect`. Stable for the plugin's lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EffectId(pub(crate) u32);

/// Handle to a mesh uploaded to a [`ParticlePlugin`](crate::ParticlePlugin) for
/// the mesh render route. Returned by `ParticlePlugin::upload_mesh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParticleMeshId(pub(crate) u32);

/// Handle to a texture uploaded to a [`ParticlePlugin`](crate::ParticlePlugin)
/// for the billboard texture / flipbook routes. Returned by
/// `ParticlePlugin::upload_texture`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ParticleTextureId(pub(crate) u32);
