//! How the renderer reads particle items. Needs the `gpu` feature.

use std::any::Any;

use viewport_lib::plugin_api::{PluginItem, PluginItemCollection};
use viewport_lib::scene::material::ItemSettings;

use super::{ParticleItem, ParticleItems};

/// So a host can submit instances as a plain `Vec<ParticleItem>` through
/// `frame.scene.items_mut::<ParticleItem>()` and never name the type string. The frame's
/// settings (the time step, the culls, the budgets) live on [`ParticleItems`], so a host
/// that wants any of them submits that instead, or as well: the plugin reads both forms and
/// simulates their instances together.
impl PluginItem for ParticleItem {
    const TYPE_NAME: &'static str = crate::ParticlePlugin::TYPE_NAME;

    fn settings(&self) -> &ItemSettings {
        &self.settings
    }
}

impl PluginItemCollection for ParticleItems {
    fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether this frame asks for nothing at all.
    ///
    /// Not just "no instances": the renderer skips a plugin whose collections are all
    /// empty, and this one carries work an instance list does not show. A `dt` advances
    /// whatever is still simulating -- a burst submitted once and left to play out owes no
    /// further submission -- and a command retires or replaces an effect. Reporting empty
    /// on the frame after a burst was fired would freeze it mid-air.
    fn is_empty(&self) -> bool {
        self.items.is_empty() && self.commands.is_empty() && self.dt <= 0.0
    }

    fn item_settings(&self, index: usize) -> &ItemSettings {
        &self.items[index].settings
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
