//! The plugin's tier requirement.
//!
//! The particle plugin simulates in compute shaders and draws through four bind
//! groups, so a renderer capped at the compatibility tier declines it at
//! registration, with the reason, rather than failing at pipeline creation.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::capabilities::Tier;
use viewport_lib::renderer::{RendererConfig, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::plugin_api::ItemTypePlugin;

use viewport_lib_particles::ParticlePlugin;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("particles-tier-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

#[test]
fn the_compatibility_tier_declines_the_plugin_with_a_reason() {
    let Some((device, _queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let mut config = RendererConfig::new(wgpu::TextureFormat::Rgba8UnormSrgb);
    config.max_tier = Some(Tier::Compatibility);
    let mut renderer = ViewportRenderer::with_config(&device, &config);
    let err = renderer
        .with_item_type_plugin(&device, Box::new(ParticlePlugin::new()))
        .expect_err("the compatibility tier has no compute");
    let reason = err.to_string();
    assert!(reason.contains("comput"), "the reason should name compute: {reason}");

    // The full tier takes it.
    let mut full = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    full.with_item_type_plugin(&device, Box::new(ParticlePlugin::new()))
        .expect("the full tier registers the plugin");
    assert_eq!(ParticlePlugin::new().requires().bind_groups, 4);
}
