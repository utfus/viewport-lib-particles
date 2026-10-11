#!/usr/bin/env bash
# Check that the particle authoring model and item types build without the
# renderer.
#
# With `gpu` off the crate holds only the authoring model and the item types,
# so an editor or an asset pipeline can use them without linking viewport-lib
# or wgpu. One stray import of a renderer type breaks that, and the default
# build never notices. This compiles the crate with `gpu` off, with and
# without serde, and fails if viewport-lib or wgpu appears in that build's
# dependency tree.
set -euo pipefail

cd "$(dirname "$0")/.."

for features in "" serde; do
    cargo check -q --no-default-features --features "$features"
    leaked="$(cargo tree -q --no-default-features --features "$features" -e normal --prefix none | grep -E '^(viewport-lib|wgpu[a-z-]*) v' || true)"
    if [ -n "$leaked" ]; then
        echo "the data-only build of viewport-lib-particles depends on:" >&2
        printf '%s\n' "$leaked" | sort -u >&2
        exit 1
    fi
done
echo "ok: viewport-lib-particles authoring and item types build without viewport-lib or wgpu"
