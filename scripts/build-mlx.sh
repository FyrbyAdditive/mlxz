#!/usr/bin/env bash
# Build MLX-dependent targets via xcodebuild.
#
# Why: targets that depend on mlx-swift / mlx-swift-lm cannot be built with `swift build`
# (upstream limitation — see mlx-swift-lm CONTRIBUTING.md; the emit-module phase fails to
# thread transitive C-shim modulemaps, and MLX Metal kernels need the Metal toolchain).
# Pure-logic targets (MLXZCore, MLXZServer, MLXZHub, MLXZUI) build/test fine with `swift build`.
#
# Usage: scripts/build-mlx.sh [scheme]   (default scheme: mlxz-serve)
set -euo pipefail

SCHEME="${1:-mlxz-serve}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Include the MLX-dependent targets (gated out of the default `swift build` graph).
export MLXZ_MLX=1

# SwiftPM clones dependency caches as bare repos, but invokes git with
# `-c safe.bareRepository=explicit`, which then refuses to read them. GIT_CONFIG_PARAMETERS
# is applied last and wins, restoring access. (Needed once the local fork triggers re-resolution.)
export GIT_CONFIG_PARAMETERS="'safe.bareRepository=all'"

# One-time: the Metal toolchain is required to compile MLX's Metal kernels.
if ! xcrun --find metal >/dev/null 2>&1; then
  echo "Metal toolchain not found — downloading (one-time, ~700MB)…"
  xcodebuild -downloadComponent MetalToolchain
fi

exec xcodebuild build \
  -scheme "$SCHEME" \
  -destination 'platform=macOS,arch=arm64' \
  -derivedDataPath .xcode-build \
  -clonedSourcePackagesDirPath .build/xcode-packages \
  -scmProvider system \
  -skipPackagePluginValidation \
  -skipMacroValidation
