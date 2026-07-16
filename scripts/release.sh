#!/usr/bin/env bash
# Build, sign, notarize, staple, and package the mlxz.app for distribution.
#
# The Xcode project builds AD-HOC (no DEVELOPMENT_TEAM is pinned in project.yml, so a
# headless xcodebuild can't pick a Developer ID). This script re-signs the built artifact
# with your Developer ID + hardened runtime, notarizes it with Apple, staples the ticket,
# and zips it — the reproducible version of the manual v0.1.0 release flow.
#
# Prerequisites (one-time):
#   - Xcode + command line tools (codesign, notarytool, stapler, xcodegen).
#   - A "Developer ID Application" cert in the login keychain (see IDENTITY below).
#   - A stored notary credential profile:
#       xcrun notarytool store-credentials "mlxz-notary" \
#         --apple-id "<your-apple-id>" --team-id "QS865LKS7W"
#
# Produces two notarized, stapled artifacts in dist/:
#   mlxz-v<version>-macos-arm64.dmg   (drag-to-Applications installer)
#   mlxz-v<version>-macos-arm64.zip   (the raw .app)
#
# Usage: scripts/release.sh            # version read from project.yml
#        scripts/release.sh 0.1.1      # or pass an explicit version
set -euo pipefail

IDENTITY="Developer ID Application: Timothy Ellis (QS865LKS7W)"
NOTARY_PROFILE="mlxz-notary"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="${1:-$(grep -m1 MARKETING_VERSION project.yml | sed 's/.*"\(.*\)".*/\1/')}"
APP=".xcode-build/Build/Products/Release/mlxz.app"
DIST="dist"
STAGE="$DIST/dmg-stage"
DMG="$DIST/mlxz-v${VERSION}-macos-arm64.dmg"
ZIP="$DIST/mlxz-v${VERSION}-macos-arm64.zip"

echo "▶︎ Building mlxz.app v${VERSION} (Release)…"
MLXZ_MLX=1 xcodegen generate >/dev/null
MLXZ_MLX=1 GIT_CONFIG_PARAMETERS="'safe.bareRepository=all'" \
  xcodebuild build -scheme mlxz -configuration Release \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath .xcode-build \
  -clonedSourcePackagesDirPath .build/xcode-packages -scmProvider system \
  -skipPackagePluginValidation -skipMacroValidation >/dev/null
echo "  built: $APP"

echo "▶︎ Signing the app (Developer ID + hardened runtime)…"
codesign --force --deep --options runtime --timestamp \
  --entitlements App/mlxz.entitlements --sign "$IDENTITY" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

echo "▶︎ Building the DMG (app + /Applications shortcut)…"
rm -rf "$DIST"; mkdir -p "$STAGE"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "mlxz" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
codesign --force --timestamp --sign "$IDENTITY" "$DMG"

echo "▶︎ Notarizing the DMG (waits for Apple; registers the app's cdhash too)…"
xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait

echo "▶︎ Stapling + verifying…"
xcrun stapler staple "$DMG"
xcrun stapler staple "$APP"           # app cdhash was notarized via the DMG submission
spctl -a -vvv --type install "$DMG"   # expect: accepted, source=Notarized Developer ID
spctl -a -vvv --type execute "$APP"   # expect: accepted, source=Notarized Developer ID

echo "▶︎ Packaging the zip from the stapled app…"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
rm -rf "$STAGE"
echo "✓ Release artifacts:"
echo "    $DMG"
echo "    $ZIP"
echo
echo "Next: create the GitHub release, e.g."
echo "  git tag -a v${VERSION} -m \"mlxz v${VERSION}\" && git push origin v${VERSION}"
echo "  gh release create v${VERSION} \"$DMG\" \"$ZIP\" --title \"mlxz v${VERSION}\" --notes-file …"
