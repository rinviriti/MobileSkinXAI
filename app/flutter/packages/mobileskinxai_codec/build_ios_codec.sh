#!/bin/bash
set -euo pipefail
CODEC_ROOT="$(cd "$(dirname "$0")" && pwd)"
command -v xcodebuild >/dev/null
command -v cmake >/dev/null
for CODEC_PLATFORM in iphoneos iphonesimulator; do
  cmake -S "$CODEC_ROOT/src" -B "$CODEC_ROOT/build/$CODEC_PLATFORM" \
    -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT="$CODEC_PLATFORM" \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
    -DCMAKE_BUILD_TYPE=Release -DMSX_IOS_FRAMEWORK=ON
  cmake --build "$CODEC_ROOT/build/$CODEC_PLATFORM" --target mobileskinxai_codec --parallel 4
done
mkdir -p "$CODEC_ROOT/ios/Frameworks"
if [ -e "$CODEC_ROOT/ios/Frameworks/MobileSkinXaiCodec.xcframework" ]; then
  echo 'Existing XCFramework preserved; archive/version it before rebuilding.' >&2
  exit 1
fi
xcodebuild -create-xcframework \
  -framework "$CODEC_ROOT/build/iphoneos/MobileSkinXaiCodec.framework" \
  -framework "$CODEC_ROOT/build/iphonesimulator/MobileSkinXaiCodec.framework" \
  -output "$CODEC_ROOT/ios/Frameworks/MobileSkinXaiCodec.xcframework"
