#!/bin/bash

# Script to manually update README download links for testing
# Usage: ./update-readme-links.sh [version]

VERSION=${1:-"latest"}
REPO="browndw/docuscope-ca-desktop"

echo "Updating README download links for version: $VERSION"

if [ "$VERSION" = "latest" ]; then
    # Get the latest release
    LATEST_TAG=$(curl -s https://api.github.com/repos/$REPO/releases/latest | jq -r .tag_name)
    echo "Latest tag: $LATEST_TAG"
    BASE_URL="https://github.com/$REPO/releases/download/$LATEST_TAG"
    ASSETS=$(curl -s https://api.github.com/repos/$REPO/releases/latest | jq -r '.assets[] | .name')
else
    BASE_URL="https://github.com/$REPO/releases/download/$VERSION"
    ASSETS=$(curl -s https://api.github.com/repos/$REPO/releases/tags/$VERSION | jq -r '.assets[] | .name')
fi

echo "Available assets:"
echo "$ASSETS"

# Extract actual filenames with flexible matching
WINDOWS_FILE=$(echo "$ASSETS" | grep -E "\.(exe|msi)$" | grep -v "\.app" | head -1)
MAC_INTEL_FILE=$(echo "$ASSETS" | grep -E "\.dmg$" | grep -E "(x86_64|intel)" | head -1)
MAC_ARM64_FILE=$(echo "$ASSETS" | grep -E "\.dmg$" | grep -E "(aarch64|arm64)" | head -1)
LINUX_DEB_FILE=$(echo "$ASSETS" | grep -E "\.deb$" | head -1)
LINUX_APPIMAGE_FILE=$(echo "$ASSETS" | grep -E "\.AppImage$" | head -1)

# If we couldn't find specific files, try broader matches
if [ -z "$WINDOWS_FILE" ]; then
    WINDOWS_FILE=$(echo "$ASSETS" | grep -E "\.(exe|msi)$" | head -1)
fi
if [ -z "$MAC_INTEL_FILE" ]; then
    MAC_INTEL_FILE=$(echo "$ASSETS" | grep -E "\.dmg$" | head -1)
fi
if [ -z "$MAC_ARM64_FILE" ] && [ -n "$MAC_INTEL_FILE" ]; then
    MAC_ARM64_FILE=$(echo "$ASSETS" | grep -E "\.dmg$" | tail -1)
fi

# Update the README with actual download links
if [ -n "$WINDOWS_FILE" ]; then
    sed -i.bak "s|^\[windows\]:.*|[windows]: $BASE_URL/$WINDOWS_FILE|" README.md
    echo "Updated Windows link: $BASE_URL/$WINDOWS_FILE"
fi
if [ -n "$MAC_INTEL_FILE" ]; then
    sed -i.bak "s|^\[mac-intel\]:.*|[mac-intel]: $BASE_URL/$MAC_INTEL_FILE|" README.md
    echo "Updated macOS Intel link: $BASE_URL/$MAC_INTEL_FILE"
fi
if [ -n "$MAC_ARM64_FILE" ]; then
    sed -i.bak "s|^\[mac-arm64\]:.*|[mac-arm64]: $BASE_URL/$MAC_ARM64_FILE|" README.md
    echo "Updated macOS ARM64 link: $BASE_URL/$MAC_ARM64_FILE"
fi
if [ -n "$LINUX_DEB_FILE" ]; then
    sed -i.bak "s|^\[linux-deb\]:.*|[linux-deb]: $BASE_URL/$LINUX_DEB_FILE|" README.md
    echo "Updated Linux DEB link: $BASE_URL/$LINUX_DEB_FILE"
fi
if [ -n "$LINUX_APPIMAGE_FILE" ]; then
    sed -i.bak "s|^\[linux-appimage\]:.*|[linux-appimage]: $BASE_URL/$LINUX_APPIMAGE_FILE|" README.md
    echo "Updated Linux AppImage link: $BASE_URL/$LINUX_APPIMAGE_FILE"
fi

# Clean up backup file
rm -f README.md.bak

echo "README.md updated successfully!"
