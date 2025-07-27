#!/bin/bash

# Install SimplySign Desktop from Windows Bundle
# Extracts SimplySign Desktop from the 264MB proCertum SmartSign bundle

set -euo pipefail

echo "=== Installing SimplySign Desktop from Windows Bundle ==="

# Bundle location
BUNDLE_FILE="../_simplysign_windows_exe/SimplySignDesktop-9.3.2.67-win-64-bit.exe"
EXTRACT_DIR="bundle_extracted"
INSTALL_DIR="/c/Program Files/Certum"

# Check if bundle exists
if [ ! -f "$BUNDLE_FILE" ]; then
  echo "❌ Bundle not found: $BUNDLE_FILE"
  exit 1
fi

echo "✅ Bundle found: $BUNDLE_FILE ($(du -h "$BUNDLE_FILE" | cut -f1))"

# Install 7-Zip if not available
if ! command -v 7z >/dev/null 2>&1; then
  echo "📦 Installing 7-Zip..."
  choco install 7zip -y
  # Add to PATH for current session
  export PATH="$PATH:/c/Program Files/7-Zip"
fi

# Extract the bundle
echo "📂 Extracting Windows bundle..."
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"

# Extract using 7z (the bundle is 7z-based)
if 7z x "$BUNDLE_FILE" -o"$EXTRACT_DIR" -y > /dev/null; then
  echo "✅ Bundle extracted successfully"
else
  echo "❌ Failed to extract bundle"
  exit 1
fi

# Find SimplySign Desktop within the extracted bundle
echo "🔍 Locating SimplySign Desktop in bundle..."

# Search for SimplySignDesktop.exe in the extracted contents
SIMPLYSIGN_LOCATION=$(find "$EXTRACT_DIR" -name "SimplySignDesktop.exe" -type f | head -1)

if [ -z "$SIMPLYSIGN_LOCATION" ]; then
  echo "❌ SimplySignDesktop.exe not found in bundle"
  echo "Bundle contents:"
  find "$EXTRACT_DIR" -name "*.exe" | head -10
  exit 1
fi

echo "✅ Found SimplySign Desktop: $SIMPLYSIGN_LOCATION"

# Get the SimplySign Desktop directory
SIMPLYSIGN_DIR=$(dirname "$SIMPLYSIGN_LOCATION")
echo "📁 SimplySign Desktop directory: $SIMPLYSIGN_DIR"

# Create installation directory
echo "📂 Creating installation directory..."
mkdir -p "$INSTALL_DIR"

# Copy SimplySign Desktop to installation location
echo "📋 Installing SimplySign Desktop..."
cp -r "$SIMPLYSIGN_DIR" "$INSTALL_DIR/SimplySign Desktop"

# Verify installation
INSTALLED_EXE="$INSTALL_DIR/SimplySign Desktop/SimplySignDesktop.exe"
if [ -f "$INSTALLED_EXE" ]; then
  INSTALLED_SIZE=$(du -sh "$INSTALL_DIR/SimplySign Desktop" | cut -f1)
  FILE_COUNT=$(find "$INSTALL_DIR/SimplySign Desktop" -type f | wc -l)
  echo "✅ SimplySign Desktop installed successfully"
  echo "📍 Location: $INSTALL_DIR/SimplySign Desktop"
  echo "📏 Size: $INSTALLED_SIZE ($FILE_COUNT files)"
  
  # List key files
  echo "🔑 Key files installed:"
  find "$INSTALL_DIR/SimplySign Desktop" -name "*.exe" -o -name "*.dll" -o -name "*.config" | head -10 | sed 's/^/   /'
else
  echo "❌ Installation verification failed"
  exit 1
fi

# Cleanup extraction directory
echo "🧹 Cleaning up extraction directory..."
rm -rf "$EXTRACT_DIR"

echo ""
echo "🚀 SimplySign Desktop installation complete!"
echo "📂 Installed to: $INSTALL_DIR/SimplySign Desktop"
echo "🎯 Ready for OAuth2 configuration"
