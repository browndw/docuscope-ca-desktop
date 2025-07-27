#!/bin/bash

# Install SimplySign Desktop from Windows Bundle
# Extracts SimplySign Desktop from the 264MB proCertum SmartSign bundle

set -euo pipefail

echo "=== Installing SimplySign Desktop from Windows Bundle ==="

# Bundle download URL and local file
BUNDLE_URL="https://files.certum.eu/software/SimplySignDesktop/Windows/9.3.2.67/SimplySignDesktop-9.3.2.67-win-64-bit.exe"
BUNDLE_FILE="SimplySignDesktop-9.3.2.67-win-64-bit.exe"
EXTRACT_DIR="bundle_extracted"
INSTALL_DIR="/c/Program Files/Certum"

# Download the bundle if not already present
if [ ! -f "$BUNDLE_FILE" ]; then
  echo "📥 Downloading SimplySign Desktop bundle..."
  echo "   URL: $BUNDLE_URL"
  
  if command -v curl >/dev/null 2>&1; then
    curl -L -o "$BUNDLE_FILE" "$BUNDLE_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$BUNDLE_FILE" "$BUNDLE_URL"
  else
    echo "❌ Neither curl nor wget available for download"
    exit 1
  fi
  
  if [ ! -f "$BUNDLE_FILE" ]; then
    echo "❌ Failed to download bundle"
    exit 1
  fi
fi

echo "✅ Bundle ready: $BUNDLE_FILE ($(du -h "$BUNDLE_FILE" | cut -f1))"

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

# The bundle is a self-extracting executable, not a 7z archive
# Try multiple extraction methods

echo "   Attempting silent extraction..."
# Method 1: Try silent extraction flags
if "$BUNDLE_FILE" /S /EXTRACT:"$(cygpath -w "$PWD/$EXTRACT_DIR")" 2>/dev/null; then
    echo "✅ Bundle extracted with /S /EXTRACT"
elif "$BUNDLE_FILE" /SILENT /DIR:"$(cygpath -w "$PWD/$EXTRACT_DIR")" 2>/dev/null; then
    echo "✅ Bundle extracted with /SILENT /DIR"
elif "$BUNDLE_FILE" -o"$EXTRACT_DIR" -y 2>/dev/null; then
    echo "✅ Bundle extracted with -o -y"
else
    echo "   Silent extraction failed, trying 7z extraction..."
    
    # Method 2: Try 7z extraction (sometimes works with self-extractors)
    if 7z x "$BUNDLE_FILE" -o"$EXTRACT_DIR" -y > /dev/null 2>&1; then
        echo "✅ Bundle extracted with 7z"
    else
        echo "   7z extraction failed, trying manual execution..."
        
        # Method 3: Run the executable and hope it extracts
        cd "$EXTRACT_DIR"
        if "../$BUNDLE_FILE" 2>/dev/null; then
            echo "✅ Bundle extracted by execution"
            cd ..
        else
            cd ..
            echo "❌ All extraction methods failed"
            echo "Bundle file info:"
            file "$BUNDLE_FILE" 2>/dev/null || echo "file command not available"
            echo "Trying to list contents with 7z..."
            7z l "$BUNDLE_FILE" 2>/dev/null | head -20 || echo "Cannot list contents"
            exit 1
        fi
    fi
fi

# Check if extraction was successful
if [ ! -d "$EXTRACT_DIR" ] || [ -z "$(ls -A "$EXTRACT_DIR" 2>/dev/null)" ]; then
    echo "❌ Extraction directory is empty"
    exit 1
else
    echo "✅ Bundle extracted successfully"
    EXTRACTED_SIZE=$(du -sh "$EXTRACT_DIR" | cut -f1)
    echo "   Extracted size: $EXTRACTED_SIZE"
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
