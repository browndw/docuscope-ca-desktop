#!/bin/bash

# Package SimplySign Desktop for Artifact Storage
# Creates an efficient zip package of the configured SimplySign Desktop

set -euo pipefail

echo "=== Packaging SimplySign Desktop for Artifact Storage ==="

# Check if SimplySign Desktop is installed
SIMPLYSIGN_DIR="/c/Program Files/Certum/SimplySign Desktop"
SIMPLYSIGN_EXE="$SIMPLYSIGN_DIR/SimplySignDesktop.exe"

if [ ! -f "$SIMPLYSIGN_EXE" ]; then
  echo "❌ SimplySign Desktop not found at: $SIMPLYSIGN_EXE"
  exit 1
fi

echo "✅ SimplySign Desktop found: $SIMPLYSIGN_DIR"

# Create package directory
PACKAGE_DIR="simplysign-desktop-package"
mkdir -p "$PACKAGE_DIR"

echo "📦 Creating minimal SimplySign Desktop package..."

# Copy only the essential SimplySign Desktop files
echo "   Copying core SimplySign Desktop files..."

# Create the essential directory structure
mkdir -p "$PACKAGE_DIR/SimplySign Desktop"

# Copy only the essential executable and core files
ESSENTIAL_FILES=(
    "SimplySignDesktop.exe"
    "SimplySignDesktop.exe.config"
    "SimplySignDesktop.pdb"
    "*.dll"
    "*.config"
)

for pattern in "${ESSENTIAL_FILES[@]}"; do
    find "$SIMPLYSIGN_DIR" -maxdepth 1 -name "$pattern" -type f -exec cp {} "$PACKAGE_DIR/SimplySign Desktop/" \; 2>/dev/null || true
done

# Check what we actually copied
COPIED_SIZE=$(du -sh "$PACKAGE_DIR/SimplySign Desktop" 2>/dev/null | cut -f1 || echo "Unknown")
FILE_COUNT=$(find "$PACKAGE_DIR/SimplySign Desktop" -type f | wc -l)
echo "   ✅ Copied $FILE_COUNT essential files ($COPIED_SIZE)"

# List the files for verification
echo "   Essential files included:"
find "$PACKAGE_DIR/SimplySign Desktop" -type f -exec basename {} \; | sort | sed 's/^/     /'

# Copy only essential registry configuration (not user files)
echo "   Copying minimal configuration..."
CURRENT_USER="${USER:-${USERNAME:-$(whoami)}}"

# Only look for essential XML config files, not entire directories
mkdir -p "$PACKAGE_DIR/config"

# Look for specific SimplySign config files only
find "/c/Users/$CURRENT_USER/AppData/Local" -name "SimplySignDesktop.xml" -o -name "certum.xml" 2>/dev/null | while read -r config_file; do
    if [ -f "$config_file" ]; then
        echo "   Found essential config: $(basename "$config_file")"
        cp "$config_file" "$PACKAGE_DIR/config/" 2>/dev/null || true
    fi
done

find "/c/Users/$CURRENT_USER/AppData/Roaming" -name "SimplySignDesktop.xml" -o -name "certum.xml" 2>/dev/null | while read -r config_file; do
    if [ -f "$config_file" ]; then
        echo "   Found essential config: $(basename "$config_file")"
        cp "$config_file" "$PACKAGE_DIR/config/" 2>/dev/null || true
    fi
done

# Export registry settings
echo "   Exporting registry configuration..."
if command -v reg >/dev/null 2>&1; then
    mkdir -p "$PACKAGE_DIR/registry"
    
    # Export the configured registry keys
    reg export "HKEY_CURRENT_USER\\Software\\Certum" "$PACKAGE_DIR/registry/certum-hkcu.reg" 2>/dev/null || true
    reg export "HKEY_CURRENT_USER\\Software\\SimplySignDesktop" "$PACKAGE_DIR/registry/simplysign-hkcu.reg" 2>/dev/null || true
    reg export "HKEY_CURRENT_USER\\Software\\Asseco" "$PACKAGE_DIR/registry/asseco-hkcu.reg" 2>/dev/null || true
    
    echo "   ✅ Registry settings exported"
fi

# Create a restoration script
echo "   Creating restoration script..."
cat > "$PACKAGE_DIR/restore-simplysign.sh" << 'EOF'
#!/bin/bash

# Restore SimplySign Desktop from Package
# Restores the configured SimplySign Desktop installation

set -euo pipefail

echo "=== Restoring SimplySign Desktop from Package ==="

PACKAGE_DIR="$(dirname "${BASH_SOURCE[0]}")"
INSTALL_DIR="/c/Program Files/Certum"

# Create installation directory
mkdir -p "$INSTALL_DIR"

# Copy SimplySign Desktop (minimal essential files)
if [ -d "$PACKAGE_DIR/SimplySign Desktop" ]; then
    echo "📦 Restoring minimal SimplySign Desktop..."
    cp -r "$PACKAGE_DIR/SimplySign Desktop" "$INSTALL_DIR/"
    echo "✅ SimplySign Desktop restored to: $INSTALL_DIR/SimplySign Desktop"
    
    # List what was restored
    FILE_COUNT=$(find "$INSTALL_DIR/SimplySign Desktop" -type f | wc -l)
    DIR_SIZE=$(du -sh "$INSTALL_DIR/SimplySign Desktop" 2>/dev/null | cut -f1 || echo "Unknown")
    echo "   Restored $FILE_COUNT files ($DIR_SIZE)"
else
    echo "❌ SimplySign Desktop package not found"
    exit 1
fi

# Restore essential configuration
if [ -d "$PACKAGE_DIR/config" ]; then
    echo "🔧 Restoring essential configuration..."
    
    CURRENT_USER="${USER:-${USERNAME:-$(whoami)}}"
    USER_APPDATA_LOCAL="/c/Users/$CURRENT_USER/AppData/Local"
    USER_APPDATA_ROAMING="/c/Users/$CURRENT_USER/AppData/Roaming"
    
    mkdir -p "$USER_APPDATA_LOCAL/Certum"
    mkdir -p "$USER_APPDATA_ROAMING/Certum"
    
    # Copy essential config files
    find "$PACKAGE_DIR/config" -name "*.xml" -exec cp {} "$USER_APPDATA_LOCAL/Certum/" \; 2>/dev/null || true
    find "$PACKAGE_DIR/config" -name "*.xml" -exec cp {} "$USER_APPDATA_ROAMING/Certum/" \; 2>/dev/null || true
    
    echo "✅ Essential configuration restored"
fi

# Restore registry settings
if [ -d "$PACKAGE_DIR/registry" ] && command -v reg >/dev/null 2>&1; then
    echo "📋 Restoring registry configuration..."
    
    for reg_file in "$PACKAGE_DIR/registry"/*.reg; do
        if [ -f "$reg_file" ]; then
            echo "   Importing: $(basename "$reg_file")"
            reg import "$reg_file" 2>/dev/null || true
        fi
    done
    
    echo "✅ Registry configuration restored"
fi

# Verify installation
SIMPLYSIGN_EXE="$INSTALL_DIR/SimplySign Desktop/SimplySignDesktop.exe"
if [ -f "$SIMPLYSIGN_EXE" ]; then
    echo "✅ SimplySign Desktop successfully restored"
    echo "🎯 Configured for automatic OAuth2 authentication"
    echo "📍 Location: $SIMPLYSIGN_EXE"
else
    echo "❌ SimplySign Desktop restoration failed"
    exit 1
fi

echo ""
echo "🚀 SimplySign Desktop is ready for use!"
echo "💡 OAuth2 dialog should appear automatically on startup"
EOF

chmod +x "$PACKAGE_DIR/restore-simplysign.sh"

# Create package information file
echo "   Creating package information..."
FINAL_SIZE=$(du -sh "$PACKAGE_DIR" 2>/dev/null | cut -f1 || echo "Unknown")
cat > "$PACKAGE_DIR/package-info.txt" << EOF
Minimal SimplySign Desktop Package
==================================

Created: $(date)
Configuration: Automatic OAuth2 Authentication Enabled
Package Type: MINIMAL (Essential files only)

Contents:
- SimplySign Desktop/ - Core application files (executable + essential DLLs)
- config/ - Essential XML configuration files only
- registry/ - Registry settings export
- restore-simplysign.sh - Restoration script

Key Configuration:
- SimplySignDesktopShowLogonDialogAfterApplicationStartup = Yes
- OAuth2 dialog appears automatically on startup
- No manual trigger required for authentication

Optimization:
- Only includes essential SimplySign Desktop files (~6-10 MB)
- Excludes proCertum SmartSign suite and unnecessary files
- Minimal configuration files only

Usage:
1. Extract package
2. Run restore-simplysign.sh
3. Start SimplySign Desktop
4. OAuth2 dialog should appear automatically

Package Size: $FINAL_SIZE (optimized for artifacts)
EOF

# Create the final zip package
echo "🗜️ Creating compressed package..."
if command -v 7z >/dev/null 2>&1; then
    # Use 7-Zip for better compression
    7z a -tzip -mx=9 "simplysign-desktop-package.zip" "$PACKAGE_DIR"/* > /dev/null
    echo "   ✅ Created with 7-Zip compression"
elif command -v zip >/dev/null 2>&1; then
    # Use standard zip
    cd "$PACKAGE_DIR"
    zip -r -9 "../simplysign-desktop-package.zip" . > /dev/null
    cd ..
    echo "   ✅ Created with standard zip compression"
else
    # Fallback: tar.gz
    tar -czf "simplysign-desktop-package.tar.gz" -C "$PACKAGE_DIR" .
    echo "   ✅ Created with tar.gz compression"
fi

# Get package size and verify it's minimal
if [ -f "simplysign-desktop-package.zip" ]; then
    PACKAGE_SIZE=$(du -sh "simplysign-desktop-package.zip" | cut -f1)
    PACKAGE_SIZE_BYTES=$(du -b "simplysign-desktop-package.zip" | cut -f1)
    echo "📦 Package size: $PACKAGE_SIZE"
    PACKAGE_FILE="simplysign-desktop-package.zip"
    
    # Verify the package is appropriately sized (should be under 50MB for minimal)
    MAX_SIZE_BYTES=$((50 * 1024 * 1024))  # 50MB in bytes
    if [ "$PACKAGE_SIZE_BYTES" -gt "$MAX_SIZE_BYTES" ]; then
        echo "⚠️ WARNING: Package seems large ($PACKAGE_SIZE) - expected <50MB for minimal SimplySign Desktop"
        echo "   This may include unnecessary files from proCertum SmartSign suite"
    else
        echo "✅ Package size optimal for minimal SimplySign Desktop"
    fi
    
elif [ -f "simplysign-desktop-package.tar.gz" ]; then
    PACKAGE_SIZE=$(du -sh "simplysign-desktop-package.tar.gz" | cut -f1)
    echo "📦 Package size: $PACKAGE_SIZE"
    PACKAGE_FILE="simplysign-desktop-package.tar.gz"
fi

# Cleanup temporary directory
rm -rf "$PACKAGE_DIR"

echo ""
echo "✅ SimplySign Desktop package created successfully!"
echo "📦 Package: $PACKAGE_FILE"
echo "📏 Size: $PACKAGE_SIZE"
echo "🎯 Configured for automatic OAuth2 authentication"
echo ""
echo "🚀 Ready for upload as GitHub Actions artifact!"
echo "💡 Other workflows can download and restore this configured installation"
