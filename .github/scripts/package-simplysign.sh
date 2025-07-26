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

echo "📦 Creating SimplySign Desktop package..."

# Copy the entire SimplySign Desktop installation
echo "   Copying SimplySign Desktop files..."
cp -r "$SIMPLYSIGN_DIR" "$PACKAGE_DIR/" || {
    echo "❌ Failed to copy SimplySign Desktop directory"
    exit 1
}

# Copy relevant user configuration (if exists)
echo "   Copying user configuration..."
CURRENT_USER="${USER:-${USERNAME:-$(whoami)}}"
USER_CONFIG_LOCATIONS=(
    "/c/Users/$CURRENT_USER/AppData/Local/Certum"
    "/c/Users/$CURRENT_USER/AppData/Roaming/Certum"
    "/c/Users/$CURRENT_USER/AppData/Local/SimplySign Desktop"
    "/c/Users/$CURRENT_USER/AppData/Roaming/SimplySign Desktop"
)

for config_dir in "${USER_CONFIG_LOCATIONS[@]}"; do
    if [ -d "$config_dir" ]; then
        echo "   Found user config: $config_dir"
        mkdir -p "$PACKAGE_DIR/user-config"
        cp -r "$config_dir" "$PACKAGE_DIR/user-config/" 2>/dev/null || true
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

# Copy SimplySign Desktop
if [ -d "$PACKAGE_DIR/SimplySign Desktop" ]; then
    echo "📦 Restoring SimplySign Desktop..."
    cp -r "$PACKAGE_DIR/SimplySign Desktop" "$INSTALL_DIR/"
    echo "✅ SimplySign Desktop restored to: $INSTALL_DIR/SimplySign Desktop"
else
    echo "❌ SimplySign Desktop package not found"
    exit 1
fi

# Restore user configuration
if [ -d "$PACKAGE_DIR/user-config" ]; then
    echo "🔧 Restoring user configuration..."
    
    CURRENT_USER="${USER:-${USERNAME:-$(whoami)}}"
    USER_APPDATA="/c/Users/$CURRENT_USER/AppData"
    
    mkdir -p "$USER_APPDATA/Local"
    mkdir -p "$USER_APPDATA/Roaming"
    
    # Copy configuration files
    find "$PACKAGE_DIR/user-config" -type d -name "Certum" -exec cp -r {} "$USER_APPDATA/Local/" \; 2>/dev/null || true
    find "$PACKAGE_DIR/user-config" -type d -name "Certum" -exec cp -r {} "$USER_APPDATA/Roaming/" \; 2>/dev/null || true
    find "$PACKAGE_DIR/user-config" -type d -name "SimplySign Desktop" -exec cp -r {} "$USER_APPDATA/Local/" \; 2>/dev/null || true
    find "$PACKAGE_DIR/user-config" -type d -name "SimplySign Desktop" -exec cp -r {} "$USER_APPDATA/Roaming/" \; 2>/dev/null || true
    
    echo "✅ User configuration restored"
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
cat > "$PACKAGE_DIR/package-info.txt" << EOF
SimplySign Desktop Package
==========================

Created: $(date)
Configuration: Automatic OAuth2 Authentication Enabled

Contents:
- SimplySign Desktop/ - Complete application installation
- user-config/ - User configuration files (if found)
- registry/ - Registry settings export
- restore-simplysign.sh - Restoration script

Key Configuration:
- SimplySignDesktopShowLogonDialogAfterApplicationStartup = Yes
- OAuth2 dialog appears automatically on startup
- No manual trigger required for authentication

Usage:
1. Extract package
2. Run restore-simplysign.sh
3. Start SimplySign Desktop
4. OAuth2 dialog should appear automatically

Size: $(du -sh "$PACKAGE_DIR" | cut -f1)
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

# Get package size
if [ -f "simplysign-desktop-package.zip" ]; then
    PACKAGE_SIZE=$(du -sh "simplysign-desktop-package.zip" | cut -f1)
    echo "📦 Package size: $PACKAGE_SIZE"
    PACKAGE_FILE="simplysign-desktop-package.zip"
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
