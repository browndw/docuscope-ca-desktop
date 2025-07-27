#!/bin/bash

# Simple SimplySign Desktop Package
# Creates a package where users just unzip and run the executable

set -euo pipefail

echo "=== Creating Simple SimplySign Desktop Package ==="

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

echo "📦 Creating simple SimplySign Desktop package..."

# Copy the entire SimplySign Desktop directory as-is
echo "   Copying SimplySign Desktop files..."
cp -r "$SIMPLYSIGN_DIR" "$PACKAGE_DIR/"

# Get the size info
COPIED_SIZE=$(du -sh "$PACKAGE_DIR/SimplySign Desktop" 2>/dev/null | cut -f1 || echo "Unknown")
FILE_COUNT=$(find "$PACKAGE_DIR/SimplySign Desktop" -type f | wc -l)
echo "   ✅ Copied $FILE_COUNT files ($COPIED_SIZE)"

# Export current registry settings to .reg files that can be double-clicked
echo "   Exporting registry configuration..."
if command -v reg >/dev/null 2>&1; then
    # Export configured registry keys to .reg files
    reg export "HKEY_CURRENT_USER\\Software\\Certum" "$PACKAGE_DIR/certum-config.reg" 2>/dev/null || true
    reg export "HKEY_CURRENT_USER\\Software\\SimplySignDesktop" "$PACKAGE_DIR/simplysign-config.reg" 2>/dev/null || true
    reg export "HKEY_CURRENT_USER\\Software\\Asseco" "$PACKAGE_DIR/asseco-config.reg" 2>/dev/null || true
    
    echo "   ✅ Registry settings exported as .reg files"
fi

# Create a simple README for users
cat > "$PACKAGE_DIR/README.txt" << 'EOF'
SimplySign Desktop - Ready to Use Package
=========================================

QUICK START:
1. Double-click the .reg files to import registry settings:
   - certum-config.reg
   - simplysign-config.reg  
   - asseco-config.reg

2. Run the executable:
   SimplySign Desktop\SimplySignDesktop.exe

3. OAuth2 dialog should appear automatically!

CONFIGURATION:
- Pre-configured for automatic OAuth2 authentication
- SimplySignDesktopShowLogonDialogAfterApplicationStartup = Yes
- No manual trigger required

NOTES:
- Import .reg files before first run
- OAuth2 dialog appears automatically on startup
- Ready for cloud signing with Certum services
EOF

# Create the final zip package
echo "🗜️ Creating compressed package..."
if command -v 7z >/dev/null 2>&1; then
    7z a -tzip -mx=9 "simplysign-desktop-package.zip" "$PACKAGE_DIR"/* > /dev/null
    echo "   ✅ Created with 7-Zip compression"
elif command -v zip >/dev/null 2>&1; then
    cd "$PACKAGE_DIR"
    zip -r -9 "../simplysign-desktop-package.zip" . > /dev/null
    cd ..
    echo "   ✅ Created with standard zip compression"
else
    tar -czf "simplysign-desktop-package.tar.gz" -C "$PACKAGE_DIR" .
    echo "   ✅ Created with tar.gz compression"
fi

# Get package size
if [ -f "simplysign-desktop-package.zip" ]; then
    PACKAGE_SIZE=$(du -sh "simplysign-desktop-package.zip" | cut -f1)
    PACKAGE_FILE="simplysign-desktop-package.zip"
elif [ -f "simplysign-desktop-package.tar.gz" ]; then
    PACKAGE_SIZE=$(du -sh "simplysign-desktop-package.tar.gz" | cut -f1)
    PACKAGE_FILE="simplysign-desktop-package.tar.gz"
fi

# Cleanup temporary directory
rm -rf "$PACKAGE_DIR"

echo ""
echo "✅ Simple SimplySign Desktop package created!"
echo "📦 Package: $PACKAGE_FILE"
echo "📏 Size: $PACKAGE_SIZE"
echo ""
echo "🚀 USER INSTRUCTIONS:"
echo "   1. Unzip the package"
echo "   2. Double-click the .reg files to import settings"
echo "   3. Run SimplySign Desktop\\SimplySignDesktop.exe"
echo "   4. OAuth2 dialog appears automatically!"
echo ""
echo "💡 No scripts needed - just unzip and run!"
