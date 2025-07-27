#!/bin/bash

# Package SimplySign Desktop (Bundle Version)
# Creates a self-contained package from bundle-extracted SimplySign Desktop

set -euo pipefail

echo "=== Packaging SimplySign Desktop (Bundle Version) ==="

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

echo "📦 Creating self-contained SimplySign Desktop package..."

# Copy the entire SimplySign Desktop directory
echo "   Copying SimplySign Desktop files..."
cp -r "$SIMPLYSIGN_DIR" "$PACKAGE_DIR/"

# Get the size info
COPIED_SIZE=$(du -sh "$PACKAGE_DIR/SimplySign Desktop" 2>/dev/null | cut -f1 || echo "Unknown")
FILE_COUNT=$(find "$PACKAGE_DIR/SimplySign Desktop" -type f | wc -l)
echo "   ✅ Copied $FILE_COUNT files ($COPIED_SIZE)"

# List key files for verification
echo "   Key files included:"
find "$PACKAGE_DIR/SimplySign Desktop" -name "*.exe" -o -name "*.dll" -o -name "*.config" | head -10 | sed 's/^/     /'

# Export current registry settings to .reg files
echo "   Exporting registry configuration..."
if command -v reg >/dev/null 2>&1; then
    # Export configured registry keys to .reg files that users can double-click
    reg export "HKEY_CURRENT_USER\\Software\\Certum" "$PACKAGE_DIR/certum-config.reg" 2>/dev/null || true
    reg export "HKEY_CURRENT_USER\\Software\\SimplySignDesktop" "$PACKAGE_DIR/simplysign-config.reg" 2>/dev/null || true
    reg export "HKEY_CURRENT_USER\\Software\\Asseco" "$PACKAGE_DIR/asseco-config.reg" 2>/dev/null || true
    
    echo "   ✅ Registry settings exported as .reg files"
fi

# Create a simple README for users
cat > "$PACKAGE_DIR/README.txt" << 'EOF'
SimplySign Desktop - Self-Contained Package
==========================================

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
- Extracted from official proCertum SmartSign bundle
- Self-contained - no additional installation needed

NOTES:
- Import .reg files before first run
- OAuth2 dialog appears automatically on startup
- Ready for cloud signing with Certum services
- No scripts needed - just unzip and run!

TROUBLESHOOTING:
- If OAuth2 dialog doesn't appear, check registry settings
- Ensure network connectivity to cloudsign.webnotarius.pl
- Run as administrator if needed
EOF

# Create a simple PowerShell script for registry import (alternative to double-clicking)
cat > "$PACKAGE_DIR/import-settings.ps1" << 'EOF'
# Import Registry Settings for SimplySign Desktop
# Alternative to double-clicking .reg files

Write-Host "Importing SimplySign Desktop registry settings..."

$regFiles = @(
    "certum-config.reg",
    "simplysign-config.reg", 
    "asseco-config.reg"
)

foreach ($regFile in $regFiles) {
    if (Test-Path $regFile) {
        Write-Host "Importing $regFile..."
        try {
            reg import $regFile
            Write-Host "✅ $regFile imported successfully"
        }
        catch {
            Write-Host "⚠️ Failed to import $regFile"
        }
    }
}

Write-Host ""
Write-Host "🚀 Registry settings imported!"
Write-Host "💡 You can now run: SimplySign Desktop\SimplySignDesktop.exe"
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
echo "✅ Self-contained SimplySign Desktop package created!"
echo "📦 Package: $PACKAGE_FILE"
echo "📏 Size: $PACKAGE_SIZE"
echo ""
echo "🚀 USER INSTRUCTIONS:"
echo "   1. Unzip the package"
echo "   2. Import settings: Double-click .reg files OR run import-settings.ps1"
echo "   3. Run: SimplySign Desktop\\SimplySignDesktop.exe"
echo "   4. OAuth2 dialog appears automatically!"
echo ""
echo "💡 Self-contained package - no installation required!"
