#!/bin/bash

# Install SimplySign Desktop from Windows Bundle
# Downloads and installs SimplySign Desktop directly using the official installer

set -euo pipefail

echo "=== Installing SimplySign Desktop from Windows Bundle ==="

# Bundle download URL and local file
BUNDLE_URL="https://files.certum.eu/software/SimplySignDesktop/Windows/9.3.2.67/SimplySignDesktop-9.3.2.67-win-64-bit.exe"
BUNDLE_FILE="SimplySignDesktop-9.3.2.67-win-64-bit.exe"
INSTALL_DIR="/c/Program Files/Certum"

# Download the bundle if not already present
if [ ! -f "$BUNDLE_FILE" ]; then
  echo "📥 Downloading SimplySign Desktop installer..."
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
    echo "❌ Failed to download installer"
    exit 1
  fi
fi

echo "✅ Installer ready: $BUNDLE_FILE ($(du -h "$BUNDLE_FILE" | cut -f1))"

# Install SimplySign Desktop using PowerShell
echo "� Installing SimplySign Desktop..."

# Use PowerShell to run the installer silently
powershell -Command "
    Write-Host 'Running SimplySign Desktop installer...'
    \$installerPath = '$(cygpath -w "$PWD/$BUNDLE_FILE")'
    \$installDir = '$(cygpath -w "$INSTALL_DIR")'
    
    # Try different silent installation methods
    Write-Host 'Attempting silent installation...'
    
    # Method 1: Standard silent switches
    try {
        \$process = Start-Process -FilePath \$installerPath -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/CLOSEAPPLICATIONS', '/RESTARTAPPLICATIONS', '/DIR=', \$installDir -Wait -PassThru
        if (\$process.ExitCode -eq 0) {
            Write-Host '✅ Installation completed successfully'
            exit 0
        } else {
            Write-Host '⚠️ Method 1 failed with exit code:' \$process.ExitCode
        }
    } catch {
        Write-Host '⚠️ Method 1 failed:' \$_.Exception.Message
    }
    
    # Method 2: NSIS-style switches
    try {
        \$process = Start-Process -FilePath \$installerPath -ArgumentList '/S', '/D=', \$installDir -Wait -PassThru
        if (\$process.ExitCode -eq 0) {
            Write-Host '✅ Installation completed successfully (Method 2)'
            exit 0
        } else {
            Write-Host '⚠️ Method 2 failed with exit code:' \$process.ExitCode
        }
    } catch {
        Write-Host '⚠️ Method 2 failed:' \$_.Exception.Message
    }
    
    # Method 3: InstallShield-style switches
    try {
        \$process = Start-Process -FilePath \$installerPath -ArgumentList '/s', '/v/qn' -Wait -PassThru
        if (\$process.ExitCode -eq 0) {
            Write-Host '✅ Installation completed successfully (Method 3)'
            exit 0
        } else {
            Write-Host '⚠️ Method 3 failed with exit code:' \$process.ExitCode
        }
    } catch {
        Write-Host '⚠️ Method 3 failed:' \$_.Exception.Message
    }
    
    # Method 4: Just run the installer and see what happens
    Write-Host '🎯 Attempting default installation...'
    try {
        \$process = Start-Process -FilePath \$installerPath -Wait -PassThru
        Write-Host 'Installation process completed with exit code:' \$process.ExitCode
    } catch {
        Write-Host '❌ Installation failed:' \$_.Exception.Message
        exit 1
    }
"

POWERSHELL_EXIT=$?

if [ $POWERSHELL_EXIT -ne 0 ]; then
    echo "❌ PowerShell installation failed"
    exit 1
fi

# Wait a moment for installation to complete
sleep 5

# Verify installation by looking for SimplySign Desktop in common locations
echo "🔍 Verifying installation..."

POSSIBLE_LOCATIONS=(
    "/c/Program Files/Certum/SimplySign Desktop"
    "/c/Program Files (x86)/Certum/SimplySign Desktop"
    "/c/Program Files/SimplySign Desktop"
    "/c/Program Files (x86)/SimplySign Desktop"
    "/c/Program Files/proCertum/SimplySign Desktop"
    "/c/Program Files (x86)/proCertum/SimplySign Desktop"
)

FOUND_LOCATION=""
for location in "${POSSIBLE_LOCATIONS[@]}"; do
    if [ -f "$location/SimplySignDesktop.exe" ]; then
        FOUND_LOCATION="$location"
        break
    fi
done

if [ -n "$FOUND_LOCATION" ]; then
    echo "✅ SimplySign Desktop found at: $FOUND_LOCATION"
    
    INSTALLED_SIZE=$(du -sh "$FOUND_LOCATION" | cut -f1)
    FILE_COUNT=$(find "$FOUND_LOCATION" -type f | wc -l)
    echo "� Size: $INSTALLED_SIZE ($FILE_COUNT files)"
    
    # List key files
    echo "� Key files installed:"
    find "$FOUND_LOCATION" -name "*.exe" -o -name "*.dll" -o -name "*.config" | head -10 | sed 's/^/   /'
    
    # Create a symlink to the standard location if it's not there
    if [ "$FOUND_LOCATION" != "/c/Program Files/Certum/SimplySign Desktop" ]; then
        echo "🔗 Creating standard location symlink..."
        mkdir -p "/c/Program Files/Certum"
        ln -sf "$FOUND_LOCATION" "/c/Program Files/Certum/SimplySign Desktop" 2>/dev/null || {
            echo "   Symlink failed, copying instead..."
            cp -r "$FOUND_LOCATION" "/c/Program Files/Certum/SimplySign Desktop"
        }
    fi
    
else
    echo "❌ SimplySign Desktop not found after installation"
    echo "🔍 Searching for any SimplySign files..."
    find /c/Program* -name "*Simply*" -type f 2>/dev/null | head -10 || echo "   No SimplySign files found"
    exit 1
fi

# Cleanup installer
echo "🧹 Cleaning up installer..."
rm -f "$BUNDLE_FILE"

echo ""
echo "🚀 SimplySign Desktop installation complete!"
echo "📂 Installed at: $FOUND_LOCATION"
echo "🎯 Ready for OAuth2 configuration"
