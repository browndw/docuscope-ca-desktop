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

# Install SimplySign Desktop using PowerShell with timeout
echo "🚀 Installing SimplySign Desktop..."

# Use timeout to prevent hanging and try multiple installation methods
timeout 300s powershell -Command "
    Write-Host 'Running SimplySign Desktop installer with timeout...'
    \$installerPath = '$(cygpath -w "$PWD/$BUNDLE_FILE")'
    
    # Method 1: NSIS-style silent switch (most common)
    try {
        Write-Host 'Method 1: /S switch...'
        \$process = Start-Process -FilePath \$installerPath -ArgumentList '/S' -PassThru
        \$completed = \$process.WaitForExit(120000) # 2 minute timeout
        if (\$completed -and \$process.ExitCode -eq 0) {
            Write-Host '✅ Installation completed successfully'
            exit 0
        } elseif (!\$completed) {
            Write-Host '⚠️ Method 1 timed out'
            try { \$process.Kill() } catch {}
        } else {
            Write-Host '⚠️ Method 1 failed with exit code:' \$process.ExitCode
        }
    } catch {
        Write-Host '⚠️ Method 1 exception:' \$_.Exception.Message
    }
    
    Start-Sleep -Seconds 3
    
    # Method 2: Inno Setup style switches
    try {
        Write-Host 'Method 2: /VERYSILENT switch...'
        \$process = Start-Process -FilePath \$installerPath -ArgumentList '/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART' -PassThru
        \$completed = \$process.WaitForExit(120000) # 2 minute timeout
        if (\$completed -and \$process.ExitCode -eq 0) {
            Write-Host '✅ Installation completed successfully (Method 2)'
            exit 0
        } elseif (!\$completed) {
            Write-Host '⚠️ Method 2 timed out'
            try { \$process.Kill() } catch {}
        } else {
            Write-Host '⚠️ Method 2 failed with exit code:' \$process.ExitCode
        }
    } catch {
        Write-Host '⚠️ Method 2 exception:' \$_.Exception.Message
    }
    
    Start-Sleep -Seconds 3
    
    # Method 3: Default run (might show GUI but we'll kill it)
    try {
        Write-Host 'Method 3: Default run with short timeout...'
        \$process = Start-Process -FilePath \$installerPath -PassThru
        \$completed = \$process.WaitForExit(60000) # 1 minute timeout
        if (!\$completed) {
            Write-Host '⚠️ Method 3 timed out (expected for GUI installer)'
            try { \$process.Kill() } catch {}
        }
        Write-Host '🤞 Checking if installation occurred anyway...'
    } catch {
        Write-Host '⚠️ Method 3 exception:' \$_.Exception.Message
    }
    
    Write-Host 'Installation attempts completed, checking for installed files...'
" || echo "⚠️ PowerShell timed out, but continuing to check for installation..."

# Wait for any lingering processes to finish
sleep 10

# Verify installation by looking for SimplySign Desktop in common locations
echo "🔍 Verifying installation..."

POSSIBLE_LOCATIONS=(
    "/c/Program Files/Certum/SimplySign Desktop"
    "/c/Program Files (x86)/Certum/SimplySign Desktop"
    "/c/Program Files/SimplySign Desktop"
    "/c/Program Files (x86)/SimplySign Desktop"
    "/c/Program Files/proCertum/SimplySign Desktop"
    "/c/Program Files (x86)/proCertum/SimplySign Desktop"
    "/c/Program Files/proCertum SmartSign/SimplySign Desktop"
    "/c/Program Files (x86)/proCertum SmartSign/SimplySign Desktop"
)

FOUND_LOCATION=""
for location in "${POSSIBLE_LOCATIONS[@]}"; do
    echo "   Checking: $location"
    if [ -f "$location/SimplySignDesktop.exe" ]; then
        FOUND_LOCATION="$location"
        echo "   ✅ Found!"
        break
    fi
done

if [ -n "$FOUND_LOCATION" ]; then
    echo "✅ SimplySign Desktop found at: $FOUND_LOCATION"
    
    INSTALLED_SIZE=$(du -sh "$FOUND_LOCATION" | cut -f1)
    FILE_COUNT=$(find "$FOUND_LOCATION" -type f | wc -l)
    echo "📏 Size: $INSTALLED_SIZE ($FILE_COUNT files)"
    
    # List key files
    echo "🔑 Key files installed:"
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
    
    echo "🔍 Searching for any proCertum files..."
    find /c/Program* -name "*proCertum*" -type d 2>/dev/null | head -10 || echo "   No proCertum directories found"
    
    echo "🔍 All executables in Program Files..."
    find /c/Program* -name "*.exe" 2>/dev/null | grep -i sign | head -10 || echo "   No signing executables found"
    
    exit 1
fi

# Cleanup installer
echo "🧹 Cleaning up installer..."
rm -f "$BUNDLE_FILE"

echo ""
echo "🚀 SimplySign Desktop installation complete!"
echo "📂 Installed at: $FOUND_LOCATION"
echo "🎯 Ready for OAuth2 configuration"
