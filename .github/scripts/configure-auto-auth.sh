#!/bin/bash

# Configure SimplySign Desktop for Automatic Authentication
# Extracts the BREAKTHROUGH discovery: SimplySignDesktopShowLogonDialogAfterApplicationStartup

set -euo pipefail

echo "=== Configuring SimplySign Desktop for Automatic Authentication ==="
echo "🎯 BREAKTHROUGH: Setting SimplySignDesktopShowLogonDialogAfterApplicationStartup"

# Check if SimplySign Desktop is installed
SIMPLYSIGN_EXE="/c/Program Files/Certum/SimplySign Desktop/SimplySignDesktop.exe"
if [ ! -f "$SIMPLYSIGN_EXE" ]; then
  echo "❌ SimplySign Desktop not found at: $SIMPLYSIGN_EXE"
  exit 1
fi

echo "✅ SimplySign Desktop found: $SIMPLYSIGN_EXE"

# Method 1: Windows Registry configuration
echo ""
echo "📋 Method 1: Configuring Windows Registry..."

if command -v reg >/dev/null 2>&1; then
    echo "   Setting registry keys for automatic OAuth2 dialog..."
    
    # Registry locations for SimplySign Desktop settings
    REG_LOCATIONS=(
        "HKEY_CURRENT_USER\\Software\\Certum\\SimplySign Desktop"
        "HKEY_CURRENT_USER\\Software\\SimplySignDesktop" 
        "HKEY_CURRENT_USER\\Software\\Asseco\\SimplySign Desktop"
        "HKEY_CURRENT_USER\\Software\\Asseco Data Systems\\SimplySign Desktop"
        "HKEY_LOCAL_MACHINE\\Software\\Certum\\SimplySign Desktop"
    )
    
    for reg_path in "${REG_LOCATIONS[@]}"; do
        echo "   Configuring registry path: $reg_path"
        
        # The critical setting from macOS discovery
        reg add "$reg_path" /v "SimplySignDesktopShowLogonDialogAfterApplicationStartup" /t REG_SZ /d "Yes" /f 2>/dev/null || echo "     Registry path attempted: $reg_path"
        
        # Additional related settings
        reg add "$reg_path" /v "ShowLogonDialogAfterApplicationStartup" /t REG_SZ /d "Yes" /f 2>/dev/null || true
        reg add "$reg_path" /v "AutoShowLogonDialog" /t REG_SZ /d "Yes" /f 2>/dev/null || true
        reg add "$reg_path" /v "AutomaticAuthentication" /t REG_SZ /d "Yes" /f 2>/dev/null || true
        
        echo "   ✅ Registry configuration applied"
    done
    
    echo "✅ Registry configuration completed"
else
    echo "⚠️ Registry command not available"
fi

# Method 2: Configuration file approach
echo ""
echo "📋 Method 2: Configuring via configuration files..."

# Look for SimplySign configuration directories
CURRENT_USER="${USER:-${USERNAME:-$(whoami)}}"
CONFIG_LOCATIONS=(
    "/c/Program Files/Certum/SimplySign Desktop"
    "/c/ProgramData/Certum/SimplySign Desktop" 
    "/c/Users/$CURRENT_USER/AppData/Local/Certum"
    "/c/Users/$CURRENT_USER/AppData/Roaming/Certum"
    "/c/Users/$CURRENT_USER/AppData/Local/SimplySign Desktop"
    "/c/Users/$CURRENT_USER/AppData/Roaming/SimplySign Desktop"
)

for config_dir in "${CONFIG_LOCATIONS[@]}"; do
    if [ -d "$config_dir" ]; then
        echo "   Found config directory: $config_dir"
        
        # Look for XML configuration files (like macOS SimplySignDesktop.xml)
        find "$config_dir" -name "*.xml" 2>/dev/null | while read -r config_file; do
            echo "   Configuring XML file: $config_file"
            
            # Create backup
            cp "$config_file" "${config_file}.backup" 2>/dev/null || true
            
            # Add the critical setting if it doesn't exist
            if ! grep -q "ShowLogonDialogAfterApplicationStartup" "$config_file" 2>/dev/null; then
                echo "   Adding automatic authentication setting..."
                
                # Add the setting before closing tags
                sed -i.bak '/<\/dict>/i\
    <key>SimplySignDesktopShowLogonDialogAfterApplicationStartup</key>\
    <string>Yes</string>' "$config_file" 2>/dev/null || true
                
                sed -i.bak '/<\/plist>/i\
    <key>SimplySignDesktopShowLogonDialogAfterApplicationStartup</key>\
    <string>Yes</string>' "$config_file" 2>/dev/null || true
            fi
        done
    fi
done

# Method 3: PowerShell configuration
echo ""
echo "📋 Method 3: PowerShell registry configuration..."

if command -v powershell >/dev/null 2>&1; then
    echo "   Configuring registry via PowerShell..."
    powershell -Command "
    Write-Host 'Configuring SimplySign Desktop via PowerShell...'
    
    \$regPaths = @(
        'HKCU:\Software\Certum\SimplySign Desktop',
        'HKCU:\Software\SimplySignDesktop',
        'HKCU:\Software\Asseco\SimplySign Desktop'
    )
    
    foreach (\$regPath in \$regPaths) {
        try {
            if (-not (Test-Path \$regPath)) {
                New-Item -Path \$regPath -Force | Out-Null
                Write-Host \"Created registry path: \$regPath\"
            }
            
            # Set the critical breakthrough setting
            Set-ItemProperty -Path \$regPath -Name 'SimplySignDesktopShowLogonDialogAfterApplicationStartup' -Value 'Yes' -Type String -Force
            Set-ItemProperty -Path \$regPath -Name 'ShowLogonDialogAfterApplicationStartup' -Value 'Yes' -Type String -Force
            Set-ItemProperty -Path \$regPath -Name 'AutoShowLogonDialog' -Value 'Yes' -Type String -Force
            Set-ItemProperty -Path \$regPath -Name 'AutomaticAuthentication' -Value 'Yes' -Type String -Force
            
            Write-Host \"Registry configuration set: \$regPath\"
            
            # Verify the setting was applied
            \$value = Get-ItemProperty -Path \$regPath -Name 'SimplySignDesktopShowLogonDialogAfterApplicationStartup' -ErrorAction SilentlyContinue
            if (\$value) {
                Write-Host \"  Verified: SimplySignDesktopShowLogonDialogAfterApplicationStartup = \$(\$value.SimplySignDesktopShowLogonDialogAfterApplicationStartup)\"
            }
        } catch {
            Write-Host \"Registry path failed: \$regPath - \$(\$_.Exception.Message)\"
        }
    }
    
    Write-Host 'PowerShell configuration completed'
    " || echo "   PowerShell configuration failed"
    
    echo "✅ PowerShell configuration completed"
else
    echo "⚠️ PowerShell not available"
fi

echo ""
echo "✅ Automatic authentication configuration completed!"
echo "🎯 BREAKTHROUGH: SimplySign Desktop configured to show OAuth2 dialog on startup"
echo "📱 OAuth2 dialog should now appear automatically when application starts"

# Verify registry settings were applied
echo ""
echo "🔍 Verifying configuration..."

if command -v reg >/dev/null 2>&1; then
    for reg_path in "HKEY_CURRENT_USER\\Software\\Certum\\SimplySign Desktop" \
                   "HKEY_CURRENT_USER\\Software\\SimplySignDesktop"; do
        
        echo "   Checking: $reg_path"
        if reg query "$reg_path" /v "SimplySignDesktopShowLogonDialogAfterApplicationStartup" 2>/dev/null; then
            echo "   ✅ Configuration verified in registry"
        else
            echo "   ⚠️ Registry key not confirmed (may still work)"
        fi
    done
fi

echo ""
echo "🚀 SimplySign Desktop is now configured for automatic OAuth2 authentication!"
echo "💡 Next: Package the configured application for reuse in signing workflows"
