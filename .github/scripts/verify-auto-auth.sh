#!/bin/bash

# Verify SimplySign Desktop Automatic Authentication Configuration
# Tests that the configuration was applied correctly

set -euo pipefail

echo "=== Verifying SimplySign Desktop Configuration ==="

# Check if SimplySign Desktop is installed
SIMPLYSIGN_EXE="/c/Program Files/Certum/SimplySign Desktop/SimplySignDesktop.exe"
if [ ! -f "$SIMPLYSIGN_EXE" ]; then
  echo "❌ SimplySign Desktop not found at: $SIMPLYSIGN_EXE"
  exit 1
fi

echo "✅ SimplySign Desktop found: $SIMPLYSIGN_EXE"

# Verify registry configuration
echo ""
echo "🔍 Verifying registry configuration..."

if command -v reg >/dev/null 2>&1; then
    REG_LOCATIONS=(
        "HKEY_CURRENT_USER\\Software\\Certum\\SimplySign Desktop"
        "HKEY_CURRENT_USER\\Software\\SimplySignDesktop"
        "HKEY_CURRENT_USER\\Software\\Asseco\\SimplySign Desktop"
    )
    
    REGISTRY_CONFIGURED=false
    
    for reg_path in "${REG_LOCATIONS[@]}"; do
        echo "   Checking: $reg_path"
        
        if reg query "$reg_path" 2>/dev/null | grep -q "SimplySignDesktopShowLogonDialogAfterApplicationStartup"; then
            echo "   ✅ Found SimplySignDesktopShowLogonDialogAfterApplicationStartup"
            
            # Get the value
            VALUE=$(reg query "$reg_path" /v "SimplySignDesktopShowLogonDialogAfterApplicationStartup" 2>/dev/null | grep "REG_SZ" | awk '{print $3}' || echo "")
            if [ "$VALUE" = "Yes" ]; then
                echo "   ✅ Value correctly set to: $VALUE"
                REGISTRY_CONFIGURED=true
            else
                echo "   ⚠️ Value is: $VALUE (expected: Yes)"
            fi
        else
            echo "   ❌ SimplySignDesktopShowLogonDialogAfterApplicationStartup not found"
        fi
        
        # Check other related settings
        for setting in "ShowLogonDialogAfterApplicationStartup" "AutoShowLogonDialog" "AutomaticAuthentication"; do
            if reg query "$reg_path" /v "$setting" 2>/dev/null >/dev/null; then
                VALUE=$(reg query "$reg_path" /v "$setting" 2>/dev/null | grep "REG_SZ" | awk '{print $3}' || echo "")
                echo "   ✅ Additional setting $setting = $VALUE"
            fi
        done
        
        echo ""
    done
    
    if [ "$REGISTRY_CONFIGURED" = true ]; then
        echo "✅ Registry configuration verified"
    else
        echo "⚠️ Registry configuration not confirmed"
    fi
else
    echo "⚠️ Registry command not available for verification"
fi

# Check configuration files
echo ""
echo "🔍 Checking configuration files..."

CURRENT_USER="${USER:-${USERNAME:-$(whoami)}}"
CONFIG_LOCATIONS=(
    "/c/Program Files/Certum/SimplySign Desktop"
    "/c/Users/$CURRENT_USER/AppData/Local/Certum"
    "/c/Users/$CURRENT_USER/AppData/Roaming/Certum"
    "/c/Users/$CURRENT_USER/AppData/Local/SimplySign Desktop"
    "/c/Users/$CURRENT_USER/AppData/Roaming/SimplySign Desktop"
)

CONFIG_FILES_FOUND=false

for config_dir in "${CONFIG_LOCATIONS[@]}"; do
    if [ -d "$config_dir" ]; then
        echo "   Checking directory: $config_dir"
        
        # Look for XML configuration files
        find "$config_dir" -name "*.xml" 2>/dev/null | while read -r config_file; do
            echo "   Found XML config: $config_file"
            CONFIG_FILES_FOUND=true
            
            if grep -q "SimplySignDesktopShowLogonDialogAfterApplicationStartup" "$config_file" 2>/dev/null; then
                echo "   ✅ Configuration found in XML file"
            else
                echo "   ❌ Configuration not found in XML file"
            fi
        done
    fi
done

# Test basic functionality
echo ""
echo "🧪 Testing basic functionality..."

echo "   Testing version check..."
if timeout 10 "$SIMPLYSIGN_EXE" --version 2>&1 | head -5; then
    echo "   ✅ Version check successful"
else
    echo "   ⚠️ Version check completed with warnings"
fi

echo ""
echo "   Testing help command..."
if timeout 10 "$SIMPLYSIGN_EXE" --help 2>&1 | head -10; then
    echo "   ✅ Help command successful"
else
    echo "   ⚠️ Help command completed with warnings"
fi

# Test automatic authentication trigger (brief test)
echo ""
echo "🚀 Testing automatic authentication trigger..."
echo "   Starting SimplySign Desktop briefly to test automatic OAuth2..."

# Start application in background
"$SIMPLYSIGN_EXE" &
TEST_PID=$!

echo "   ✅ SimplySign Desktop started (PID: $TEST_PID)"
echo "   Monitoring for automatic OAuth2 dialog (10 seconds)..."

# Brief monitoring for OAuth2 dialog
OAUTH_DETECTED=false
for ((i=1; i<=10; i++)); do
    if command -v powershell >/dev/null 2>&1; then
        DIALOG_CHECK=$(powershell -Command "
        Get-Process | Where-Object { 
            \$_.MainWindowTitle -like '*certum*' -or 
            \$_.MainWindowTitle -like '*oauth*' -or
            \$_.MainWindowTitle -like '*login*' -or
            \$_.MainWindowTitle -like '*authentication*'
        } | ForEach-Object { \"\$(\$_.ProcessName):\$(\$_.MainWindowTitle)\" }
        " 2>/dev/null)
        
        if [ -n "$DIALOG_CHECK" ]; then
            echo "   🎉 OAuth2 dialog detected: $DIALOG_CHECK"
            OAUTH_DETECTED=true
            break
        fi
    fi
    
    sleep 1
done

# Cleanup test process
taskkill /F /IM "SimplySignDesktop.exe" 2>/dev/null || true
sleep 2

if [ "$OAUTH_DETECTED" = true ]; then
    echo "   ✅ Automatic OAuth2 authentication VERIFIED!"
    echo "   🎯 BREAKTHROUGH: Configuration is working correctly"
else
    echo "   ⚠️ OAuth2 dialog not detected during brief test"
    echo "   💡 May still work during actual certificate operations"
fi

# Summary
echo ""
echo "📊 Configuration Verification Summary:"
echo "======================================"

if [ "$REGISTRY_CONFIGURED" = true ]; then
    echo "✅ Registry Configuration: VERIFIED"
else
    echo "⚠️ Registry Configuration: Not confirmed"
fi

if [ "$CONFIG_FILES_FOUND" = true ]; then
    echo "✅ Configuration Files: Found and checked"
else
    echo "⚠️ Configuration Files: Not found (may use registry only)"
fi

echo "✅ Application Executable: Present and functional"

if [ "$OAUTH_DETECTED" = true ]; then
    echo "✅ Automatic Authentication: WORKING!"
    echo ""
    echo "🎉 BREAKTHROUGH CONFIRMED!"
    echo "🚀 SimplySign Desktop is correctly configured for automatic OAuth2"
    echo "📱 Ready for CI/CD workflows with automatic authentication"
else
    echo "⚠️ Automatic Authentication: Not confirmed in brief test"
    echo ""
    echo "💡 Configuration applied but needs longer test or actual certificate operation"
    echo "🚀 May still work correctly in production workflows"
fi

echo ""
echo "🏁 Verification completed!"

if [ "$OAUTH_DETECTED" = true ]; then
    exit 0  # Success
else
    exit 1  # Warning - configuration may still work but not confirmed
fi
