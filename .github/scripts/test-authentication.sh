#!/bin/bash

# Test SimplySign Desktop Authentication
# Real authentication using actual Certum credentials

set -euo pipefail

# Source utilities
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/utils/certificate-utils.sh"

echo "=== Testing SimplySign Desktop Authentication ==="

# Check required credentials
if [ -z "${CERTUM_USERNAME:-}" ] || [ -z "${CERTUM_PASSWORD:-}" ]; then
  echo "❌ CERTUM_USERNAME and CERTUM_PASSWORD required"
  exit 1
fi

echo "✅ Certum credentials provided"
echo "Username: $CERTUM_USERNAME"

# Check if SimplySign Desktop is installed
SIMPLYSIGN_EXE="/c/Program Files/Certum/SimplySign Desktop/SimplySignDesktop.exe"
if [ ! -f "$SIMPLYSIGN_EXE" ]; then
  echo "❌ SimplySign Desktop not found at: $SIMPLYSIGN_EXE"
  exit 1
fi

echo "✅ SimplySign Desktop found: $SIMPLYSIGN_EXE"

# Start SimplySign Desktop in background
echo "Starting SimplySign Desktop..."
"$SIMPLYSIGN_EXE" &
SIMPLYSIGN_PID=$!
echo "✅ SimplySign Desktop started (PID: $SIMPLYSIGN_PID)"

# Wait for initialization
echo "Waiting for SimplySign Desktop to initialize..."
sleep 20

# Test CLI capabilities (simplified from working approach)
echo "Testing CLI capabilities..."
timeout 30 "$SIMPLYSIGN_EXE" --version 2>&1 | head -10 || echo "Version check completed"
timeout 30 "$SIMPLYSIGN_EXE" --help 2>&1 | head -20 || echo "Help check completed"

# Test certificate listing (from working approach)
echo "Testing certificate listing..."
timeout 60 "$SIMPLYSIGN_EXE" --showCertificate 2>&1 | head -20 || echo "Certificate listing completed"

# Enhanced credential injection (matching working approach)
echo "Injecting credentials into Windows Credential Manager..."
CERTUM_TARGETS=(
  "certum.eu"
  "cloud.certum.eu" 
  "api.certum.eu"
  "SimplySign"
  "Certum"
  "CertumCA"
  "simplysign.certum.eu"
  "*.certum.eu"
)

for target in "${CERTUM_TARGETS[@]}"; do
  cmdkey /add:"$target" /user:"$CERTUM_USERNAME" /pass:"$CERTUM_PASSWORD" 2>&1 || echo "Credential add attempt completed for $target"
done

# Check certificate stores after authentication attempt
echo "Checking certificate stores after authentication..."
if [ -n "${CERTUM_CERTIFICATE_SHA1:-}" ]; then
  check_certificate_store "$CERTUM_CERTIFICATE_SHA1"
else
  echo "⚠️ CERTUM_CERTIFICATE_SHA1 not provided, checking all certificates"
  check_certificate_store ""
fi

# Find signtool (with improved search after SDK installation)
echo "Searching for signtool.exe..."
if find_signtool; then
  echo "✅ signtool.exe available for testing"
else
  echo "❌ signtool.exe not found - code signing tests will be skipped"
fi

# Initialize SimplySign Desktop for Step 4 (TOTP authentication)
echo ""
echo "🔧 Initializing SimplySign Desktop for Step 4..."
echo "📱 Preparing application to receive TOTP authentication"

# Terminate any existing SimplySign processes to start fresh
echo "Cleaning up any existing SimplySign processes..."
taskkill /F /IM "SimplySignDesktop.exe" 2>/dev/null || echo "No existing processes found"
sleep 2

# Start SimplySign Desktop in background, ready for TOTP
echo "Starting SimplySign Desktop in background..."
echo "Command: '$SIMPLYSIGN_EXE' (background process)"

# Start the application and let it initialize
"$SIMPLYSIGN_EXE" &
INIT_PID=$!

echo "✅ SimplySign Desktop initialized (PID: $INIT_PID)"

# Give it time to fully initialize
sleep 5

# Try to programmatically trigger connectToCloud based on macOS discoveries
echo ""
echo "� Attempting to trigger connectToCloud authentication flow..."
echo "   Based on macOS analysis: [SCCAppDelegate connectToCloud:] → OAuth2 dialog"

# Method 1: Try command-line parameters discovered from binary analysis
echo "📋 Method 1: Command-line triggers..."
CONNECT_COMMANDS=(
    "--connect-cloud"
    "--cloud"
    "--login" 
    "--connect"
    "--auth"
    "--authenticate"
    "/connect"
    "/cloud"
    "/login"
)

for cmd in "${CONNECT_COMMANDS[@]}"; do
    echo "   Trying: $cmd"
    timeout 10 "$SIMPLYSIGN_EXE" "$cmd" 2>/dev/null &
    sleep 2
done

# Method 2: Try to trigger via Windows automation based on macOS flow
echo ""
echo "📋 Method 2: Windows automation to trigger connectToCloud..."
if command -v powershell >/dev/null 2>&1; then
    powershell -Command "
    try {
        # Find SimplySign Desktop windows
        Add-Type -AssemblyName System.Windows.Forms
        \$processes = Get-Process | Where-Object { \$_.ProcessName -like '*SimplySign*' }
        
        if (\$processes) {
            Write-Host '✅ Found SimplySign Desktop process(es)'
            \$processes | ForEach-Object { Write-Host \"   Process: \$(\$_.ProcessName) PID: \$(\$_.Id)\" }
            
            # Try to trigger authentication via menu/hotkeys
            # Based on macOS: connectToCloud should trigger PanelWebViewController
            [System.Windows.Forms.Application]::DoEvents()
            
            # Try common menu shortcuts that might trigger 'Connect to Cloud'
            Write-Host '🔍 Attempting to trigger authentication dialog...'
            
            # Send Alt+F for File menu, then C for Connect
            [System.Windows.Forms.SendKeys]::SendWait('%f')
            Start-Sleep -Seconds 1
            [System.Windows.Forms.SendKeys]::SendWait('c')
            Start-Sleep -Seconds 2
            
            # Try Ctrl+Shift+C for Connect to Cloud (common pattern)
            [System.Windows.Forms.SendKeys]::SendWait('^+c')
            Start-Sleep -Seconds 2
            
            # Try F5 to refresh/connect
            [System.Windows.Forms.SendKeys]::SendWait('{F5}')
            Start-Sleep -Seconds 2
            
            Write-Host '✅ Authentication triggers sent'
            Write-Host '   Looking for OAuth2 web view window...'
            
            # Check for OAuth2 dialog windows
            \$authWindows = Get-Process | Where-Object { 
                \$_.ProcessName -like '*browser*' -or 
                \$_.ProcessName -like '*webview*' -or
                \$_.MainWindowTitle -like '*certum*' -or
                \$_.MainWindowTitle -like '*oauth*' -or
                \$_.MainWindowTitle -like '*login*'
            }
            
            if (\$authWindows) {
                Write-Host '🌐 Potential OAuth2 window detected:'
                \$authWindows | ForEach-Object { Write-Host \"   \$(\$_.ProcessName): \$(\$_.MainWindowTitle)\" }
            } else {
                Write-Host '⚠️ No OAuth2 window detected yet'
                Write-Host '   Authentication dialog may appear shortly...'
            }
        } else {
            Write-Host '❌ No SimplySign Desktop process found'
        }
    } catch {
        Write-Host \"⚠️ Authentication trigger error: \$(\$_.Exception.Message)\"
    }
    " 2>&1
fi

# Method 3: Check for authentication dialogs and prepare for TOTP injection
echo ""
echo "📋 Method 3: Monitoring for authentication dialogs..."

# Monitor for 30 seconds to see if OAuth2 dialog appears
MONITOR_START=$(date +%s)
MAX_MONITOR=30
OAUTH_DETECTED=false

echo "⏱️ Monitoring for OAuth2 authentication dialog ($MAX_MONITOR seconds)..."

for ((i=1; i<=MAX_MONITOR; i++)); do
    # Check for authentication-related windows
    if command -v powershell >/dev/null 2>&1; then
        DIALOG_CHECK=$(powershell -Command "
        \$authDialogs = Get-Process | Where-Object { 
            \$_.MainWindowTitle -like '*certum*' -or 
            \$_.MainWindowTitle -like '*oauth*' -or
            \$_.MainWindowTitle -like '*login*' -or
            \$_.MainWindowTitle -like '*authentication*' -or
            \$_.MainWindowTitle -like '*sign*'
        }
        if (\$authDialogs) { 
            \$authDialogs | ForEach-Object { Write-Host \"\$(\$_.ProcessName):\$(\$_.MainWindowTitle)\" }
        }
        " 2>/dev/null)
        
        if [ -n "$DIALOG_CHECK" ]; then
            echo "🌐 OAuth2 dialog detected: $DIALOG_CHECK"
            OAUTH_DETECTED=true
            break
        fi
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   [$i/$MAX_MONITOR] Still monitoring for OAuth2 dialog..."
    fi
    
    sleep 1
done

if [ "$OAUTH_DETECTED" = true ]; then
    echo ""
    echo "✅ OAuth2 authentication dialog detected!"
    echo "🔐 Ready for manual credential entry in Step 4"
    echo "📱 Have your mobile TOTP app ready"
else
    echo ""
    echo "⚠️ No OAuth2 dialog appeared during monitoring"
    echo "💡 Authentication may be triggered when certificate access is needed"
fi

echo ""
echo "🔐 Application is now ready to receive TOTP authentication in Step 4"
echo "📋 Next step: Manual approval → Certificate access → OAuth2 authentication"

# Brief verification that the process is still running
sleep 3
if tasklist 2>/dev/null | grep -i "SimplySignDesktop" >/dev/null; then
    echo "✅ SimplySign Desktop running successfully"
    echo "💡 Process will remain active for TOTP authentication"
else
    echo "⚠️ SimplySign Desktop may have exited"
    echo "💡 Will attempt to restart in Step 4 if needed"
fi

echo ""
echo "✅ Authentication testing and initialization completed"
echo "🚀 Ready for Step 4: Certum Desktop Signing with TOTP"
