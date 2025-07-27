#!/bin/bash

# Verify OAuth2 Dialog Detection
# Tests if OAuth2 dialog appears and can be detected in CI environment

set -euo pipefail

echo "=== Verifying OAuth2 Dialog Detection ==="

PACKAGE_DIR="simplysign-desktop-package"
SIMPLYSIGN_EXE="$PACKAGE_DIR/SimplySign Desktop/SimplySignDesktop.exe"

if [ ! -f "$SIMPLYSIGN_EXE" ]; then
  echo "❌ SimplySign Desktop not found at: $SIMPLYSIGN_EXE"
  exit 1
fi

# Check registry configuration
echo "🔍 Checking registry configuration..."
if command -v powershell >/dev/null 2>&1; then
    # Use PowerShell for reliable registry access
    AUTO_DIALOG=$(powershell -Command "
        try {
            \$value = Get-ItemProperty -Path 'HKCU:\\Software\\Certum\\SimplySignDesktop' -Name 'SimplySignDesktopShowLogonDialogAfterApplicationStartup' -ErrorAction SilentlyContinue
            if (\$value) { \$value.SimplySignDesktopShowLogonDialogAfterApplicationStartup } else { 'NotFound' }
        } catch { 'Error' }
    ")
    
    if [ "$AUTO_DIALOG" = "Yes" ]; then
        echo "✅ Auto OAuth2 dialog: $AUTO_DIALOG"
    else
        echo "⚠️ Auto OAuth2 dialog setting: $AUTO_DIALOG"
    fi
else
    echo "⚠️ PowerShell not available - cannot verify registry"
fi

# Check network connectivity to OAuth2 endpoints
echo "🌐 Checking OAuth2 endpoint connectivity..."
OAUTH_BASE="https://cloudsign.webnotarius.pl/idp/oauth2.0"

ENDPOINTS=(
    "$OAUTH_BASE/authorize"
    "$OAUTH_BASE/token"
    "$OAUTH_BASE/introspect"
)

for endpoint in "${ENDPOINTS[@]}"; do
    if curl -s --connect-timeout 10 --max-time 30 -I "$endpoint" > /dev/null 2>&1; then
        echo "   ✅ $endpoint - reachable"
    else
        echo "   ❌ $endpoint - not reachable"
    fi
done

# Test OAuth2 dialog detection in background
echo "🖥️ Testing OAuth2 dialog detection..."

# Install virtual display for headless testing
if ! command -v Xvfb >/dev/null 2>&1; then
    echo "   Installing virtual display support..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y xvfb
    elif command -v choco >/dev/null 2>&1; then
        # Windows - we'll use different approach
        echo "   Windows environment - using native window detection"
    fi
fi

# Create dialog detection script
DETECT_SCRIPT="detect_oauth2_dialog.sh"
cat > "$DETECT_SCRIPT" << 'EOF'
#!/bin/bash
# OAuth2 Dialog Detection Script

TIMEOUT=30
INTERVAL=2
ELAPSED=0

echo "🔍 Monitoring for OAuth2 dialog (timeout: ${TIMEOUT}s)..."

while [ $ELAPSED -lt $TIMEOUT ]; do
    # Windows: Check for OAuth2 dialog windows
    if command -v powershell >/dev/null 2>&1; then
        OAUTH_WINDOW=$(powershell -Command "
            Get-Process | Where-Object { \$_.MainWindowTitle -like '*OAuth*' -or \$_.MainWindowTitle -like '*Login*' -or \$_.MainWindowTitle -like '*Authentication*' } | Select-Object -First 1 -ExpandProperty MainWindowTitle
        " 2>/dev/null)
        
        if [ -n "$OAUTH_WINDOW" ] && [ "$OAUTH_WINDOW" != "" ]; then
            echo "✅ OAuth2 dialog detected: $OAUTH_WINDOW"
            exit 0
        fi
    fi
    
    # Linux/macOS: Check for X11 windows (if available)
    if command -v xwininfo >/dev/null 2>&1; then
        if xwininfo -root -tree 2>/dev/null | grep -i "oauth\|login\|authentication" > /dev/null; then
            echo "✅ OAuth2 dialog detected via X11"
            exit 0
        fi
    fi
    
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    echo "   Waiting... (${ELAPSED}s/${TIMEOUT}s)"
done

echo "⏰ Timeout reached - no OAuth2 dialog detected"
exit 1
EOF

chmod +x "$DETECT_SCRIPT"

# Start SimplySign Desktop in background and monitor for dialog
echo "🚀 Starting SimplySign Desktop..."

if [ -f "$SIMPLYSIGN_EXE" ]; then
    # Start SimplySign Desktop in background
    echo "   Launching: $SIMPLYSIGN_EXE"
    
    # Start dialog detection in background
    ./"$DETECT_SCRIPT" &
    DETECT_PID=$!
    
    # Start SimplySign Desktop
    if command -v powershell >/dev/null 2>&1; then
        # Windows: Start and detach
        powershell -Command "Start-Process '$(cygpath -w "$SIMPLYSIGN_EXE")' -WindowStyle Hidden" &
        SIMPLYSIGN_PID=$!
    else
        # Linux/macOS: Start with virtual display if available
        if command -v xvfb-run >/dev/null 2>&1; then
            xvfb-run -a "$SIMPLYSIGN_EXE" &
        else
            "$SIMPLYSIGN_EXE" &
        fi
        SIMPLYSIGN_PID=$!
    fi
    
    echo "   ✅ SimplySign Desktop started (PID: $SIMPLYSIGN_PID)"
    echo "   🔍 Dialog detection running (PID: $DETECT_PID)"
    
    # Wait for detection script to complete
    wait $DETECT_PID
    DETECT_RESULT=$?
    
    # Cleanup SimplySign Desktop process
    if kill -0 $SIMPLYSIGN_PID 2>/dev/null; then
        echo "   🧹 Stopping SimplySign Desktop..."
        kill $SIMPLYSIGN_PID 2>/dev/null || true
        sleep 2
        kill -9 $SIMPLYSIGN_PID 2>/dev/null || true
    fi
    
    # Report results
    if [ $DETECT_RESULT -eq 0 ]; then
        echo "✅ OAuth2 dialog detection: SUCCESS"
        echo "🎯 Dialog appeared automatically as expected"
    else
        echo "❌ OAuth2 dialog detection: FAILED"
        echo "⚠️ Dialog may not appear in headless CI environment"
    fi
    
else
    echo "❌ SimplySign Desktop executable not found"
    exit 1
fi

# Cleanup
rm -f "$DETECT_SCRIPT"

echo ""
echo "🏁 OAuth2 dialog verification complete"
echo "💡 Next step: Implement credential injection for detected dialog"
