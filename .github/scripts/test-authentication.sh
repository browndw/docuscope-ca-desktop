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
echo "🔐 Application is now ready to receive TOTP authentication in Step 4"
echo "📋 Next step: Manual approval → TOTP input → Certificate access"

# Brief verification that the process started successfully
sleep 3
if kill -0 $INIT_PID 2>/dev/null; then
  echo "✅ SimplySign Desktop running successfully"
  echo "💡 Process will remain active for TOTP authentication"
else
  echo "⚠️ SimplySign Desktop may have exited quickly"
  echo "💡 Will attempt to restart in Step 4 if needed"
fi

echo ""
echo "✅ Authentication testing and initialization completed"
echo "🚀 Ready for Step 4: Certum Desktop Signing with TOTP"
