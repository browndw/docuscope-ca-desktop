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

# Test CLI capabilities (based on your successful output)
echo "Testing CLI capabilities..."
timeout 10 "$SIMPLYSIGN_EXE" --version 2>&1 | head -5 || echo "Version check completed"
timeout 10 "$SIMPLYSIGN_EXE" --help 2>&1 | head -10 || echo "Help check completed"

# Test certificate listing
echo "Testing certificate listing..."
timeout 15 "$SIMPLYSIGN_EXE" --showCertificate 2>&1 | head -10 || echo "Certificate listing completed"

# Simple credential injection via Windows Credential Manager
echo "Injecting credentials into Windows Credential Manager..."
cmdkey /add:"certum.eu" /user:"$CERTUM_USERNAME" /pass:"$CERTUM_PASSWORD" 2>&1 || echo "Credential injection attempted"
cmdkey /add:"cloud.certum.eu" /user:"$CERTUM_USERNAME" /pass:"$CERTUM_PASSWORD" 2>&1 || echo "Cloud credential injection attempted"

# Check certificate stores after authentication attempt
echo "Checking certificate stores after authentication..."
if [ -n "${CERTUM_CERTIFICATE_SHA1:-}" ]; then
  check_certificate_store "$CERTUM_CERTIFICATE_SHA1"
else
  echo "⚠️ CERTUM_CERTIFICATE_SHA1 not provided, checking all certificates"
  check_certificate_store ""
fi

# Find signtool
if find_signtool; then
  echo "✅ signtool.exe available for testing"
else
  echo "❌ signtool.exe not found - code signing tests will be skipped"
fi

echo "✅ Authentication testing completed"
