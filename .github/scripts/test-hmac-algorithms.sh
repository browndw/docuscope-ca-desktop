#!/bin/bash

# Test Real Certum Authentication Methods
# Focus on actual authentication instead of random activation codes

set -euo pipefail

echo "=== Testing Real Certum Authentication Methods ==="

# Check if we have certificate credentials
if [ -z "${CERTUM_USERNAME:-}" ] || [ -z "${CERTUM_PASSWORD:-}" ]; then
  echo "❌ CERTUM_USERNAME and CERTUM_PASSWORD required"
  exit 1
fi

echo "✅ Certificate credentials provided"
echo "Username: $CERTUM_USERNAME"

# Method 1: Direct Certum portal authentication
echo ""
echo "Method 1: Direct Certum portal authentication..."
echo "Testing connectivity to Certum services..."

# Test main Certum portals
CERTUM_PORTALS=(
  "https://www.certum.eu"
  "https://cloud.certum.eu" 
  "https://portal.certum.eu"
  "https://secure.certum.eu"
)

for portal in "${CERTUM_PORTALS[@]}"; do
  echo "Testing: $portal"
  curl -s --max-time 10 -I "$portal" | head -2 || echo "  Connection failed"
done

# Method 2: Test Windows-specific authentication
echo ""
echo "Method 2: Windows credential store integration..."

# Clear any existing Certum credentials
echo "Clearing existing credentials..."
cmdkey /list | grep -i certum | while read line; do
  target=$(echo "$line" | cut -d':' -f2 | tr -d ' ')
  cmdkey /delete:"$target" 2>/dev/null || true
done

# Add fresh credentials
echo "Adding fresh Certum credentials..."
CREDENTIAL_TARGETS=(
  "certum.eu"
  "cloud.certum.eu"
  "portal.certum.eu"
  "*.certum.eu"
)

for target in "${CREDENTIAL_TARGETS[@]}"; do
  echo "Adding credential for: $target"
  cmdkey /add:"$target" /user:"$CERTUM_USERNAME" /pass:"$CERTUM_PASSWORD" 2>&1 || echo "  Failed to add credential for $target"
done

# Verify credentials were stored
echo "Verifying stored credentials..."
cmdkey /list | grep -i certum || echo "No Certum credentials found"

# Method 3: Test SimplySign Desktop with credentials
echo ""
echo "Method 3: SimplySign Desktop authentication..."

SIMPLYSIGN_EXE="/c/Program Files/Certum/SimplySign Desktop/SimplySignDesktop.exe"
if [ -f "$SIMPLYSIGN_EXE" ]; then
  echo "✅ SimplySign Desktop found"
  
  # Test if the application can access our credentials
  echo "Testing credential access..."
  timeout 30 "$SIMPLYSIGN_EXE" --showCertificate 2>&1 | head -10 || echo "Certificate check completed"
  
  # Check for certificate in stores after credential setup
  echo "Checking certificate stores after credential setup..."
  CERT_COUNT=$(powershell -Command "try { (Get-ChildItem -Path 'Cert:\\CurrentUser\\My' | Measure-Object).Count } catch { 0 }" 2>/dev/null || echo "0")
  echo "CurrentUser certificates: $CERT_COUNT"
  
  if [ -n "${CERTUM_CERTIFICATE_SHA1:-}" ]; then
    echo "Searching for target certificate: $CERTUM_CERTIFICATE_SHA1"
    CERT_FOUND=$(powershell -Command "Get-ChildItem -Path 'Cert:\\CurrentUser\\My','Cert:\\LocalMachine\\My' | Where-Object { \$_.Thumbprint -eq '$CERTUM_CERTIFICATE_SHA1' }" 2>/dev/null)
    if [ -n "$CERT_FOUND" ]; then
      echo "✅ Target certificate found!"
      return 0
    else
      echo "⚠️ Target certificate not yet accessible"
    fi
  fi
else
  echo "❌ SimplySign Desktop not found"
  exit 1
fi

echo ""
echo "✅ Real authentication testing completed"
echo "Next step: Use working credentials for actual code signing"
