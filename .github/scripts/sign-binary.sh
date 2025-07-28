#!/bin/bash

# Final Code Signing with SimplySign Desktop
# Clean signing implementation using discovered certificate and HMAC

set -euo pipefail

# Source utilities
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/utils/certificate-utils.sh"

echo "=== Final Code Signing ==="

# Check required variables
if [ -z "${BINARY_PATH:-}" ]; then
  echo "❌ BINARY_PATH not set - no binary to sign"
  exit 1
fi

if [ ! -f "$BINARY_PATH" ]; then
  echo "❌ Binary not found at: $BINARY_PATH"
  exit 1
fi

if [ -z "${CERTUM_CERTIFICATE_SHA1:-}" ]; then
  echo "❌ CERTUM_CERTIFICATE_SHA1 not provided"
  exit 1
fi

echo "✅ Binary to sign: $BINARY_PATH"
echo "✅ Certificate SHA1: $CERTUM_CERTIFICATE_SHA1"

# Find signtool (avoiding problematic SDK versions)
echo "🔧 Finding working signtool.exe (avoiding 10.0.22621.0 due to /fd parameter regression)"
if ! find_signtool; then
  echo "❌ signtool.exe not found"
  exit 1
fi

SIGNTOOL_PATH=$(grep "SIGNTOOL_PATH=" "$GITHUB_OUTPUT" | cut -d'=' -f2)

# Check certificate availability
if ! check_certificate_store "$CERTUM_CERTIFICATE_SHA1"; then
  echo "❌ Certificate not found in certificate store"
  exit 1
fi

# Perform code signing
echo "Signing binary with Certum certificate..."

# Try multiple certificate store locations for cloud certificates
echo "Attempting signing with various certificate store parameters..."

# Method 1: Default (Current User Personal store)
echo "Method 1: Default certificate lookup"
echo "Command: $SIGNTOOL_PATH sign /sha1 $CERTUM_CERTIFICATE_SHA1 /fd SHA256 /tr http://time.certum.pl /td SHA256 $BINARY_PATH"
if "$SIGNTOOL_PATH" sign /sha1 "$CERTUM_CERTIFICATE_SHA1" /fd SHA256 /tr http://time.certum.pl /td SHA256 "$BINARY_PATH" 2>/dev/null; then
  echo "✅ Code signing successful with default method!"
else
  echo "Method 1 failed, trying Method 2..."
  
  # Method 2: Explicitly specify Current User MY store
  echo "Method 2: Current User Personal (MY) store"
  echo "Command: $SIGNTOOL_PATH sign /sha1 $CERTUM_CERTIFICATE_SHA1 /s MY /fd SHA256 /tr http://time.certum.pl /td SHA256 $BINARY_PATH"
  if "$SIGNTOOL_PATH" sign /sha1 "$CERTUM_CERTIFICATE_SHA1" /s MY /fd SHA256 /tr http://time.certum.pl /td SHA256 "$BINARY_PATH" 2>/dev/null; then
    echo "✅ Code signing successful with MY store!"
  else
    echo "Method 2 failed, trying Method 3..."
    
    # Method 3: Local Machine Personal store
    echo "Method 3: Local Machine Personal (MY) store"
    echo "Command: $SIGNTOOL_PATH sign /sha1 $CERTUM_CERTIFICATE_SHA1 /s MY /sm /fd SHA256 /tr http://time.certum.pl /td SHA256 $BINARY_PATH"
    if "$SIGNTOOL_PATH" sign /sha1 "$CERTUM_CERTIFICATE_SHA1" /s MY /sm /fd SHA256 /tr http://time.certum.pl /td SHA256 "$BINARY_PATH" 2>/dev/null; then
      echo "✅ Code signing successful with Local Machine MY store!"
    else
      echo "Method 3 failed, trying Method 4..."
      
      # Method 4: Try ROOT store (sometimes cloud certs end up here)
      echo "Method 4: ROOT certificate store"
      echo "Command: $SIGNTOOL_PATH sign /sha1 $CERTUM_CERTIFICATE_SHA1 /s ROOT /fd SHA256 /tr http://time.certum.pl /td SHA256 $BINARY_PATH"
      if "$SIGNTOOL_PATH" sign /sha1 "$CERTUM_CERTIFICATE_SHA1" /s ROOT /fd SHA256 /tr http://time.certum.pl /td SHA256 "$BINARY_PATH" 2>/dev/null; then
        echo "✅ Code signing successful with ROOT store!"
      else
        echo "❌ All certificate store methods failed"
        echo "Certificate may not be accessible or SHA1 may be incorrect"
        exit 1
      fi
    fi
  fi
fi

# Verify the signature (common for all successful signing methods)
echo "Verifying signature..."
if "$SIGNTOOL_PATH" verify /pa "$BINARY_PATH"; then
  echo "✅ Signature verification successful!"
  
  # Get signature info
  echo "Signature details:"
  "$SIGNTOOL_PATH" verify /pa /v "$BINARY_PATH" | head -20
    
    echo ""
    echo "🎉 Code signing completed successfully!"
    echo "✅ Binary is now properly signed with Certum certificate"
    
  else
    echo "❌ Signature verification failed"
    exit 1
  fi
else
  echo "❌ Code signing failed"
  exit 1
fi
