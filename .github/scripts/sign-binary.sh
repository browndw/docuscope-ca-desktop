#!/bin/bash

# PKCS#11 Code Signing with SimplySign Desktop
# Uses PKCS#11 interface as per official Certum documentation
# BREAKTHROUGH: Certificates are accessible via PKCS#11, not Windows certificate stores

set -euo pipefail

echo "=== PKCS#11 Code Signing with SimplySign Desktop ==="
echo "🎯 Using PKCS#11 interface to access Certum cloud certificates"

# Source certificate utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/certificate-utils.sh"

# Check required variables
if [ -z "${BINARY_PATH:-}" ]; then
  echo "❌ BINARY_PATH not set - no binary to sign"
  exit 1
fi

if [ ! -f "$BINARY_PATH" ]; then
  echo "❌ Binary not found at: $BINARY_PATH"
  exit 1
fi

echo "✅ Binary to sign: $BINARY_PATH"
echo "📏 Binary size: $(stat -c%s "$BINARY_PATH" 2>/dev/null || echo "unknown") bytes"

# Step 1: Verify PKCS#11 setup
echo ""
echo "🔍 Step 1: Verifying PKCS#11 setup..."
if ! check_pkcs11_certificates; then
  echo "❌ PKCS#11 certificate check failed"
  exit 1
fi

# Step 2: Find PKCS#11-compatible signing tools
echo ""
echo "🔧 Step 2: Finding PKCS#11-compatible signing tools..."
SIGNING_TOOL=""
TOOL_TYPE=""

# Try osslsigncode first (best PKCS#11 support)
if find_pkcs11_signing_tool; then
  SIGNING_TOOL="$OSSLSIGNCODE_PATH"
  TOOL_TYPE="osslsigncode"
  echo "✅ Will use osslsigncode for PKCS#11 signing"
elif install_osslsigncode; then
  SIGNING_TOOL="$OSSLSIGNCODE_PATH"
  TOOL_TYPE="osslsigncode"
  echo "✅ osslsigncode installed and ready"
else
  # Fallback to signtool with smart card detection
  echo "⚠️ osslsigncode not available, trying signtool fallback..."
  if find_signtool; then
    SIGNING_TOOL="$SIGNTOOL_PATH"
    TOOL_TYPE="signtool"
    echo "✅ Will use signtool with smart card auto-select"
  else
    echo "❌ No compatible signing tools found"
    exit 1
  fi
fi
# Step 3: Perform code signing
echo ""
echo "🔐 Step 3: Performing PKCS#11 code signing..."

case "$TOOL_TYPE" in
  "osslsigncode")
    echo "Using osslsigncode with PKCS#11..."
    
    # Create PKCS#11 configuration
    PKCS11_CONFIG="pkcs11_signing.conf"
    cat > "$PKCS11_CONFIG" << EOF
name=SimplySignPKCS
library=/c/Windows/System32/SimplySignPKCS.dll
slotListIndex=0
EOF
    
    echo "✅ Created PKCS#11 configuration: $PKCS11_CONFIG"
    
    # Sign with osslsigncode + PKCS#11
    echo "Executing osslsigncode with PKCS#11 interface..."
    if "$SIGNING_TOOL" sign \
        -pkcs11engine "/c/Windows/System32/SimplySignPKCS.dll" \
        -pkcs11module "/c/Windows/System32/SimplySignPKCS.dll" \
        -ts http://time.certum.pl \
        -h sha256 \
        -in "$BINARY_PATH" \
        -out "${BINARY_PATH}.signed" 2>&1 | tee osslsigncode_output.log; then
        
        # Replace original with signed version
        mv "${BINARY_PATH}.signed" "$BINARY_PATH"
        echo "✅ Code signing successful with osslsigncode + PKCS#11!"
        SIGNING_SUCCESS=true
        
    else
        echo "❌ osslsigncode + PKCS#11 signing failed"
        cat osslsigncode_output.log
        SIGNING_SUCCESS=false
    fi
    
    # Clean up config
    rm -f "$PKCS11_CONFIG"
    ;;
    
  "signtool")
    echo "Using signtool with smart card auto-select..."
    
    # Use signtool with auto-select (/a) to find PKCS#11 certificates
    echo "Executing signtool with smart card detection..."
    if "$SIGNING_TOOL" sign \
        /a \
        /fd SHA256 \
        /tr http://time.certum.pl \
        /td SHA256 \
        /v \
        "$BINARY_PATH" 2>&1 | tee signtool_output.log; then
        
        echo "✅ Code signing successful with signtool auto-select!"
        SIGNING_SUCCESS=true
        
    else
        echo "❌ Signtool auto-select signing failed"
        cat signtool_output.log
        SIGNING_SUCCESS=false
    fi
    ;;
    
  *)
    echo "❌ Unknown tool type: $TOOL_TYPE"
    exit 1
    ;;
esac

# Check if signing was successful
if [ "$SIGNING_SUCCESS" != "true" ]; then
  echo "❌ All PKCS#11 signing methods failed"
  exit 1
fi

# Step 4: Verify the signature
echo ""
echo "🔍 Step 4: Verifying signature..."

# Find signtool for verification (if not already found)
if [ "$TOOL_TYPE" != "signtool" ]; then
  if ! find_signtool; then
    echo "⚠️ Cannot verify signature - signtool not available"
    echo "   Assuming signing was successful since no errors occurred"
    echo ""
    echo "🎉 PKCS#11 Code signing completed!"
    echo "✅ Binary signed with Certum certificate via PKCS#11"
    exit 0
  fi
fi

# Use signtool to verify
VERIFY_TOOL="$SIGNTOOL_PATH"
if [ "$TOOL_TYPE" == "signtool" ]; then
  VERIFY_TOOL="$SIGNING_TOOL"
fi

echo "Verifying signature with: $VERIFY_TOOL"
if "$VERIFY_TOOL" verify /pa /v "$BINARY_PATH" 2>&1 | tee verification_output.log; then
  echo "✅ Signature verification successful!"
  
  # Show certificate details
  echo ""
  echo "📋 Signature details:"
  "$VERIFY_TOOL" verify /pa /all "$BINARY_PATH" 2>/dev/null || echo "Could not get detailed signature info"
  
  echo ""
  echo "🎉 PKCS#11 Code signing completed successfully!"
  echo "✅ Binary is now properly signed with Certum certificate via PKCS#11"
  echo "🔐 Authentication: ✅"
  echo "☁️ Certificate access: ✅" 
  echo "🖊️ Code signing: ✅"
  echo "✔️ Signature verification: ✅"
  
else
  echo "❌ Signature verification failed"
  cat verification_output.log
  exit 1
fi
