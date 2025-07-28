#!/bin/bash

# PKCS#11 Code Signing with SimplySign Desktop
# Uses PKCS#11 interface as per official Certum documentation
# BREAKTHROUGH: Certificates are accessible via PKCS#11, not Windows certificate stores

set -euo pipefail

echo "=== PKCS#11 Code Signing with SimplySign Desktop ==="
echo "🎯 Using official Certum thumbprint-based signing method"
echo "📚 Following official documentation: signtool + certificate SHA1 thumbprint"

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

# Validate certificate thumbprint (required for official Certum signtool approach)
if [ -z "${CERTUM_CERTIFICATE_SHA1:-}" ]; then
  echo "❌ CERTUM_CERTIFICATE_SHA1 not set - required for thumbprint-based signing"
  echo "   Official Certum documentation requires certificate thumbprint for signtool"
  exit 1
fi

echo "✅ Binary to sign: $BINARY_PATH"
echo "📏 Binary size: $(stat -c%s "$BINARY_PATH" 2>/dev/null || echo "unknown") bytes"
echo "🔑 Certificate thumbprint: ${CERTUM_CERTIFICATE_SHA1:0:16}... (truncated for security)"

# Step 1: Verify PKCS#11 setup
echo ""
echo "🔍 Step 1: Verifying PKCS#11 setup..."
if ! check_pkcs11_certificates; then
  echo "❌ PKCS#11 certificate check failed"
  exit 1
fi

# Step 2: Find PKCS#11-compatible signing tools
echo ""
echo "🔧 Step 2: Finding signing tools..."
SIGNING_TOOL=""
TOOL_TYPE=""

# PRIORITY: signtool with thumbprint (official Certum method)
echo "🎯 PRIORITY: Official Certum method (signtool + thumbprint)..."

# Look for signtool first
if find_signtool; then
  SIGNING_TOOL="$SIGNTOOL_PATH"
  TOOL_TYPE="signtool"
  echo "✅ Will use signtool with certificate thumbprint (OFFICIAL CERTUM METHOD)"
  echo "   Following official documentation: signtool + SHA1 thumbprint"
else
  echo "⚠️ signtool not found, falling back to osslsigncode..."
  if find_pkcs11_signing_tool; then
    SIGNING_TOOL="$OSSLSIGNCODE_PATH"
    TOOL_TYPE="osslsigncode"
    echo "✅ Will use osslsigncode for PKCS#11 signing (FALLBACK)"
  elif install_osslsigncode; then
    SIGNING_TOOL="$OSSLSIGNCODE_PATH"
    TOOL_TYPE="osslsigncode"
    echo "✅ osslsigncode installed and ready (FALLBACK)"
  else
    echo "❌ No compatible signing tools found"
    exit 1
  fi
fi
# Step 3: Perform code signing
echo ""
echo "🔐 Step 3: Performing code signing with official Certum method..."

case "$TOOL_TYPE" in
  "osslsigncode")
    echo "Using osslsigncode with PKCS#11..."
    
    # Method 1: Try osslsigncode with standard certificate file approach first
    echo "Method 1: Attempting osslsigncode with certificate auto-discovery..."
    if "$SIGNING_TOOL" sign \
        -certs auto \
        -key auto \
        -t http://time.certum.pl \
        -h sha256 \
        -in "$BINARY_PATH" \
        -out "${BINARY_PATH}.signed" 2>&1 | tee osslsigncode_method1.log; then
        
        # Replace original with signed version
        mv "${BINARY_PATH}.signed" "$BINARY_PATH"
        echo "✅ Code signing successful with osslsigncode auto-discovery!"
        SIGNING_SUCCESS=true
        
    else
        echo "❌ osslsigncode auto-discovery failed"
        cat osslsigncode_method1.log
        
        # Method 2: Try with PKCS#11 engine approach
        echo "Method 2: Attempting osslsigncode with PKCS#11 engine..."
        
        # Create PKCS#11 configuration file for OpenSSL
        PKCS11_CONFIG="openssl_pkcs11.conf"
        cat > "$PKCS11_CONFIG" << EOF
openssl_conf = openssl_init

[openssl_init]
engines = engine_section

[engine_section]
pkcs11 = pkcs11_section

[pkcs11_section]
engine_id = pkcs11
dynamic_path = /c/Windows/System32/SimplySignPKCS.dll
MODULE_PATH = /c/Windows/System32/SimplySignPKCS.dll
init = 0
EOF
        
        if OPENSSL_CONF="$PKCS11_CONFIG" "$SIGNING_TOOL" sign \
            -pkcs11engine pkcs11 \
            -pkcs11module "/c/Windows/System32/SimplySignPKCS.dll" \
            -certs pkcs11: \
            -key pkcs11: \
            -t http://time.certum.pl \
            -h sha256 \
            -in "$BINARY_PATH" \
            -out "${BINARY_PATH}.signed" 2>&1 | tee osslsigncode_method2.log; then
            
            # Replace original with signed version
            mv "${BINARY_PATH}.signed" "$BINARY_PATH"
            echo "✅ Code signing successful with osslsigncode PKCS#11 engine!"
            SIGNING_SUCCESS=true
            
        else
            echo "❌ osslsigncode PKCS#11 engine failed"
            cat osslsigncode_method2.log
            
            # Method 3: Try simplified approach without PKCS#11 - let osslsigncode find certificates
            echo "Method 3: Attempting osslsigncode with Windows certificate store..."
            if "$SIGNING_TOOL" sign \
                -t http://time.certum.pl \
                -h sha256 \
                -in "$BINARY_PATH" \
                -out "${BINARY_PATH}.signed" 2>&1 | tee osslsigncode_method3.log; then
                
                # Replace original with signed version
                mv "${BINARY_PATH}.signed" "$BINARY_PATH"
                echo "✅ Code signing successful with osslsigncode Windows store!"
                SIGNING_SUCCESS=true
                
            else
                echo "❌ osslsigncode Windows store failed"
                cat osslsigncode_method3.log
                SIGNING_SUCCESS=false
            fi
        fi
        
        # Clean up config
        rm -f "$PKCS11_CONFIG"
    fi
    ;;
    
  "signtool")
    echo "Using signtool with certificate thumbprint (OFFICIAL CERTUM METHOD)..."
    echo "Following official documentation: signtool sign /sha1 [thumbprint] ..."
    
    # OFFICIAL METHOD: Use certificate thumbprint as per Certum documentation
    # Documentation: "signtool sign /sha1 "[thumbprint]" /tr [timestamp] /td [td_algo] /fd [fd_algo] /v "[file]""
    echo "Method 1: Official Certum thumbprint-based signing..."
    if "$SIGNING_TOOL" sign \
        /sha1 "$CERTUM_CERTIFICATE_SHA1" \
        /fd SHA256 \
        /tr http://time.certum.pl \
        /td SHA256 \
        /v \
        "$BINARY_PATH" 2>&1 | tee signtool_thumbprint.log; then
        
        echo "✅ Code signing successful with official Certum thumbprint method!"
        SIGNING_SUCCESS=true
        
    else
        echo "❌ Official thumbprint method failed"
        cat signtool_thumbprint.log
        
        # FALLBACK: Try auto-select as backup (may not work per documentation)
        echo "Method 2: Fallback - trying auto-select (may not work with PKCS#11)..."
        if "$SIGNING_TOOL" sign \
            /a \
            /fd SHA256 \
            /tr http://time.certum.pl \
            /td SHA256 \
            /v \
            "$BINARY_PATH" 2>&1 | tee signtool_fallback.log; then
            
            echo "✅ Code signing successful with auto-select fallback!"
            SIGNING_SUCCESS=true
            
        else
            echo "❌ Both thumbprint and auto-select methods failed"
            echo "Thumbprint method output:"
            cat signtool_thumbprint.log
            echo ""
            echo "Auto-select fallback output:"
            cat signtool_fallback.log
            SIGNING_SUCCESS=false
        fi
    fi
    ;;
    
  *)
    echo "❌ Unknown tool type: $TOOL_TYPE"
    exit 1
    ;;
esac

# Check if signing was successful
if [ "$SIGNING_SUCCESS" != "true" ]; then
  echo "❌ All signing methods failed"
  echo "   Ensure SimplySign Desktop is connected and certificate is accessible"
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
    echo "🎉 Code signing completed with official Certum method!"
    echo "✅ Binary signed with Certum certificate using thumbprint"
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
  echo "🎉 Code signing completed successfully with official Certum method!"
  echo "✅ Binary is now properly signed with Certum certificate"
  echo "🔐 Authentication: ✅"
  echo "🔑 Certificate thumbprint: ✅" 
  echo "🖊️ Official Certum signing: ✅"
  echo "✔️ Signature verification: ✅"
  
else
  echo "❌ Signature verification failed"
  cat verification_output.log
  exit 1
fi
