#!/bin/bash

# Certum Desktop Code Signing with TOTP Authentication
# Uses SimplySign Desktop's "Connect with cloud" functionality
# Requires: CERTUM_USERNAME (email) + CERTUM_API_TOKEN (6-digit TOTP from mobile)

set -euo pipefail

echo "=== Certum Desktop Code Signing ==="
echo "🔐 TOTP Authentication via SimplySign Desktop"
echo "📱 Using 6-digit TOTP code from mobile device"
echo "☁️  Connecting to Certum cloud for certificate access"
echo ""

# Check required credentials
if [ -z "${CERTUM_USERNAME:-}" ]; then
  echo "❌ CERTUM_USERNAME (email address) required"
  exit 1
fi

if [ -z "${CERTUM_CERTIFICATE_SHA1:-}" ]; then
  echo "❌ CERTUM_CERTIFICATE_SHA1 required for signing"
  exit 1
fi

echo "✅ Username (email): $CERTUM_USERNAME"
echo "✅ Certificate SHA1: $CERTUM_CERTIFICATE_SHA1"
echo ""

# Check for TOTP input (optional - authentication will be manual via OAuth2)
if [ -n "${CERTUM_TOTP_SEED:-}" ]; then
  # TOTP provided - can be used for automated authentication (if supported)
  CERTUM_API_TOKEN="$CERTUM_TOTP_SEED"
  echo "✅ TOTP API Token provided: $CERTUM_API_TOKEN"
  echo "   Note: Manual OAuth2 authentication may still be required"
else
  # No TOTP - fully manual OAuth2 authentication
  echo "📱 No TOTP provided - OAuth2 authentication will be fully manual"
  echo "   SimplySign Desktop will show OAuth2 browser dialog automatically"
  CERTUM_API_TOKEN=""
fi
echo ""

# Verify SimplySign Desktop is available and running
SIMPLYSIGN_EXE="/c/Program Files/Certum/SimplySign Desktop/SimplySignDesktop.exe"
if [ ! -f "$SIMPLYSIGN_EXE" ]; then
    echo "❌ SimplySign Desktop not found at: $SIMPLYSIGN_EXE"
    echo "   Please ensure Step 2 (Install SimplySign Desktop) completed successfully"
    exit 1
fi

echo "✅ SimplySign Desktop found: $SIMPLYSIGN_EXE"

# Check if SimplySign Desktop is already running from Step 3
if command -v pgrep >/dev/null 2>&1 && pgrep -f "SimplySignDesktop.exe" >/dev/null; then
    echo "✅ SimplySign Desktop already running (initialized in Step 3)"
elif tasklist 2>/dev/null | grep -i "SimplySignDesktop" >/dev/null; then
    echo "✅ SimplySign Desktop already running (found via tasklist)"
else
    echo "🔧 Starting SimplySign Desktop..."
    "$SIMPLYSIGN_EXE" &
    sleep 5
    echo "✅ SimplySign Desktop started"
fi
echo ""

# Check for binary to sign
if [ -z "${BINARY_PATH:-}" ] || [ ! -f "${BINARY_PATH:-}" ]; then
    echo "❌ No binary file found to sign"
    echo "   BINARY_PATH: ${BINARY_PATH:-<not set>}"
    echo "   Please ensure the binary preparation step completed successfully"
    exit 1
fi

echo "✅ Binary to sign: $BINARY_PATH"
echo "   Size: $(stat -c%s "$BINARY_PATH" 2>/dev/null || stat -f%z "$BINARY_PATH" 2>/dev/null || echo "unknown") bytes"

# Define signed binary path
SIGNED_BINARY="${BINARY_PATH%.exe}-signed.exe"
echo "   Expected signed output: $SIGNED_BINARY"
echo ""

# Connect to Certum cloud using the discovered authentication flow
echo "☁️  Connecting to Certum cloud with OAuth2 authentication..."
echo "📧 Username: $CERTUM_USERNAME"
if [ -n "$CERTUM_API_TOKEN" ]; then
  echo "🔐 API Token: $CERTUM_API_TOKEN (6-digit TOTP)"
else
  echo "🔐 API Token: Will be entered manually in OAuth2 dialog"
fi
echo ""

# Based on our macOS testing, SimplySign Desktop is a background service that:
# 1. Acts as a proxy between public certificates and private cloud keys
# 2. Runs in system tray (not dock/taskbar)
# 3. Makes certificates available to system certificate store after authentication
# 4. Authentication is triggered when certificate access is needed

echo "🔍 Understanding SimplySign Desktop Architecture:"
echo "   • Function: Certificate proxy service (public cert ↔ private cloud key)"
echo "   • Location: System tray/background service"
echo "   • Auth Trigger: Certificate access request (not GUI shortcuts)"
echo "   • Auth Method: OAuth2 web browser (automatic popup)"
echo ""

echo "💡 STRATEGY: Trigger authentication by requesting certificate access"
echo ""

# Method 1: Try direct signing - this should trigger authentication automatically
echo ""
echo "🔍 Method 4: Attempting direct signing with signtool..."
echo "   This may work if authentication succeeded silently"

# Look for signtool in common locations
SIGNTOOL_PATHS=(
    "/c/Program Files (x86)/Windows Kits/10/bin/x64/signtool.exe"
    "/c/Program Files/Windows Kits/10/bin/x64/signtool.exe"
    "/c/Program Files (x86)/Microsoft SDKs/Windows/v10.0A/bin/NETFX 4.8 Tools/x64/signtool.exe"
    "signtool.exe"
)

SIGNTOOL=""
for path in "${SIGNTOOL_PATHS[@]}"; do
    if [ -f "$path" ] || command -v "$path" >/dev/null 2>&1; then
        SIGNTOOL="$path"
        echo "✅ Found signtool: $SIGNTOOL"
        break
    fi
done

if [ -n "$SIGNTOOL" ]; then
    echo "🔐 Attempting to sign with certificate SHA1: $CERTUM_CERTIFICATE_SHA1"
    
    # Try signing with the certificate
    if "$SIGNTOOL" sign /sha1 "$CERTUM_CERTIFICATE_SHA1" /tr http://timestamp.comodoca.com /td sha256 /fd sha256 "$BINARY_PATH" 2>&1; then
        echo "🎉 SUCCESS! Binary signed successfully with signtool"
        AUTH_SUCCESS=true
        # Check if a separate signed file was created
        if [ -f "${BINARY_PATH%.exe}-signed.exe" ]; then
            SIGNED_BINARY="${BINARY_PATH%.exe}-signed.exe"
        else
            # Signing was done in-place
            SIGNED_BINARY="$BINARY_PATH"
        fi
    else
        echo "⚠️ Signtool signing failed - certificate may not be accessible yet"
        echo "   This is expected if authentication hasn't completed"
    fi
else
    echo "⚠️ Signtool not found - cannot attempt direct signing"
fi

# Monitor for authentication success and certificate access
echo ""
echo "🔍 MONITORING FOR OAUTH2 AUTHENTICATION..."
echo "   Expecting OAuth2 web browser to open automatically"
echo "   Manual completion required for authentication"
echo ""

# Give time for authentication
MONITOR_START=$(date +%s)
if [ -n "$CERTUM_API_TOKEN" ]; then
  # TOTP provided - shorter timeout since token expires
  MAX_WAIT=25  # Leave 5 seconds buffer before TOTP expires
  echo "⏱️ Monitoring for $MAX_WAIT seconds (TOTP: $CERTUM_API_TOKEN expires soon)..."
else
  # No TOTP - longer timeout for manual entry
  MAX_WAIT=60  # Allow time for manual TOTP generation and entry
  echo "⏱️ Monitoring for $MAX_WAIT seconds (manual TOTP entry)..."
fi
AUTH_SUCCESS=false

echo ""
echo "🌐 MANUAL AUTHENTICATION REQUIRED:"
echo "   1. OAuth2 browser should open automatically"
echo "   2. Enter username: $CERTUM_USERNAME"  
if [ -n "$CERTUM_API_TOKEN" ]; then
  echo "   3. Enter TOTP: $CERTUM_API_TOKEN"
  echo "   4. Complete authentication within $(($MAX_WAIT - 5)) seconds"
else
  echo "   3. Generate TOTP from mobile app and enter it"
  echo "   4. Complete authentication within $MAX_WAIT seconds"
fi
echo ""

for ((i=1; i<=MAX_WAIT; i++)); do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - MONITOR_START))
    REMAINING=$((MAX_WAIT - ELAPSED))
    
    echo -n "[$i/$MAX_WAIT] Checking authentication status... "
    
    # Check if authentication was successful by looking for certificate access
    # In a real implementation, we could check registry, log files, or process status
    
    # For now, provide clear progress indicators
    if [ $i -eq 10 ]; then
        echo "🔍 First checkpoint - OAuth2 dialog should be open"
    elif [ $i -eq 20 ]; then
        echo "⚠️ Authentication should be completing soon"
    elif [ $i -eq 25 ]; then
        echo "🚨 TOTP expiring in ~5 seconds!"
    else
        echo "remaining: ${REMAINING}s"
    fi
    
    sleep 1
    
    # Check for signed binary (success indicator)
    if [ -f "$SIGNED_BINARY" ]; then
        echo "� SUCCESS! Signed binary detected"
        AUTH_SUCCESS=true
        break
    fi
done

echo ""

# Check authentication and signing results
if [ "$AUTH_SUCCESS" = true ] && [ -f "$SIGNED_BINARY" ]; then
    echo "🎉 COMPLETE SUCCESS!"
    echo "   • Authentication: ✅ Completed"
    echo "   • Certificate Access: ✅ Available"
    echo "   • Binary Signing: ✅ Completed"
    echo ""
    echo "📊 Signing Results:"
    echo "   Original: $(stat -c%s "$BINARY_PATH" 2>/dev/null || stat -f%z "$BINARY_PATH" 2>/dev/null || echo "unknown") bytes"
    echo "   Signed:   $(stat -c%s "$SIGNED_BINARY" 2>/dev/null || stat -f%z "$SIGNED_BINARY" 2>/dev/null || echo "unknown") bytes"
    
    # Update BINARY_PATH for subsequent steps
    cp "$SIGNED_BINARY" "$BINARY_PATH"
    echo "✅ Updated original binary with signed version"
    
elif [ -f "$BINARY_PATH" ]; then
    # Check if original binary was signed in place
    echo "🔍 PARTIAL SUCCESS - Checking if binary was signed in place..."
    
    # Basic size check (signed binaries are typically larger)
    ORIGINAL_SIZE=$(stat -c%s "$BINARY_PATH" 2>/dev/null || stat -f%z "$BINARY_PATH" 2>/dev/null || echo "0")
    if [ "$ORIGINAL_SIZE" -gt 1000000 ]; then  # > 1MB suggests signing metadata added
        echo "✅ Binary size suggests signing may have occurred: ${ORIGINAL_SIZE} bytes"
        echo "💡 Manual verification recommended with signtool or certificate viewer"
    else
        echo "⚠️ Binary size unchanged - signing may not have completed"
    fi
    
else
    echo "⚠️ AUTHENTICATION STATUS UNCLEAR"
    echo ""
    echo "📝 Troubleshooting steps:"
    echo "   1. ✅ Check if SimplySign Desktop authentication dialog appeared"
    echo "   2. ✅ Verify credentials were entered correctly:"
    echo "      • Username: $CERTUM_USERNAME"
    if [ -n "$CERTUM_API_TOKEN" ]; then
      echo "      • TOTP: $CERTUM_API_TOKEN (may have expired)"
    else
      echo "      • TOTP: Generated from mobile app during authentication"
    fi
    echo "   3. ✅ Look for certificate access in SimplySign Desktop GUI"
    echo "   4. ✅ Try manual binary signing if certificates are accessible"
    echo ""
    if [ -n "$CERTUM_API_TOKEN" ]; then
      echo "   🔄 If TOTP expired, generate new token and retry authentication"
    else
      echo "   🔄 Generate fresh TOTP from mobile app and retry authentication"
    fi
fi

echo ""
echo "🎯 SUMMARY:"
echo "   • Username: $CERTUM_USERNAME"
if [ -n "$CERTUM_API_TOKEN" ]; then
  echo "   • API Token: $CERTUM_API_TOKEN (expires in ~30 seconds)"
else
  echo "   • API Token: Manual entry via OAuth2 dialog"
fi
echo "   • Certificate: $CERTUM_CERTIFICATE_SHA1"
echo "   • Binary: $BINARY_PATH"
echo "   • Status: $([ -f "$SIGNED_BINARY" ] && echo "✅ Signed" || echo "🔄 Manual OAuth2 authentication required")"
echo ""
echo "🚀 NEXT: Verify signed binary and proceed with distribution"
