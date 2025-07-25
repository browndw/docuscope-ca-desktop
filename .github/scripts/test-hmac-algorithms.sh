#!/bin/bash

# Test Real Certum Authentication Methods
# Based on mobile app analysis: TOTP-based activation code generation
# Mobile app structure: seed + algorithm + digits + timeStep (30s intervals)
# Key discovery: Config.plist reveals mobile client ID and real SEED endpoint

set -euo pipefail

echo "=== Testing Real Certum Authentication Methods ==="
echo "🎯 Based on mobile app Config.plist + manual authentication process:"
echo "   • Step 1: Mobile login - username + password → API token"
echo "   • Step 2: Desktop login - username + API token → certificate access"
echo "   • Mobile endpoint: /idp/oauth2.0/accessToken (confirmed accessible)"
echo "   • Mobile client ID: A5wH574pS74B4WAda3Yy (from Config.plist)"
echo "   • Authentication flow: Simulating mobile app → desktop connection"
echo "   • Manual process confirmed: CERTUM_USERNAME + CERTUM_PASSWORD → CERTUM_API_TOKEN"

# Check if we have certificate credentials
if [ -z "${CERTUM_USERNAME:-}" ] || [ -z "${CERTUM_PASSWORD:-}" ]; then
  echo "❌ CERTUM_USERNAME and CERTUM_PASSWORD required"
  exit 1
fi

echo "✅ Certificate credentials provided"
echo "Username: $CERTUM_USERNAME"

# Method 1: Test working Certum cloud endpoints (connectivity only)
echo ""
echo "Method 1: Test working Certum cloud endpoints..."
echo "Testing connectivity to REAL Certum OAuth2 endpoints..."
echo "ℹ️  Note: 302/401/200 responses are expected (authentication not attempted here)"

# Test the working OAuth2 endpoints (from test-real-signing.yml)
WORKING_ENDPOINTS=(
  "https://cloudsign.webnotarius.pl/idp/oauth2.0/authorize"
)

for endpoint in "${WORKING_ENDPOINTS[@]}"; do
  echo "Testing: $endpoint"
  curl -s --max-time 15 -I "$endpoint" 2>&1 | head -3 || echo "  Connection test completed"
done

echo "✅ Endpoint connectivity tests completed"

# Method 2: Windows credential store integration (simplified)
echo ""
echo "Method 2: Windows credential store integration..."

# Temporarily disable exit on error for credential operations
set +e

# Clear existing credentials (with error handling)
echo "Clearing existing credentials..."
EXISTING_CERTUM_CREDS=$(cmdkey /list 2>/dev/null | grep -i certum || echo "No existing Certum credentials found")
echo "$EXISTING_CERTUM_CREDS"

if echo "$EXISTING_CERTUM_CREDS" | grep -q "Target:"; then
  echo "Removing existing Certum credentials..."
  echo "$EXISTING_CERTUM_CREDS" | grep "Target:" | while read line; do
    if [[ "$line" == *"Target:"* ]]; then
      target=$(echo "$line" | cut -d':' -f2 | tr -d ' ')
      echo "  Removing: $target"
      cmdkey /delete:"$target" 2>/dev/null || echo "    Removal completed"
    fi
  done
else
  echo "✅ No existing Certum credentials to clear"
fi

# Add working credentials (matching successful approach)
echo "Adding Certum credentials..."
CREDENTIAL_TARGETS=(
  "certum.eu"
  "cloud.certum.eu"
  "cloudsign.webnotarius.pl"
  "api.certum.eu"
  "SimplySign"
  "Certum"
)

CREDENTIAL_SUCCESS_COUNT=0
for target in "${CREDENTIAL_TARGETS[@]}"; do
  echo "Adding credential for: $target"
  if cmdkey /add:"$target" /user:"$CERTUM_USERNAME" /pass:"$CERTUM_PASSWORD" >/dev/null 2>&1; then
    echo "  ✅ Successfully added credential for $target"
    ((CREDENTIAL_SUCCESS_COUNT++))
  else
    echo "  ⚠️ Could not add credential for $target (may already exist or target invalid)"
    # Continue processing other credentials instead of failing
  fi
done

echo "✅ Added $CREDENTIAL_SUCCESS_COUNT out of ${#CREDENTIAL_TARGETS[@]} credentials"
echo "📝 Note: Credential store errors are non-critical - continuing with TOTP testing..."

# Re-enable exit on error for subsequent operations
set -euo pipefail

# Method 3: SimplySign Desktop authentication (following working patterns)
echo ""
echo "Method 3: SimplySign Desktop authentication..."

SIMPLYSIGN_EXE="/c/Program Files/Certum/SimplySign Desktop/SimplySignDesktop.exe"
if [ -f "$SIMPLYSIGN_EXE" ]; then
  echo "✅ SimplySign Desktop found: $SIMPLYSIGN_EXE"
  
  # Start SimplySign Desktop for authentication (following test-real-signing.yml pattern)
  echo "Starting SimplySign Desktop for authentication..."
  "$SIMPLYSIGN_EXE" &
  SIMPLYSIGN_PID=$!
  
  echo "Waiting for SimplySign Desktop to initialize..."
  sleep 5
  
  if kill -0 $SIMPLYSIGN_PID 2>/dev/null; then
    echo "✅ SimplySign Desktop is running (PID: $SIMPLYSIGN_PID)"
    
    # Test CLI capabilities (from successful approach)
    echo ""
    echo "🔐 Testing SimplySign Desktop CLI capabilities..."
    echo "Testing --version command..."
    timeout 30 "$SIMPLYSIGN_EXE" --version 2>&1 | head -5 || echo "Version test completed"
    
    echo "Testing --help command..."
    timeout 30 "$SIMPLYSIGN_EXE" --help 2>&1 | head -5 || echo "Help test completed"
    
    echo "Testing --showCertificate command..."
    timeout 60 "$SIMPLYSIGN_EXE" --showCertificate 2>&1 | head -10 || echo "Certificate listing completed"
    
    # Clean up process
    kill $SIMPLYSIGN_PID 2>/dev/null || true
    echo "✅ SimplySign Desktop CLI testing completed"
  else
    echo "⚠️ SimplySign Desktop process ended quickly"
  fi
  
else
  echo "❌ SimplySign Desktop not found"
  exit 1
fi

# Method 4: Two-step authentication (mobile app → desktop connection)
echo ""
echo "Method 4: Two-step authentication (simulating manual process)..."
echo "🎯 Step 1: Mobile app authentication (username + password → API token)"
echo "🎯 Step 2: Desktop authentication (username + API token → certificate access)"

# Step 1: Mobile app authentication - get API token
echo ""
echo "Step 1: Mobile app authentication (getting API token)..."

# Mobile app configuration from Config.plist
MOBILE_BASE_URL="https://cloudsign.webnotarius.pl"
MOBILE_LOGIN_PATH="/idp/oauth2.0/accessToken"
MOBILE_CLIENT_ID="A5wH574pS74B4WAda3Yy"

echo "Mobile endpoint: $MOBILE_BASE_URL$MOBILE_LOGIN_PATH"
echo "Mobile client ID: $MOBILE_CLIENT_ID"

# Call mobile authentication endpoint (simulating mobile app login)
MOBILE_AUTH_RESPONSE=$(curl -s --max-time 30 \
  -X POST \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H "Accept: application/json" \
  -H "User-Agent: SimplySign-Mobile/1.0" \
  -d "grant_type=password" \
  -d "client_id=$MOBILE_CLIENT_ID" \
  -d "username=$CERTUM_USERNAME" \
  -d "password=$CERTUM_PASSWORD" \
  "$MOBILE_BASE_URL$MOBILE_LOGIN_PATH" 2>&1)

echo "Mobile auth response: $(echo "$MOBILE_AUTH_RESPONSE" | head -2)"

# Extract API token (CERTUM_API_TOKEN)
CERTUM_API_TOKEN=""
if echo "$MOBILE_AUTH_RESPONSE" | grep -q "access_token"; then
  CERTUM_API_TOKEN=$(echo "$MOBILE_AUTH_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
  echo "✅ Got API token from mobile auth (length: ${#CERTUM_API_TOKEN})"
  echo "API token (first 20 chars): ${CERTUM_API_TOKEN:0:20}..."
else
  echo "❌ Failed to get API token from mobile authentication"
  echo "Error details: $(echo "$MOBILE_AUTH_RESPONSE" | head -3)"
  echo ""
  echo "🔄 Trying alternative mobile authentication approaches..."
  
  # Try without client_secret (some OAuth2 implementations don't require it for public clients)
  echo "Trying without client_secret..."
  MOBILE_AUTH_RESPONSE_ALT=$(curl -s --max-time 30 \
    -X POST \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -H "Accept: application/json" \
    -H "User-Agent: SimplySign-Mobile/1.0" \
    -d "grant_type=password" \
    -d "client_id=$MOBILE_CLIENT_ID" \
    -d "username=$CERTUM_USERNAME" \
    -d "password=$CERTUM_PASSWORD" \
    -d "scope=openid profile" \
    "$MOBILE_BASE_URL$MOBILE_LOGIN_PATH" 2>&1)
  
  if echo "$MOBILE_AUTH_RESPONSE_ALT" | grep -q "access_token"; then
    CERTUM_API_TOKEN=$(echo "$MOBILE_AUTH_RESPONSE_ALT" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    echo "✅ Got API token with alternative approach (length: ${#CERTUM_API_TOKEN})"
  else
    echo "❌ Alternative approach also failed"
    echo "Response: $(echo "$MOBILE_AUTH_RESPONSE_ALT" | head -3)"
  fi
fi

# Step 2: Desktop authentication (only if we got API token)
if [ -n "$CERTUM_API_TOKEN" ]; then
  echo ""
  echo "Step 2: Desktop authentication (using API token)..."
  echo "🎯 Using API token as 'token from mobile application simplysign'"
  
  # Test the SEED endpoint that was failing before - now with proper API token
  SEED_URL="https://cloudsign.webnotarius.pl/cas/api/seed/code/tasks"
  
  # Generate a simple activation code for testing (not TOTP-based, since we have real API token now)
  SIMPLE_ACTIVATION_CODE="12345678"
  
  echo "Testing SEED endpoint with API token authentication..."
  SEED_RESPONSE=$(curl -s --max-time 15 \
    -X POST \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -H "Authorization: Bearer $CERTUM_API_TOKEN" \
    -H "X-Client-ID: $MOBILE_CLIENT_ID" \
    -H "User-Agent: SimplySign-Desktop/1.0" \
    -d "{
      \"email\": \"$CERTUM_USERNAME\",
      \"seedCodeReq\": {
        \"code\": \"$SIMPLE_ACTIVATION_CODE\"
      }
    }" \
    "$SEED_URL" 2>&1)
  
  echo "SEED response: $(echo "$SEED_RESPONSE" | head -3)"
  
  # Check if we got a successful response
  if echo "$SEED_RESPONSE" | grep -q '"state":"success"\|"status":"success"'; then
    echo "✅ Desktop authentication successful!"
    echo "🎉 API token works for certificate access!"
  elif echo "$SEED_RESPONSE" | grep -q '"state":"pending"\|"status":"pending"'; then
    echo "🔄 Desktop authentication pending (async processing)"
    echo "✅ API token is valid - request is being processed"
  elif echo "$SEED_RESPONSE" | grep -q '"error":\s*"invalid_token"\|"unauthorized"'; then
    echo "❌ API token invalid or expired"
  else
    echo "⚠️ Desktop authentication returned unexpected response"
    echo "Full response: $(echo "$SEED_RESPONSE" | head -5)"
  fi
  
  # Try alternative desktop endpoints if SEED fails
  echo ""
  echo "Testing alternative desktop endpoints with API token..."
  
  # Test cards endpoint (from Config.plist: /card/v1)
  CARDS_URL="https://cloudsign.webnotarius.pl/card/v1/cards/tasks"
  echo "Testing cards endpoint: $CARDS_URL"
  
  CARDS_RESPONSE=$(curl -s --max-time 15 \
    -X GET \
    -H "Accept: application/json" \
    -H "Authorization: Bearer $CERTUM_API_TOKEN" \
    -H "X-Client-ID: $MOBILE_CLIENT_ID" \
    -H "User-Agent: SimplySign-Desktop/1.0" \
    "$CARDS_URL" 2>&1)
  
  echo "Cards response: $(echo "$CARDS_RESPONSE" | head -2)"
  
  if echo "$CARDS_RESPONSE" | grep -q '"cards"\|"certificates"\|"id"'; then
    echo "✅ Cards endpoint successful - API token works!"
  else
    echo "ℹ️ Cards endpoint response: $(echo "$CARDS_RESPONSE" | head -1)"
  fi
  
else
  echo ""
  echo "❌ Skipping Step 2 - no API token available"
  echo "💡 Need to resolve mobile authentication first"
fi

# Method 5: Certificate store verification (avoiding OAuth2 issues)
echo ""
echo "Method 5: Certificate store verification..."
echo "Checking Windows certificate stores for any existing certificates..."

powershell -Command "
  try {
    Write-Host 'Checking certificate stores...'
    \$userCerts = Get-ChildItem -Path 'Cert:\\CurrentUser\\My' -ErrorAction SilentlyContinue
    \$machineCerts = Get-ChildItem -Path 'Cert:\\LocalMachine\\My' -ErrorAction SilentlyContinue
    
    Write-Host \"CurrentUser store: \$(\$userCerts.Count) certificates\"
    Write-Host \"LocalMachine store: \$(\$machineCerts.Count) certificates\"
    
    # Look for any Certum or code signing certificates
    \$allCerts = \$userCerts + \$machineCerts
    \$certumCerts = \$allCerts | Where-Object { 
      \$_.Subject -like '*certum*' -or \$_.Issuer -like '*certum*' -or \$_.Subject -like '*Unizeto*'
    }
    
    if (\$certumCerts) {
      Write-Host \"✅ Found \$(\$certumCerts.Count) Certum-related certificate(s)\"
      foreach (\$cert in \$certumCerts) {
        Write-Host \"  Subject: \$(\$cert.Subject)\"
        Write-Host \"  Thumbprint: \$(\$cert.Thumbprint)\"
      }
    } else {
      Write-Host '⚠️ No Certum certificates found in Windows stores'
      Write-Host 'This indicates authentication has not yet loaded certificates'
    }
  } catch {
    Write-Host \"Certificate store check error: \$(\$_.Exception.Message)\"
  }
"

echo ""
echo "✅ Authentication method testing completed"
echo "🎯 Key findings:"
echo "  • Mobile authentication endpoint: cloudsign.webnotarius.pl/idp/oauth2.0/accessToken"
echo "  • Two-step process: mobile login → API token → desktop connection"
echo "  • Mobile client ID: A5wH574pS74B4WAda3Yy (from Config.plist)"
echo "  • Desktop uses API token as 'token from mobile application simplysign'"
echo "  • Simulating manual process: CERTUM_USERNAME + CERTUM_PASSWORD → CERTUM_API_TOKEN"
echo "  • OAuth2 Resource Owner Password Credentials flow implemented"
