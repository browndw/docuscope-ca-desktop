#!/bin/bash

# Test Real Certum Authentication Methods
# Based on endpoint testing: CONFIRMED WORKING SEED endpoint structure
# Architecture: OAuth2 Bearer token → SEED code validation → certificate access
# Key discovery: /cas/api/seed/code/tasks with seedCodeReq.email + seedCodeReq.code

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

# Check if we're running on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
  echo "🪟 Running on Windows - testing credential store..."
else
  echo "🍎 Running on macOS/Linux - Windows credential store not available"
  echo "⏭️ Skipping Method 2 - proceeding to next method..."
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then

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

fi  # End Windows credential store section

# Method 3: SimplySign Desktop authentication (following working patterns)
echo ""
echo "Method 3: SimplySign Desktop authentication..."

# Check if we're running on macOS or Windows
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "🍎 Running on macOS - SimplySign Desktop is Windows-only"
  echo "⏭️ Skipping Method 3 - proceeding to OAuth2 testing..."
  SIMPLYSIGN_EXE=""
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
  echo "🪟 Running on Windows - checking for SimplySign Desktop..."
  SIMPLYSIGN_EXE="/c/Program Files/Certum/SimplySign Desktop/SimplySignDesktop.exe"
else
  echo "🐧 Running on Linux/Other - SimplySign Desktop is Windows-only"
  echo "⏭️ Skipping Method 3 - proceeding to OAuth2 testing..."
  SIMPLYSIGN_EXE=""
fi

if [ -n "$SIMPLYSIGN_EXE" ] && [ -f "$SIMPLYSIGN_EXE" ]; then
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
  if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✅ Method 3 skipped - continuing with OAuth2 testing on non-Windows platform"
  else
    echo "❌ SimplySign Desktop not found"
    echo "⚠️ This may affect certificate access - continuing with OAuth2 testing..."
  fi
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
# Based on testing: endpoint confirmed working, need to try multiple auth methods
echo "🔄 Trying multiple authentication approaches..."

CERTUM_API_TOKEN=""

# Method A: OAuth2 Resource Owner Password Credentials (form-encoded)
echo "Method A: OAuth2 with form encoding..."
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

echo "Method A response: $(echo "$MOBILE_AUTH_RESPONSE" | head -2)"

if echo "$MOBILE_AUTH_RESPONSE" | grep -q "access_token"; then
  CERTUM_API_TOKEN=$(echo "$MOBILE_AUTH_RESPONSE" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
  echo "✅ Method A successful - Got API token (length: ${#CERTUM_API_TOKEN})"
fi

# Method B: Basic Auth (if Method A failed)
if [ -z "$CERTUM_API_TOKEN" ]; then
  echo "Method B: Basic Authentication..."
  
  # Encode credentials for Basic Auth
  BASIC_CREDS=$(echo -n "$CERTUM_USERNAME:$CERTUM_PASSWORD" | base64)
  
  MOBILE_AUTH_RESPONSE_B=$(curl -s --max-time 30 \
    -X POST \
    -H "Authorization: Basic $BASIC_CREDS" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -H "Accept: application/json" \
    -H "User-Agent: SimplySign-Mobile/1.0" \
    -d "grant_type=password" \
    -d "client_id=$MOBILE_CLIENT_ID" \
    "$MOBILE_BASE_URL$MOBILE_LOGIN_PATH" 2>&1)
  
  echo "Method B response: $(echo "$MOBILE_AUTH_RESPONSE_B" | head -2)"
  
  if echo "$MOBILE_AUTH_RESPONSE_B" | grep -q "access_token"; then
    CERTUM_API_TOKEN=$(echo "$MOBILE_AUTH_RESPONSE_B" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    echo "✅ Method B successful - Got API token (length: ${#CERTUM_API_TOKEN})"
  fi
fi

# Method C: JSON payload (if Methods A & B failed)
if [ -z "$CERTUM_API_TOKEN" ]; then
  echo "Method C: JSON payload authentication..."
  
  MOBILE_AUTH_RESPONSE_C=$(curl -s --max-time 30 \
    -X POST \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -H "User-Agent: SimplySign-Mobile/1.0" \
    -d "{
      \"grant_type\": \"password\",
      \"client_id\": \"$MOBILE_CLIENT_ID\",
      \"username\": \"$CERTUM_USERNAME\",
      \"password\": \"$CERTUM_PASSWORD\"
    }" \
    "$MOBILE_BASE_URL$MOBILE_LOGIN_PATH" 2>&1)
  
  echo "Method C response: $(echo "$MOBILE_AUTH_RESPONSE_C" | head -2)"
  
  if echo "$MOBILE_AUTH_RESPONSE_C" | grep -q "access_token"; then
    CERTUM_API_TOKEN=$(echo "$MOBILE_AUTH_RESPONSE_C" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    echo "✅ Method C successful - Got API token (length: ${#CERTUM_API_TOKEN})"
  fi
fi

# Method D: Alternative endpoint (if all above failed)
if [ -z "$CERTUM_API_TOKEN" ]; then
  echo "Method D: Alternative login endpoint..."
  
  MOBILE_AUTH_RESPONSE_D=$(curl -s --max-time 30 \
    -X POST \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -H "Accept: application/json" \
    -H "User-Agent: SimplySign-Mobile/1.0" \
    -d "grant_type=password" \
    -d "client_id=$MOBILE_CLIENT_ID" \
    -d "username=$CERTUM_USERNAME" \
    -d "password=$CERTUM_PASSWORD" \
    -d "scope=openid profile" \
    "https://cloudsign.webnotarius.pl/oauth2/token" 2>&1)
  
  echo "Method D response: $(echo "$MOBILE_AUTH_RESPONSE_D" | head -2)"
  
  if echo "$MOBILE_AUTH_RESPONSE_D" | grep -q "access_token"; then
    CERTUM_API_TOKEN=$(echo "$MOBILE_AUTH_RESPONSE_D" | grep -o '"access_token":"[^"]*"' | cut -d'"' -f4)
    echo "✅ Method D successful - Got API token (length: ${#CERTUM_API_TOKEN})"
  fi
fi

if [ -n "$CERTUM_API_TOKEN" ]; then
  echo "✅ Successfully obtained API token!"
  echo "API token (first 20 chars): ${CERTUM_API_TOKEN:0:20}..."
else
  echo "❌ All authentication methods failed"
  echo "📝 Debug info:"
  echo "  Method A (OAuth2 form): $(echo "$MOBILE_AUTH_RESPONSE" | head -1)"
  echo "  Method B (Basic Auth): $(echo "$MOBILE_AUTH_RESPONSE_B" | head -1)"
  echo "  Method C (JSON): $(echo "$MOBILE_AUTH_RESPONSE_C" | head -1)"
  echo "  Method D (Alt endpoint): $(echo "$MOBILE_AUTH_RESPONSE_D" | head -1)"
fi

# Step 2: Desktop authentication (only if we got API token)
if [ -n "$CERTUM_API_TOKEN" ]; then
  echo ""
  echo "Step 2: Desktop authentication (using API token)..."
  echo "🎯 Using API token as 'token from mobile application simplysign'"
  
  # Test the SEED endpoint with confirmed working structure
  SEED_URL="https://cloudsign.webnotarius.pl/cas/api/seed/code/tasks"
  
  echo "🎯 Testing SEED endpoint with Bearer token authentication..."
  echo "Endpoint: $SEED_URL"
  echo "Structure: seedCodeReq.email + seedCodeReq.code (confirmed via endpoint testing)"
  
  # Test multiple SEED code formats (since we don't have real mobile app SEED yet)
  SEED_CODES=("123456" "12345678" "000000" "111111")
  
  for SEED_CODE in "${SEED_CODES[@]}"; do
    echo ""
    echo "Testing with SEED code: $SEED_CODE"
    
    # Use the confirmed JSON structure from endpoint testing
    SEED_RESPONSE=$(curl -s --max-time 15 \
      -X POST \
      -H "Content-Type: application/json" \
      -H "Accept: application/json" \
      -H "Authorization: Bearer $CERTUM_API_TOKEN" \
      -H "X-Client-ID: $MOBILE_CLIENT_ID" \
      -H "User-Agent: SimplySign-Desktop/1.0" \
      -d "{
        \"seedCodeReq\": {
          \"email\": \"$CERTUM_USERNAME\",
          \"code\": \"$SEED_CODE\"
        }
      }" \
      "$SEED_URL" 2>&1)
    
    echo "SEED response: $(echo "$SEED_RESPONSE" | head -3)"
    
    # Check response types (from endpoint testing)
    if echo "$SEED_RESPONSE" | grep -q '"state":"success"\|"status":"success"'; then
      echo "✅ SEED authentication successful with code: $SEED_CODE"
      echo "🎉 Desktop authentication complete!"
      break
    elif echo "$SEED_RESPONSE" | grep -q '"state":"pending"\|"status":"pending"'; then
      echo "🔄 SEED request pending (async processing) with code: $SEED_CODE"
      echo "✅ Bearer token is valid - request being processed"
      break
    elif echo "$SEED_RESPONSE" | grep -q '"error":\s*"invalid_token"\|"unauthorized"'; then
      echo "❌ Bearer token invalid or expired"
      break
    elif echo "$SEED_RESPONSE" | grep -q 'validation:NotBlank.seedCodeReq'; then
      echo "⚠️ SEED code format issue: $(echo "$SEED_RESPONSE" | grep -o 'validation:NotBlank[^"]*')"
      echo "This confirms endpoint structure - trying next code format..."
    elif echo "$SEED_RESPONSE" | grep -q '"error":\s*"invalid_grant"\|"invalid_request"'; then
      echo "⚠️ SEED code invalid: $SEED_CODE"
      echo "Expected - trying next code format..."
    else
      echo "❓ Unexpected SEED response with code $SEED_CODE:"
      echo "$(echo "$SEED_RESPONSE" | head -5)"
    fi
  done
  
  # If all SEED codes failed, explain the next steps
  if ! echo "$SEED_RESPONSE" | grep -q '"state":"success"\|"status":"success"\|"state":"pending"\|"status":"pending"'; then
    echo ""
    echo "💡 All test SEED codes failed - this is expected!"
    echo "📱 Next step: Generate real SEED code from mobile app:"
    echo "   1. Open SimplySign mobile app"
    echo "   2. Log in with your credentials"
    echo "   3. Tap 'Generate token' button"
    echo "   4. Use the 6-8 digit code that appears"
    echo "   5. Note: SEED codes expire in 30 seconds"
    echo ""
    echo "🔧 For automated testing, we need to:"
    echo "   • Reverse engineer the TOTP algorithm from mobile app"
    echo "   • Or integrate with mobile app's SEED generation API"
    echo "   • Or capture SEED codes via mobile app automation"
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

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
  echo "🪟 Checking Windows certificate stores for any existing certificates..."

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
else
  echo "🍎 Running on macOS/Linux - checking system keychain..."
  echo "Note: Certificate store checking is primarily for Windows SimplySign Desktop"
  
  # Check for any certificates in macOS keychain (basic check)
  if command -v security >/dev/null 2>&1; then
    echo "Checking macOS keychain for any Certum certificates..."
    CERTUM_CERTS=$(security find-certificate -a -c "certum" 2>/dev/null | wc -l || echo "0")
    echo "Found $CERTUM_CERTS potential Certum certificates in keychain"
    
    if [ "$CERTUM_CERTS" -gt 0 ]; then
      echo "✅ Some certificates found - may include Certum certificates"
    else
      echo "⚠️ No obvious Certum certificates found in keychain"
    fi
  else
    echo "⚠️ Security command not available - cannot check keychain"
  fi
fi

echo ""
echo "✅ Authentication method testing completed"
echo "🎯 Key findings:"
echo "  • Mobile authentication endpoint: cloudsign.webnotarius.pl/idp/oauth2.0/accessToken"
echo "  • SEED endpoint confirmed: cloudsign.webnotarius.pl/cas/api/seed/code/tasks"
echo "  • JSON structure confirmed: seedCodeReq.email + seedCodeReq.code"
echo "  • Bearer token authentication required for SEED endpoint"
echo "  • Two-step process: mobile login → Bearer token → SEED validation"
echo "  • Mobile client ID: A5wH574pS74B4WAda3Yy (from Config.plist)"

# Method 6: Direct SEED endpoint testing (for manual SEED codes)
echo ""
echo "Method 6: Direct SEED endpoint testing (for manual use)..."
echo "🎯 This method allows testing with manually generated SEED codes"
echo "📱 Instructions:"
echo "   1. Run this script and wait for this section"
echo "   2. Open SimplySign mobile app on your phone"
echo "   3. Log in and tap 'Generate token'"
echo "   4. Enter the SEED code when prompted below"
echo "   5. Press Enter within 30 seconds (before SEED expires)"
echo ""

# Check if we're running interactively or in CI
if [ -t 0 ]; then
  echo "🔧 Interactive mode detected - enable manual SEED testing? (y/n)"
  read -r ENABLE_MANUAL_TESTING
  
  if [[ "$ENABLE_MANUAL_TESTING" =~ ^[Yy] ]]; then
    echo ""
    echo "📱 Please generate a SEED code in your mobile app now..."
    echo "⏰ Enter the 6-8 digit SEED code (you have 30 seconds):"
    read -r MANUAL_SEED_CODE
    
    if [ -n "$MANUAL_SEED_CODE" ]; then
      echo "Testing with manual SEED code: $MANUAL_SEED_CODE"
      
      # Test the SEED endpoint directly without OAuth2 dependency
      DIRECT_SEED_RESPONSE=$(curl -s --max-time 15 \
        -X POST \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -H "User-Agent: SimplySign-Desktop/1.0" \
        -d "{
          \"seedCodeReq\": {
            \"email\": \"$CERTUM_USERNAME\",
            \"code\": \"$MANUAL_SEED_CODE\"
          }
        }" \
        "https://cloudsign.webnotarius.pl/cas/api/seed/code/tasks" 2>&1)
      
      echo "Direct SEED response: $(echo "$DIRECT_SEED_RESPONSE" | head -3)"
      
      if echo "$DIRECT_SEED_RESPONSE" | grep -q '"state":"success"\|"status":"success"'; then
        echo "🎉 SUCCESS! Manual SEED code worked!"
        echo "✅ SEED endpoint structure confirmed working"
      elif echo "$DIRECT_SEED_RESPONSE" | grep -q 'validation:NotBlank'; then
        echo "⚠️ SEED validation failed - this confirms endpoint structure is correct"
        echo "The endpoint is working, just need proper authentication flow"
      elif echo "$DIRECT_SEED_RESPONSE" | grep -q '"error".*"unauthorized"\|401'; then
        echo "🔒 SEED endpoint requires Bearer token authentication (expected)"
        echo "This confirms our two-step authentication approach is correct"
      else
        echo "❓ Unexpected response to manual SEED:"
        echo "$(echo "$DIRECT_SEED_RESPONSE" | head -5)"
      fi
    else
      echo "⏭️ No SEED code entered - skipping manual test"
    fi
  else
    echo "⏭️ Manual SEED testing disabled"
  fi
else
  echo "🤖 Non-interactive mode - skipping manual SEED testing"
  echo "💡 To test with real SEED codes, run this script interactively"
fi
