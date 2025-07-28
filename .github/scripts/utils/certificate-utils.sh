#!/bin/bash

# Certificate utilities for Certum PKCS#11 code signing
# Updated for PKCS#11 interface as per official Certum documentation

# Function to check PKCS#11 certificate access
check_pkcs11_certificates() {
  echo "Checking PKCS#11 certificate access..."
  
  # Check for SimplySign PKCS#11 library
  local pkcs11_lib="/c/Windows/System32/SimplySignPKCS.dll"
  if [ ! -f "$pkcs11_lib" ]; then
    echo "❌ PKCS#11 library not found at: $pkcs11_lib"
    echo "   SimplySign Desktop may not be properly installed or connected"
    return 1
  fi
  
  echo "✅ PKCS#11 library found: $pkcs11_lib"
  
  # Check if we can enumerate certificates via PKCS#11
  echo "📋 Attempting to discover certificates via PKCS#11..."
  
  # Method 1: Use PowerShell to check for Certum certificates
  powershell -Command "
    Write-Host 'Checking for Certum certificates...'
    
    # Check various certificate stores for Certum certificates
    \$stores = @('My', 'Root', 'CA', 'TrustedPeople', 'TrustedPublisher')
    \$locations = @('CurrentUser', 'LocalMachine')
    \$certumFound = \$false
    
    foreach (\$location in \$locations) {
        foreach (\$storeName in \$stores) {
            try {
                \$store = New-Object System.Security.Cryptography.X509Certificates.X509Store(\$storeName, \$location)
                \$store.Open('ReadOnly')
                
                foreach (\$cert in \$store.Certificates) {
                    if (\$cert.Subject -like '*Certum*' -or \$cert.Issuer -like '*Certum*' -or \$cert.Subject -like '*Code Signing*') {
                        Write-Host \"✅ Found certificate in \$location\\\$storeName:\"
                        Write-Host \"  Subject: \$(\$cert.Subject)\"
                        Write-Host \"  Thumbprint: \$(\$cert.Thumbprint)\"
                        Write-Host \"  HasPrivateKey: \$(\$cert.HasPrivateKey)\"
                        Write-Host \"  NotAfter: \$(\$cert.NotAfter)\"
                        \$certumFound = \$true
                    }
                }
                \$store.Close()
            } catch {
                # Silent fail for inaccessible stores
            }
        }
    }
    
    if (-not \$certumFound) {
        Write-Host '⚠️ No Certum certificates found in Windows certificate stores'
        Write-Host '   This is expected - certificates may only be accessible via PKCS#11'
    }
  " 2>/dev/null || echo "PowerShell certificate check failed"
  
  return 0
}

# Function to find PKCS#11-compatible signing tools
find_pkcs11_signing_tool() {
  echo "Searching for PKCS#11-compatible signing tools..."
  
  # Check for osslsigncode (best option for PKCS#11)
  local osslsigncode_locations=(
    "/c/Program Files/osslsigncode/osslsigncode.exe"
    "/c/Program Files (x86)/osslsigncode/osslsigncode.exe"
    "osslsigncode.exe"
    "osslsigncode"
  )
  
  for location in "${osslsigncode_locations[@]}"; do
    if command -v "$location" >/dev/null 2>&1; then
      echo "✅ Found osslsigncode: $location"
      if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "OSSLSIGNCODE_PATH=$location" >> "$GITHUB_OUTPUT"
      fi
      export OSSLSIGNCODE_PATH="$location"
      return 0
    fi
  done
  
  echo "⚠️ osslsigncode not found - will attempt installation or use fallback"
  return 1
}

# Function to install osslsigncode if needed
install_osslsigncode() {
  echo "Attempting to install osslsigncode..."
  
  # Try chocolatey first (common on Windows CI)
  if command -v choco >/dev/null 2>&1; then
    echo "Installing osslsigncode via Chocolatey..."
    if choco install osslsigncode -y; then
      echo "✅ osslsigncode installed successfully"
      export OSSLSIGNCODE_PATH="osslsigncode"
      return 0
    else
      echo "❌ Chocolatey installation failed"
    fi
  fi
  
  # Try winget (Windows Package Manager)
  if command -v winget >/dev/null 2>&1; then
    echo "Trying winget installation..."
    winget install osslsigncode || echo "winget installation failed"
  fi
  
  # Check if installation worked
  if command -v osslsigncode >/dev/null 2>&1; then
    echo "✅ osslsigncode now available"
    export OSSLSIGNCODE_PATH="osslsigncode"
    return 0
  fi
  
  echo "❌ Could not install osslsigncode automatically"
  return 1
}

# Function to find signtool.exe
find_signtool() {
  echo "Searching for signtool.exe..."
  echo "🔧 Prioritizing working SDK versions, avoiding 10.0.22621.0 due to /fd parameter regression"
  
  # Working signtool paths (prioritize older stable versions that work with /fd)
  SIGNTOOL_PATHS=(
    # Prioritize known working versions first
    "/c/Program Files (x86)/Windows Kits/10/bin/10.0.17763.0/x64/signtool.exe"
    "/c/Program Files (x86)/Windows Kits/10/bin/10.0.18362.0/x64/signtool.exe"
    "/c/Program Files (x86)/Windows Kits/10/bin/10.0.19041.0/x64/signtool.exe"
    "/c/Program Files (x86)/Windows Kits/10/bin/10.0.20348.0/x64/signtool.exe"
    "/c/Program Files (x86)/Windows Kits/10/bin/10.0.22000.0/x64/signtool.exe"
    # Legacy SDK versions (also reliable)
    "/c/Program Files (x86)/Microsoft SDKs/Windows/v7.1A/Bin/signtool.exe"
    "/c/Program Files (x86)/Windows Kits/8.1/bin/x64/signtool.exe"
    # Wildcard patterns for any other versions (but lower priority)
    "/c/Program Files (x86)/Windows Kits/10/bin/*/x64/signtool.exe"
    "/c/Program Files/Windows Kits/10/bin/*/x64/signtool.exe"
    # Check PATH (last resort)
    "$(which signtool.exe 2>/dev/null)"
  )
  
  for path_pattern in "${SIGNTOOL_PATHS[@]}"; do
    if [[ "$path_pattern" == *"*"* ]]; then
      # Handle wildcard patterns
      for path in $path_pattern; do
        if [ -f "$path" ]; then
          # Skip problematic SDK version
          if [[ "$path" == *"10.0.22621.0"* ]]; then
            echo "⚠️ Skipping problematic SDK version: $path (known /fd parameter regression)"
            continue
          fi
          
          echo "✅ Found signtool.exe: $path"
          if [ -n "${GITHUB_OUTPUT:-}" ]; then
            echo "SIGNTOOL_PATH=$path" >> "$GITHUB_OUTPUT"
          fi
          export SIGNTOOL_PATH="$path"
          return 0
        fi
      done
    else
      # Handle direct paths
      if [ -f "$path_pattern" ] && [ -n "$path_pattern" ]; then
        # Skip problematic SDK version
        if [[ "$path_pattern" == *"10.0.22621.0"* ]]; then
          echo "⚠️ Skipping problematic SDK version: $path_pattern (known /fd parameter regression)"
          continue
        fi
        
        echo "✅ Found signtool.exe: $path_pattern"
        if [ -n "${GITHUB_OUTPUT:-}" ]; then
          echo "SIGNTOOL_PATH=$path_pattern" >> "$GITHUB_OUTPUT"
        fi
        export SIGNTOOL_PATH="$path_pattern"
        return 0
      fi
    fi
  done
  
  echo "❌ signtool.exe not found"
  return 1
}

# Function to test PKCS#11 signing capability
test_pkcs11_signing() {
  local tool_path="$1"
  local test_file="$2"
  local tool_type="${3:-auto}"
  
  if [ ! -f "$test_file" ]; then
    echo "❌ Test file not found: $test_file"
    return 1
  fi
  
  echo "Testing PKCS#11 code signing capability..."
  echo "Tool: $tool_path"
  echo "Test file: $test_file"
  
  case "$tool_type" in
    "osslsigncode")
      echo "Testing osslsigncode with PKCS#11..."
      # Create minimal PKCS#11 config for testing
      local pkcs11_config="test_pkcs11.conf"
      cat > "$pkcs11_config" << EOF
name=SimplySignPKCS
library=/c/Windows/System32/SimplySignPKCS.dll
slotListIndex=0
EOF
      
      # Test signing (without timestamp for speed)
      if "$tool_path" sign \
          -pkcs11engine "/c/Windows/System32/SimplySignPKCS.dll" \
          -pkcs11module "/c/Windows/System32/SimplySignPKCS.dll" \
          -h sha256 \
          -in "$test_file" \
          -out "${test_file}.test" 2>/dev/null; then
        echo "✅ PKCS#11 signing test successful with osslsigncode"
        rm -f "${test_file}.test" "$pkcs11_config"
        return 0
      else
        echo "❌ PKCS#11 signing test failed with osslsigncode"
        rm -f "${test_file}.test" "$pkcs11_config"
        return 1
      fi
      ;;
      
    "signtool")
      echo "Testing signtool with smart card auto-select..."
      # Test with auto-select (/a) which should find PKCS#11 certificates
      if "$tool_path" sign /a /fd SHA256 "$test_file" 2>/dev/null; then
        echo "✅ Smart card signing test successful with signtool"
        return 0
      else
        echo "❌ Smart card signing test failed with signtool"
        return 1
      fi
      ;;
      
    *)
      echo "❌ Unknown tool type: $tool_type"
      return 1
      ;;
  esac
}
