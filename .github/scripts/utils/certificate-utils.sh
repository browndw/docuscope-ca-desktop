#!/bin/bash

# Certificate utilities for Certum code signing
# Common functions for certificate operations

# Function to check certificate store
check_certificate_store() {
  local cert_sha1="$1"
  
  echo "Checking certificate stores..."
  
  # Check Current User store
  echo "Current User certificates:"
  powershell -Command "Get-ChildItem -Path 'Cert:\\CurrentUser\\My' | Select-Object Subject, Thumbprint, NotAfter | Format-Table" 2>/dev/null || echo "Could not access CurrentUser store"
  
  # Check Local Machine store
  echo "Local Machine certificates:"
  powershell -Command "Get-ChildItem -Path 'Cert:\\LocalMachine\\My' | Select-Object Subject, Thumbprint, NotAfter | Format-Table" 2>/dev/null || echo "Could not access LocalMachine store"
  
  # Check for specific certificate if SHA1 provided
  if [ -n "$cert_sha1" ]; then
    echo "Searching for certificate: $cert_sha1"
    if powershell -Command "Get-ChildItem -Path 'Cert:\\CurrentUser\\My','Cert:\\LocalMachine\\My' | Where-Object { \$_.Thumbprint -eq '$cert_sha1' } | Select-Object Subject, Thumbprint, NotAfter" 2>/dev/null; then
      echo "✅ Certificate found"
      return 0
    else
      echo "❌ Certificate not found"
      return 1
    fi
  fi
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

# Function to test signing capability
test_signing() {
  local signtool_path="$1"
  local cert_sha1="$2"
  local test_file="$3"
  
  if [ ! -f "$signtool_path" ]; then
    echo "❌ signtool.exe not found at: $signtool_path"
    return 1
  fi
  
  if [ ! -f "$test_file" ]; then
    echo "❌ Test file not found: $test_file"
    return 1
  fi
  
  echo "Testing code signing with certificate: $cert_sha1"
  echo "Test file: $test_file"
  
  # Test signing command (without timestamp for speed)
  if "$signtool_path" sign /sha1 "$cert_sha1" /fd SHA256 "$test_file"; then
    echo "✅ Code signing test successful"
    return 0
  else
    echo "❌ Code signing test failed"
    return 1
  fi
}
