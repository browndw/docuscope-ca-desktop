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
  
  # Common signtool paths
  SIGNTOOL_PATHS=(
    "/c/Program Files (x86)/Windows Kits/10/bin/*/x64/signtool.exe"
    "/c/Program Files/Windows Kits/10/bin/*/x64/signtool.exe"
    "/c/Program Files (x86)/Windows Kits/8.1/bin/x64/signtool.exe"
    "/c/Program Files/Windows Kits/8.1/bin/x64/signtool.exe"
  )
  
  for path_pattern in "${SIGNTOOL_PATHS[@]}"; do
    for path in $path_pattern; do
      if [ -f "$path" ]; then
        echo "✅ Found signtool.exe: $path"
        echo "SIGNTOOL_PATH=$path" >> "$GITHUB_OUTPUT"
        return 0
      fi
    done
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
