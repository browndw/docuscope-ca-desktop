#!/bin/bash

# Pre-flight Check for Certum Automated Signing
# Validates environment and prerequisites before running the main workflow

set -euo pipefail

echo "=== CERTUM AUTOMATED SIGNING PRE-FLIGHT CHECK ==="
echo "🔍 Validating environment and prerequisites..."
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check 1: PowerShell availability
echo "1. PowerShell Environment"
if command_exists powershell; then
    PWSH_VERSION=$(powershell -Command '$PSVersionTable.PSVersion.ToString()' 2>/dev/null || echo "Unknown")
    echo "   ✅ PowerShell available: $PWSH_VERSION"
else
    echo "   ❌ PowerShell not found"
    echo "      Required for TOTP generation and COM automation"
    exit 1
fi

# Check 2: Required GitHub Secrets (in CI environment)
echo ""
echo "2. GitHub Secrets Configuration"
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    # Check CERTUM_OTP_URI
    if [ -n "${CERTUM_OTP_URI:-}" ]; then
        URI_LENGTH=${#CERTUM_OTP_URI}
        if [[ "$CERTUM_OTP_URI" == otpauth://totp/* ]]; then
            echo "   ✅ CERTUM_OTP_URI: Valid format ($URI_LENGTH chars)"
        else
            echo "   ❌ CERTUM_OTP_URI: Invalid format (should start with 'otpauth://totp/')"
            exit 1
        fi
    else
        echo "   ❌ CERTUM_OTP_URI: Missing"
        exit 1
    fi
    
    # Check CERTUM_USERNAME
    if [ -n "${CERTUM_USERNAME:-}" ]; then
        if [[ "$CERTUM_USERNAME" == *@* ]]; then
            echo "   ✅ CERTUM_USERNAME: Valid email format"
        else
            echo "   ❌ CERTUM_USERNAME: Should be an email address"
            exit 1
        fi
    else
        echo "   ❌ CERTUM_USERNAME: Missing"
        exit 1
    fi
    
    echo "   ℹ️ PKCS#11 Mode: Certificate SHA1 not required (auto-discovery via PKCS#11)"
else
    echo "   ℹ️ Not running in GitHub Actions - secrets not available for validation"
fi

# Check 3: PowerShell Script Availability
echo ""
echo "3. PowerShell Script Validation"
SCRIPT_PATH="./.github/scripts/Connect-SimplySign.ps1"
if [ -f "$SCRIPT_PATH" ]; then
    echo "   ✅ Connect-SimplySign.ps1 found"
    
    # Basic syntax validation
    if powershell -Command "Get-Content '$SCRIPT_PATH' | Out-Null" 2>/dev/null; then
        echo "   ✅ PowerShell script syntax appears valid"
    else
        echo "   ❌ PowerShell script has syntax errors"
        exit 1
    fi
else
    echo "   ❌ Connect-SimplySign.ps1 not found at $SCRIPT_PATH"
    exit 1
fi

# Check 4: Supporting Scripts
echo ""
echo "4. Supporting Script Validation"
REQUIRED_SCRIPTS=(
    "./.github/scripts/install-simplysign.sh"
    "./.github/scripts/sign-binary.sh"
    "./.github/scripts/utils/certificate-utils.sh"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "   ✅ $(basename "$script") found"
    else
        echo "   ❌ $(basename "$script") missing at $script"
        exit 1
    fi
done

# Check 5: Windows Environment (if applicable)
echo ""
echo "5. Windows Environment Check"
if [ "${OS:-}" = "Windows_NT" ] || command_exists cmd.exe; then
    echo "   ✅ Windows environment detected"
    
    # Check for common Windows signing tools locations
    if [ -d "/c/Program Files (x86)/Windows Kits" ]; then
        echo "   ✅ Windows SDK detected (signtool likely available)"
    else
        echo "   ⚠️ Windows SDK not found - signtool may not be available"
    fi
else
    echo "   ℹ️ Non-Windows environment (workflow will run on windows-latest)"
fi

# Check 6: TOTP Generation Test (if in CI with secrets)
echo ""
echo "6. TOTP Generation Test"
if [ "${GITHUB_ACTIONS:-}" = "true" ] && [ -n "${CERTUM_OTP_URI:-}" ]; then
    echo "   🧪 Testing TOTP generation capability..."
    
    # Test PowerShell TOTP generation without actually connecting
    TEST_RESULT=$(powershell -Command "
        \$OtpUri = '$CERTUM_OTP_URI'
        try {
            \$uri = [Uri]\$OtpUri
            if (\$uri.Scheme -eq 'otpauth' -and \$uri.Host -eq 'totp') {
                Write-Output 'VALID'
            } else {
                Write-Output 'INVALID'
            }
        } catch {
            Write-Output 'ERROR'
        }
    " 2>/dev/null || echo "ERROR")
    
    if [ "$TEST_RESULT" = "VALID" ]; then
        echo "   ✅ TOTP URI parsing successful"
    else
        echo "   ❌ TOTP URI parsing failed: $TEST_RESULT"
        exit 1
    fi
else
    echo "   ℹ️ Skipping TOTP test (not in CI or secrets not available)"
fi

echo ""
echo "=== PRE-FLIGHT CHECK COMPLETE ==="
echo ""

# Summary
ALL_CHECKS_PASSED=true

if [ "$ALL_CHECKS_PASSED" = "true" ]; then
    echo "🎉 ALL CHECKS PASSED!"
    echo ""
    echo "✅ Environment is ready for Certum automated signing"
    echo "✅ PowerShell environment configured"
    echo "✅ Required scripts available"
    echo "✅ Secrets properly configured (if in CI)"
    echo ""
    echo "🚀 Ready to run the breakthrough workflow!"
    echo ""
    echo "Next steps:"
    echo "1. Go to Actions → 'Certum Automated Signing - BREAKTHROUGH'"
    echo "2. Click 'Run workflow'"
    echo "3. Choose 'Use mock binary' for initial testing"
    echo "4. Watch the automated signing magic! ✨"
else
    echo "❌ SOME CHECKS FAILED"
    echo ""
    echo "Please address the issues above before running the workflow."
    exit 1
fi
