#!/bin/bash

# Setup Instructions for Certum Automated Signing
# This script documents the required GitHub secrets for the breakthrough workflow

echo "=== CERTUM AUTOMATED SIGNING SETUP ==="
echo "🎉 BREAKTHROUGH: Complete TOTP automation achieved!"
echo ""
echo "Required GitHub Repository Secrets:"
echo "====================================="
echo ""

echo "1. CERTUM_OTP_URI"
echo "   Description: The complete otpauth:// URI from your TOTP setup"
echo "   Format: otpauth://totp/[label]?secret=[SECRET]&issuer=[ISSUER]"
echo "   How to get: Scan the QR code with 1Password or similar, then extract the URI"
echo "   Example: otpauth://totp/Certum?secret=ABCDEFGHIJKLMNOP&issuer=Certum"
echo "   Status: $([ -n "${CERTUM_OTP_URI:-}" ] && echo "✅ Set" || echo "❌ Missing")"
echo ""

echo "2. CERTUM_USERID"
echo "   Description: Your Certum email address (username)"
echo "   Format: email@domain.com"
echo "   Example: john.doe@company.com"
echo "   Status: $([ -n "${CERTUM_USERID:-}" ] && echo "✅ Set" || echo "❌ Missing")"
echo ""

echo "3. CERTUM_CERTIFICATE_SHA1"
echo "   Description: SHA1 thumbprint of your code signing certificate"
echo "   Format: 40-character hexadecimal string"
echo "   How to get: From certificate details in SimplySign Desktop"
echo "   Example: 90986E3AC5FEBFF4CF998F174E82CB4C9E6FFC19"
echo "   Status: $([ -n "${CERTUM_CERTIFICATE_SHA1:-}" ] && echo "✅ Set" || echo "❌ Missing")"
echo ""

echo "Setup Instructions:"
echo "==================="
echo ""
echo "Step 1: Extract TOTP URI (One-time setup)"
echo "  • Use 1Password, Bitwarden, or similar to scan the Certum QR code"
echo "  • Edit the TOTP entry to reveal the otpauth:// URI"
echo "  • Copy the complete URI including all parameters"
echo ""
echo "Step 2: Add GitHub Secrets"
echo "  • Go to your repository Settings → Secrets and variables → Actions"
echo "  • Add each secret with the exact names listed above"
echo "  • Paste the values carefully (no extra spaces)"
echo ""
echo "Step 3: Run the Workflow"
echo "  • Go to Actions → 'Certum Automated Signing - BREAKTHROUGH'"
echo "  • Click 'Run workflow'"
echo "  • Choose 'Use mock binary' for initial testing"
echo "  • Watch the magic happen! 🎉"
echo ""

echo "Breakthrough Benefits:"
echo "====================="
echo "✅ No manual TOTP entry required"
echo "✅ No 30-second timing pressure"
echo "✅ Fully automated cloud certificate access"
echo "✅ Complete CI/CD integration"
echo "✅ Production-ready signing pipeline"
echo ""

echo "Security Notes:"
echo "==============="
echo "• GitHub secrets are encrypted and only accessible to workflows"
echo "• TOTP URI is treated like any other authentication secret"
echo "• This follows industry standard practices for CI/CD automation"
echo "• Your certificate remains securely in Certum's cloud"
echo ""

# Check if running in CI
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    echo "🤖 Running in GitHub Actions - secrets should be configured"
    echo ""
    echo "Environment Check:"
    echo "  CERTUM_OTP_URI: $([ -n "${CERTUM_OTP_URI:-}" ] && echo "Present (${#CERTUM_OTP_URI} chars)" || echo "Missing")"
    echo "  CERTUM_USERID: $([ -n "${CERTUM_USERID:-}" ] && echo "Present" || echo "Missing")"
    echo "  CERTUM_CERTIFICATE_SHA1: $([ -n "${CERTUM_CERTIFICATE_SHA1:-}" ] && echo "Present" || echo "Missing")"
else
    echo "💻 Running locally - GitHub secrets not available here"
fi

echo ""
echo "🎯 Ready to revolutionize your code signing workflow!"
echo "🚀 This breakthrough eliminates the last manual step in Certum automation!"
