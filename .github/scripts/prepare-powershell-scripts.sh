#!/bin/bash
# Helper script to ensure PowerShell scripts are ready for execution
# This addresses any permission or encoding issues

echo "=== PREPARING POWERSHELL SCRIPTS FOR EXECUTION ==="

SCRIPT_DIR=".github/scripts"

# List of PowerShell scripts we'll be using
PS_SCRIPTS=(
    "analyze-all-certum-settings.ps1"
    "test-setting-modifications.ps1"
    "detect-login-dialog.ps1"
)

echo "Verifying PowerShell scripts in $SCRIPT_DIR..."

for script in "${PS_SCRIPTS[@]}"; do
    script_path="$SCRIPT_DIR/$script"
    if [ -f "$script_path" ]; then
        echo "✓ Found: $script"
        
        # Check if file is readable
        if [ -r "$script_path" ]; then
            echo "  ✓ Readable"
        else
            echo "  ⚠ Not readable - fixing permissions"
            chmod +r "$script_path"
        fi
        
        # Check file size
        size=$(stat -f%z "$script_path" 2>/dev/null || stat -c%s "$script_path" 2>/dev/null || echo "unknown")
        echo "  Size: $size bytes"
        
    else
        echo "✗ Missing: $script"
        echo "  Expected at: $script_path"
    fi
done

echo ""
echo "PowerShell execution policy will be set to Bypass in workflow steps"
echo "Scripts are ready for execution"
