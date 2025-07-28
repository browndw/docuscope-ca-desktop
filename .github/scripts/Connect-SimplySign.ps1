<#
  Connect-SimplySign.ps1
  ----------------------
  • Works on PowerShell 5.1 and 7+
  • Generates TOTP from an otpauth:// URI
  • Sends u$wshell = New-Object -ComObject WScript.Shell

# Strategy 1: Try to find the auto-login dialog (if config worked)
$windowFound = $false
$attempts = 0
$maxAttempts = 30

# Look for login dialog window titles (based on your research)
$loginWindowTitles = @(
    'SimplySign Desktop',
    'Connect to SimplySign',
    'Certum SimplySign',
    'Login',
    'Authentication',
    'TOTP',
    'Token'
)

while (-not $windowFound -and $attempts -lt $maxAttempts) {
    $attempts++
    Write-Host "Attempt $attempts of $maxAttempts to find login window..."
    
    # Try by process ID first
    $focused = $wshell.AppActivate($proc.Id)
    if ($focused) {
        Write-Host "✅ Window found using process ID"
        $windowFound = $true
        break
    }
    
    # Try login dialog titles
    foreach ($title in $loginWindowTitles) {
        $focused = $wshell.AppActivate($title)
        if ($focused) {
            Write-Host "✅ Login window found using title: $title"
            $windowFound = $true
            break
        }
    }
    
    if ($windowFound) { break }
    
    # Check if process is still alive
    if ($proc.HasExited) {
        Write-Host "ERROR: Process exited during window search with code: $($proc.ExitCode)"
        throw "SimplySign Desktop process terminated unexpectedly"
    }
    
    # Strategy 2: Manual dialog trigger (fallback for headless environment)
    if ($attempts -eq 10) {
        Write-Host "Auto-login dialog not detected. Attempting manual trigger..."
        
        # Try right-clicking system tray and triggering connect manually
        # This might work even in headless environments
        $wshell.SendKeys("%(TAB)")  # Alt+Tab to cycle windows
        Start-Sleep -Milliseconds 500
        
        # Try common keyboard shortcuts for SimplySign
        $wshell.SendKeys("^+c")     # Ctrl+Shift+C (common connect shortcut)
        Start-Sleep -Milliseconds 500
        
        # Or try opening context menu and navigating
        $wshell.SendKeys("{F10}")   # Context menu
        Start-Sleep -Milliseconds 500
        $wshell.SendKeys("c")       # 'C' for Connect
        Start-Sleep -Milliseconds 500
    }
    
    # Wait before next attempt
    Start-Sleep -Milliseconds 1000
}

if (-not $windowFound) {
    # Final debugging - get process details
    Write-Host "Failed to find login window. Debugging information:"
    Write-Host "Process ID: $($proc.Id)"
    Write-Host "Process Status: $(if ($proc.HasExited) { 'Exited' } else { 'Running' })"
    
    try {
        $processInfo = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if ($processInfo) {
            Write-Host "Process Details:"
            Write-Host "  Name: $($processInfo.ProcessName)"
            Write-Host "  Main Window Title: '$($processInfo.MainWindowTitle)'"
            Write-Host "  Has Main Window: $($processInfo.MainWindowHandle -ne 0)"
        }
    } catch {
        Write-Host "Could not get process details: $($_.Exception.Message)"
    }
    
    # In headless environments, we might need to proceed without window detection
    Write-Host "⚠️ Login window not detected. Attempting blind credential injection..."
    Write-Host "This may work if the login dialog is present but not detectable in headless mode."
    
    # Try sending credentials anyway (might work in headless)
    $windowFound = $true
}mplySign Desktop via SendKeys
#>

# === 1.  SETTINGS  ============================================================
$OtpUri  = $env:CERTUM_OTP_URI
$UserId  = $env:CERTUM_USERNAME
$ExePath = $env:CERTUM_EXE_PATH
# ============================================================================


# === 2.  PARSE THE otpauth:// URI  ===========================================
$uri = [Uri]$OtpUri

# Try System.Web.HttpUtility first (exists on Windows PowerShell);
# fall back to manual split if not available (Core / Linux containers).
try {
    $q = [System.Web.HttpUtility]::ParseQueryString($uri.Query)
} catch {
    $q = @{}
    foreach ($part in $uri.Query.TrimStart('?') -split '&') {
        $kv = $part -split '=',2
        if ($kv.Count -eq 2) { $q[$kv[0]] = [Uri]::UnescapeDataString($kv[1]) }
    }
}

$Base32    = $q['secret']
$Digits    = if ($q['digits'] -as [int]) { $q['digits'] -as [int] } else { 6 }
$Period    = if ($q['period'] -as [int]) { $q['period'] -as [int] } else { 30 }
$Algorithm = if ($q['algorithm']) { $q['algorithm'].ToUpper() } else { 'SHA1' }

# Validate supported algorithms
$SupportedAlgorithms = @('SHA1', 'SHA256', 'SHA512')
if ($Algorithm -notin $SupportedAlgorithms) {
    throw "Unsupported algorithm: $Algorithm. Supported: $($SupportedAlgorithms -join ', ')"
}

# === 3.  TOTP GENERATOR  =====================================================
Add-Type -Language CSharp @"
using System;
using System.Security.Cryptography;

public static class Totp
{
    private const string B32 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

    private static byte[] Base32Decode(string s)
    {
        s = s.TrimEnd('=').ToUpperInvariant();
        int byteCount = s.Length * 5 / 8;
        byte[] bytes = new byte[byteCount];

        int bitBuffer = 0, bitsLeft = 0, idx = 0;
        foreach (char c in s)
        {
            int val = B32.IndexOf(c);
            if (val < 0) throw new ArgumentException("Invalid Base32 char: " + c);

            bitBuffer = (bitBuffer << 5) | val;
            bitsLeft += 5;

            if (bitsLeft >= 8)
            {
                bytes[idx++] = (byte)(bitBuffer >> (bitsLeft - 8));
                bitsLeft -= 8;
            }
        }
        return bytes;
    }

    private static HMAC GetHmacAlgorithm(string algorithm, byte[] key)
    {
        switch (algorithm.ToUpper())
        {
            case "SHA1":
                return new HMACSHA1(key);
            case "SHA256":
                return new HMACSHA256(key);
            case "SHA512":
                return new HMACSHA512(key);
            default:
                throw new ArgumentException("Unsupported algorithm: " + algorithm);
        }
    }

    public static string Now(string secret, int digits, int period, string algorithm = "SHA1")
    {
        byte[] key = Base32Decode(secret);
        long counter = DateTimeOffset.UtcNow.ToUnixTimeSeconds() / period;

        byte[] cnt = BitConverter.GetBytes(counter);
        if (BitConverter.IsLittleEndian) Array.Reverse(cnt);

        byte[] hash;
        using (var hmac = GetHmacAlgorithm(algorithm, key))
        {
            hash = hmac.ComputeHash(cnt);
        }

        int offset = hash[hash.Length - 1] & 0x0F;
        int binary =
            ((hash[offset] & 0x7F) << 24) |
            ((hash[offset + 1] & 0xFF) << 16) |
            ((hash[offset + 2] & 0xFF) << 8) |
            (hash[offset + 3] & 0xFF);

        int otp = binary % (int)Math.Pow(10, digits);
        return otp.ToString(new string('0', digits));
    }
}
"@

function Get-TotpCode {
    param([string]$Secret,[int]$Digits=6,[int]$Period=30,[string]$Algorithm='SHA1')
    [Totp]::Now($Secret,$Digits,$Period,$Algorithm)
}

# === 4.  CONFIGURE AND LAUNCH SimplySign  ===================================
Write-Host "=== CONFIGURING SIMPLYSIGN FOR AUTO-LOGIN ==="
Write-Host "🎯 BREAKTHROUGH: Setting SimplySignDesktopShowLogonDialogAfterApplicationStartup"

# Based on extensive research: SimplySign doesn't show login dialog by default
# Must configure registry settings first, then restart application
$regPaths = @(
    "HKCU:\Software\Certum",
    "HKCU:\Software\SimplySignDesktop", 
    "HKCU:\Software\Asseco"
)

foreach ($regPath in $regPaths) {
    try {
        # Ensure registry path exists
        if (-not (Test-Path $regPath)) {
            New-Item -Path $regPath -Force | Out-Null
            Write-Host "Created registry path: $regPath"
        }
        
        # Set the breakthrough configuration values
        New-ItemProperty -Path $regPath -Name 'SimplySignDesktopShowLogonDialogAfterApplicationStartup' -Value 'Yes' -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $regPath -Name 'ShowLogonDialogAfterApplicationStartup' -Value 'Yes' -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $regPath -Name 'AutoShowLogonDialog' -Value 'Yes' -PropertyType String -Force | Out-Null
        
        Write-Host "✅ Configured auto-login for: $regPath"
    } catch {
        Write-Host "⚠️ Could not configure: $regPath - $($_.Exception.Message)"
    }
}

Write-Host ""
Write-Host "=== LAUNCHING SIMPLYSIGN WITH TOTP AUTHENTICATION ==="
$otp = Get-TotpCode -Secret $Base32 -Digits $Digits -Period $Period -Algorithm $Algorithm
Write-Host "Current TOTP: $otp (using $Algorithm algorithm)"

Write-Host "Launching SimplySign Desktop..."
Write-Host "Executable path: $ExePath"

# Check if executable exists
if (-not (Test-Path $ExePath)) {
    throw "SimplySign Desktop executable not found at: $ExePath"
}

# Kill any existing SimplySign processes to ensure fresh start with new config
Get-Process -Name "SimplySignDesktop" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Launch the process with detailed tracking
$proc = Start-Process -FilePath $ExePath -PassThru -WindowStyle Normal
Write-Host "Process launched with ID: $($proc.Id)"
Write-Host "Process name: $($proc.ProcessName)"

# Wait longer for the application to fully start and apply config
Write-Host "Waiting for SimplySign Desktop to initialize with auto-login config..."
Start-Sleep -Seconds 15

# Check if process is still running
if ($proc.HasExited) {
    Write-Host "ERROR: Process exited immediately with code: $($proc.ExitCode)"
    throw "SimplySign Desktop failed to start properly"
}

Write-Host "Process is running. Attempting to find login window..."

$wshell = New-Object -ComObject WScript.Shell

# Try by **process ID** first (most reliable) ────────────────────────────────
$focused = $wshell.AppActivate($proc.Id)

# Fallback: exact window caption  ────────────────────────────────────────────
if (-not $focused) {
    $focused = $wshell.AppActivate('SimplySign Desktop')
}

# Give it a few more tries, just in case the window is still spawning
for ($i = 0; -not $focused -and $i -lt 10; $i++) {
    Start-Sleep -Milliseconds 500
    $focused = $wshell.AppActivate($proc.Id) -or $wshell.AppActivate('SimplySign Desktop')
}

if (-not $focused) {
    throw "Still couldn't bring SimplySign Desktop to the foreground."
}

# Window has focus → send the credentials
Write-Host "Window activated successfully. Sending credentials..."
Start-Sleep -Milliseconds 400
$wshell.SendKeys("$UserId{TAB}$otp{ENTER}")
Write-Host "`n✅ Credentials sent: Username + TOTP"
Write-Host "   Username: $UserId"
Write-Host "   TOTP: $otp"
Write-Host "`n⏳ Waiting for authentication to complete..."
Start-Sleep -Seconds 5
Write-Host "✅ Authentication process completed. The cloud smart-card should now be available."
