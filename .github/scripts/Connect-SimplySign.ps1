<#
  Connect-SimplySign.ps1
  ----------------------
  • Works on PowerShell 5.1 and 7+
  • Generates TOTP from an otpauth:// URI
  • Sends username + OTP to SimplySign Desktop via SendKeys
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

# === 4.  LAUNCH SimplySign AND SEND CREDENTIALS  =============================
$otp = Get-TotpCode -Secret $Base32 -Digits $Digits -Period $Period -Algorithm $Algorithm
Write-Host "Current TOTP: $otp (using $Algorithm algorithm)"

$proc = Start-Process -FilePath $ExePath -PassThru
Write-Host "Waiting for SimplySign Desktop to appear…"
Start-Sleep -Seconds 5      # crude warm-up; tweak as needed

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
    throw "Still couldn’t bring SimplySign Desktop to the foreground."
}

# Window has focus → send the credentials
Start-Sleep -Milliseconds 400
$wshell.SendKeys("$otp{ENTER}")
Write-Host "`n✅  Credentials sent.  The cloud smart-card should mount in a few seconds."
