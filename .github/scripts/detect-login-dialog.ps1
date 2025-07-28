param(
  [int]$TimeoutSeconds = 180,
  [string]$DebugMode = "false"
)

# Convert string to boolean
$DebugModeBoolean = ($DebugMode -eq "true" -or $DebugMode -eq "True" -or $DebugMode -eq $true)

Write-Host "Starting SimplySign Desktop for dialog detection..."

# Kill any existing instances
Get-Process -Name "SimplySignDesktop" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

# Start SimplySign Desktop
$exePath = "C:\Program Files\Certum\SimplySign Desktop\SimplySignDesktop.exe"
if (-not (Test-Path $exePath)) {
  Write-Host "ERROR: SimplySign Desktop not found at: $exePath"
  exit 1
}

Write-Host "SUCCESS: Launching SimplySign Desktop from: $exePath"
$proc = Start-Process -FilePath $exePath -PassThru -WindowStyle Normal
Write-Host "SUCCESS: Process started - PID: $($proc.Id)"

# Initialize detection tracking
$detectionResults = @{
  ProcessMethod = $false
  WindowTitleMethod = $false
  AllWindowsMethod = $false
  WebViewMethod = $false
  ChildWindowMethod = $false
  NetworkMethod = $false
}

$startTime = Get-Date
$endTime = $startTime.AddSeconds($TimeoutSeconds)
$detectionAttempts = 0

Write-Host "Detection period: $TimeoutSeconds seconds"
Write-Host "Starting comprehensive detection loop..."

while ((Get-Date) -lt $endTime) {
  $detectionAttempts++
  $elapsed = ((Get-Date) - $startTime).TotalSeconds
  
  if ($DebugModeBoolean -or ($detectionAttempts % 10 -eq 0)) {
    Write-Host "[$([int]$elapsed)s] Detection attempt $detectionAttempts..."
  }
  
  # Check if process is still alive
  if ($proc.HasExited) {
    Write-Host "ERROR: SimplySign Desktop process exited with code: $($proc.ExitCode)"
    break
  }
  
  # METHOD 1: Standard Process Window Detection
  try {
    $processInfo = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
    if ($processInfo -and $processInfo.MainWindowTitle -and $processInfo.MainWindowTitle.Trim() -ne "") {
      if (-not $detectionResults.ProcessMethod) {
        Write-Host "SUCCESS METHOD 1: Process window detected"
        Write-Host "   Window Title: '$($processInfo.MainWindowTitle)'"
        Write-Host "   Window Handle: $($processInfo.MainWindowHandle)"
        $detectionResults.ProcessMethod = $true
      }
    }
  } catch {
    if ($DebugModeBoolean) { Write-Host "   Method 1 error: $($_.Exception.Message)" }
  }
  
  # METHOD 2: Window Title Enumeration
  try {
    $loginTitles = @(
      'SimplySign Desktop',
      'Connect to SimplySign',
      'Certum SimplySign',
      'Login', 'Authentication', 'TOTP', 'Token',
      'OAuth2', 'Authorization', 'Sign In'
    )
    
    foreach ($title in $loginTitles) {
      $windows = Get-Process | Where-Object { $_.MainWindowTitle -like "*$title*" }
      if ($windows) {
        if (-not $detectionResults.WindowTitleMethod) {
          Write-Host "SUCCESS METHOD 2: Window title match found"
          Write-Host "   Matched title: '$title'"
          Write-Host "   Process: $($windows[0].ProcessName)"
          $detectionResults.WindowTitleMethod = $true
        }
      }
    }
  } catch {
    if ($DebugModeBoolean) { Write-Host "   Method 2 error: $($_.Exception.Message)" }
  }
  
  # METHOD 3: All Windows Enumeration
  try {
    # Skip complex window enumeration for now - placeholder method
    # This would typically use Win32 APIs to enumerate all windows
    if (-not $detectionResults.AllWindowsMethod) {
      $detectionResults.AllWindowsMethod = $false  # Placeholder - not implemented
    }
  } catch {
    if ($DebugModeBoolean) { Write-Host "   Method 3 error: $($_.Exception.Message)" }
  }
  
  # METHOD 4: Web View Detection
  try {
    # Check for WebView2 or embedded browser processes
    $webViewProcesses = Get-Process | Where-Object { 
      $_.ProcessName -like "*WebView*" -or 
      $_.ProcessName -like "*Chrome*" -or 
      $_.ProcessName -eq "msedgewebview2" 
    }
    
    if ($webViewProcesses -and -not $detectionResults.WebViewMethod) {
      Write-Host "SUCCESS METHOD 4: WebView/Browser process detected"
      foreach ($webProc in $webViewProcesses) {
        Write-Host "   WebView Process: $($webProc.ProcessName) (PID: $($webProc.Id))"
      }
      $detectionResults.WebViewMethod = $true
    }
  } catch {
    if ($DebugModeBoolean) { Write-Host "   Method 4 error: $($_.Exception.Message)" }
  }
  
  # METHOD 5: Child Window Detection
  try {
    # Check for child processes or windows spawned by SimplySign
    $childProcesses = Get-WmiObject Win32_Process | Where-Object { $_.ParentProcessId -eq $proc.Id }
    if ($childProcesses -and -not $detectionResults.ChildWindowMethod) {
      Write-Host "SUCCESS METHOD 5: Child process detected"
      foreach ($child in $childProcesses) {
        Write-Host "   Child Process: $($child.Name) (PID: $($child.ProcessId))"
      }
      $detectionResults.ChildWindowMethod = $true
    }
  } catch {
    if ($DebugModeBoolean) { Write-Host "   Method 5 error: $($_.Exception.Message)" }
  }
  
  # METHOD 6: Network Activity Detection
  try {
    $connections = netstat -an | Select-String 'webnotarius'
    if ($connections -and -not $detectionResults.NetworkMethod) {
      Write-Host "SUCCESS METHOD 6: OAuth2 network activity detected"
      Write-Host "   Connections: $connections"
      $detectionResults.NetworkMethod = $true
    }
  } catch {
    if ($DebugModeBoolean) { Write-Host "   Method 6 error: $($_.Exception.Message)" }
  }
  
  # Brief pause between detection attempts
  Start-Sleep -Milliseconds 1000
}

# Final Results Summary
Write-Host ""
Write-Host "=== DETECTION RESULTS SUMMARY ==="
Write-Host "Total detection attempts: $detectionAttempts"
Write-Host "Detection duration: $([int]((Get-Date) - $startTime).TotalSeconds) seconds"
Write-Host ""

$successCount = 0
foreach ($method in $detectionResults.GetEnumerator()) {
  $status = if ($method.Value) { "SUCCESS"; $successCount++ } else { "FAILED" }
  Write-Host "$status - $($method.Key)"
}

Write-Host ""
Write-Host "Success Rate: $successCount / $($detectionResults.Count) methods"

if ($successCount -eq 0) {
  Write-Host "NO DETECTION METHODS SUCCESSFUL"
  Write-Host "   This indicates login dialog may not appear in headless environment"
} elseif ($successCount -lt $detectionResults.Count) {
  Write-Host "PARTIAL DETECTION SUCCESS"
  Write-Host "   Some methods worked - authentication may be possible with correct approach"
} else {
  Write-Host "ALL DETECTION METHODS SUCCESSFUL"
  Write-Host "   Login dialog is fully detectable in this environment"
}

# Cleanup
if (-not $proc.HasExited) {
  Write-Host "Stopping SimplySign Desktop process..."
  $proc | Stop-Process -Force
}

# Save results for workflow
$detectionResults | ConvertTo-Json | Out-File -FilePath "detection_results.json"
Write-Host "Detection results saved to detection_results.json"

return $successCount
