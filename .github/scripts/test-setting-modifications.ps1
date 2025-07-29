#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Test potential login dialog trigger settings by modifying them systematically
.DESCRIPTION
    This script takes the findings from analyze-all-certum-settings.ps1 and tests
    each potential setting by modifying it and checking if it triggers a login dialog.
.PARAMETER SettingsAnalysisFile
    Path to the JSON file from the settings analysis
.PARAMETER TestDurationSeconds
    How long to wait after each setting change to test for effects
.PARAMETER RestoreOriginal
    Whether to restore original values after testing
#>

param(
    [string]$SettingsAnalysisFile = "settings_analysis_detailed_results.json",
    [int]$TestDurationSeconds = 15,
    [switch]$RestoreOriginal
)

$ErrorActionPreference = "Continue"

Write-Host "=== SYSTEMATIC SETTING MODIFICATION TESTING ==="
Write-Host "Analysis file: $SettingsAnalysisFile"
Write-Host "Test duration: $TestDurationSeconds seconds per setting"
Write-Host "Restore original: $RestoreOriginal"
Write-Host ""

function Test-SettingChange {
    param(
        [string]$RegPath,
        [string]$Name,
        [object]$OriginalValue,
        [object]$NewValue,
        [int]$WaitSeconds
    )
    
    $testResult = @{
        SettingName = $Name
        RegPath = $RegPath
        OriginalValue = $OriginalValue
        NewValue = $NewValue
        ChangeSuccessful = $false
        ProcessCountBefore = 0
        ProcessCountAfter = 0
        ProcessNames = @()
        RestoreSuccessful = $false
        Timestamp = Get-Date
    }
    
    try {
        # Record baseline process count
        $processesBefore = Get-Process -Name "*SimplySign*" -ErrorAction SilentlyContinue
        $testResult.ProcessCountBefore = $processesBefore.Count
        
        Write-Host "  Testing: $Name = $OriginalValue → $NewValue"
        Write-Host "  Registry path: $RegPath"
        
        # Make the change
        if (Test-Path $RegPath) {
            Set-ItemProperty -Path $RegPath -Name $Name -Value $NewValue -ErrorAction Stop
            Write-Host "  Setting changed successfully"
            $testResult.ChangeSuccessful = $true
            
            # Wait for potential effects
            Write-Host "  Waiting $WaitSeconds seconds for effects..."
            Start-Sleep -Seconds $WaitSeconds
            
            # Check for new processes or dialogs
            $processesAfter = Get-Process -Name "*SimplySign*" -ErrorAction SilentlyContinue
            $testResult.ProcessCountAfter = $processesAfter.Count
            $testResult.ProcessNames = $processesAfter | ForEach-Object { $_.ProcessName }
            
            if ($testResult.ProcessCountAfter -gt $testResult.ProcessCountBefore) {
                Write-Host "  NEW PROCESS(ES) DETECTED! Setting may have triggered something!"
                $newProcesses = $processesAfter | Where-Object { $_.Id -notin ($processesBefore | ForEach-Object { $_.Id }) }
                foreach ($proc in $newProcesses) {
                    Write-Host "    New process: $($proc.ProcessName) (PID: $($proc.Id))"
                }
                $testResult.TriggeredNewProcess = $true
            } else {
                Write-Host "  - No new processes detected"
                $testResult.TriggeredNewProcess = $false
            }
            
            # Restore original value if requested
            if ($RestoreOriginal) {
                try {
                    Set-ItemProperty -Path $RegPath -Name $Name -Value $OriginalValue -ErrorAction Stop
                    Write-Host "  Restored original value"
                    $testResult.RestoreSuccessful = $true
                } catch {
                    Write-Host "  Failed to restore original value: $($_.Exception.Message)"
                    $testResult.RestoreSuccessful = $false
                }
            }
            
        } else {
            Write-Host "  Registry path not found: $RegPath"
        }
        
    } catch {
        Write-Host "  Failed to modify setting: $($_.Exception.Message)"
    }
    
    return $testResult
}

function Start-SimplySignAndTest {
    param([string]$TestDescription)
    
    Write-Host "  Test: $TestDescription"
    
    # Find SimplySign executable
    $possiblePaths = @(
        "${env:ProgramFiles}\Certum\SimplySign Desktop\SimplySignDesktop.exe",
        "${env:ProgramFiles(x86)}\Certum\SimplySign Desktop\SimplySignDesktop.exe"
    )
    
    $exePath = $null
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $exePath = $path
            break
        }
    }
    
    if (-not $exePath) {
        Write-Host "  SimplySign executable not found"
        return $false
    }
    
    try {
        # Start SimplySign
        $process = Start-Process -FilePath $exePath -WindowStyle Hidden -PassThru
        Write-Host "  Started SimplySign (PID: $($process.Id))"
        
        # Wait briefly for initialization
        Start-Sleep -Seconds 5
        
        # Check if it's still running
        if (-not $process.HasExited) {
            Write-Host "  SimplySign running normally"
            
            # Stop it
            $process | Stop-Process -Force
            Write-Host "  Stopped SimplySign"
            return $true
        } else {
            Write-Host "  SimplySign exited immediately (Exit Code: $($process.ExitCode))"
            return $false
        }
        
    } catch {
        Write-Host "  Error testing SimplySign: $($_.Exception.Message)"
        return $false
    }
}

# ==============================================================================
# LOAD ANALYSIS RESULTS
# ==============================================================================

if (-not (Test-Path $SettingsAnalysisFile)) {
    Write-Host "Analysis file not found: $SettingsAnalysisFile"
    Write-Host "Run analyze-all-certum-settings.ps1 first to generate this file."
    exit 1
}

Write-Host "Loading analysis results from: $SettingsAnalysisFile"
try {
    $analysisResults = Get-Content -Path $SettingsAnalysisFile -Raw | ConvertFrom-Json
    Write-Host "Analysis results loaded"
} catch {
    Write-Host "Failed to load analysis results: $($_.Exception.Message)"
    exit 1
}

# ==============================================================================
# IDENTIFY SETTINGS TO TEST
# ==============================================================================

Write-Host ""
Write-Host "IDENTIFYING SETTINGS TO TEST"
Write-Host "============================="

$settingsToTest = @()

# Add potential login triggers
if ($analysisResults.PotentialTriggers) {
    foreach ($trigger in $analysisResults.PotentialTriggers) {
        $settingsToTest += @{
            Name = $trigger.Name
            RegPath = $trigger.RegPath
            CurrentValue = $trigger.Value
            TestValue = if ($trigger.Value -eq 0) { 1 } elseif ($trigger.Value -eq 1) { 0 } 
                       elseif ($trigger.Value -eq "No") { "Yes" } elseif ($trigger.Value -eq "Yes") { "No" }
                       elseif ($trigger.Value -eq "false") { "true" } elseif ($trigger.Value -eq "true") { "false" }
                       else { "Unknown" }
            Priority = "HIGH"
            Reason = "Potential login trigger identified by analysis"
        }
    }
}

# Add the Autostart setting specifically
$autostartSetting = $analysisResults.AllCertumSettings | Where-Object { $_.Name -eq "Autostart" }
if ($autostartSetting -and $autostartSetting.Name -notin ($settingsToTest | ForEach-Object { $_.Name })) {
    $settingsToTest += @{
        Name = $autostartSetting.Name
        RegPath = $autostartSetting.RegPath
        CurrentValue = $autostartSetting.Value
        TestValue = if ($autostartSetting.Value -eq 0) { 1 } else { 0 }
        Priority = "HIGH"
        Reason = "Autostart setting - likely controls startup behavior"
    }
}

# Add other boolean settings as lower priority
if ($analysisResults.BooleanSettings) {
    foreach ($boolSetting in $analysisResults.BooleanSettings) {
        if ($boolSetting.Name -notin ($settingsToTest | ForEach-Object { $_.Name })) {
            $settingsToTest += @{
                Name = $boolSetting.Name
                RegPath = $boolSetting.RegPath
                CurrentValue = $boolSetting.Value
                TestValue = if ($boolSetting.Value -eq 0) { 1 } elseif ($boolSetting.Value -eq 1) { 0 }
                           elseif ($boolSetting.Value -eq "No") { "Yes" } elseif ($boolSetting.Value -eq "Yes") { "No" }
                           else { "Unknown" }
                Priority = "MEDIUM"
                Reason = "Boolean setting - potential configuration control"
            }
        }
    }
}

Write-Host "Found $($settingsToTest.Count) settings to test:"
$settingsToTest | Sort-Object Priority | ForEach-Object {
    Write-Host "  [$($_.Priority)] $($_.Name) = $($_.CurrentValue) → $($_.TestValue)"
    Write-Host "    Reason: $($_.Reason)"
    Write-Host "    Path: $($_.RegPath)"
}

if ($settingsToTest.Count -eq 0) {
    Write-Host "No settings identified for testing"
    exit 1
}

# ==============================================================================
# TEST SETTINGS SYSTEMATICALLY
# ==============================================================================

Write-Host ""
Write-Host "SYSTEMATIC SETTING TESTING"
Write-Host "==========================="

$testResults = @()
$significantFindings = @()

# Baseline test - start SimplySign normally
Write-Host ""
Write-Host "BASELINE TEST"
Write-Host "----------------"
$baselineWorking = Start-SimplySignAndTest -TestDescription "Baseline - normal SimplySign startup"

$settingIndex = 1
foreach ($setting in ($settingsToTest | Sort-Object Priority)) {
    Write-Host ""
    Write-Host "TEST $settingIndex/$($settingsToTest.Count) - $($setting.Priority) PRIORITY"
    Write-Host "================================================================"
    
    $testResult = Test-SettingChange -RegPath $setting.RegPath -Name $setting.Name -OriginalValue $setting.CurrentValue -NewValue $setting.TestValue -WaitSeconds $TestDurationSeconds
    $testResults += $testResult
    
    # Test SimplySign startup with modified setting
    $startupTest = Start-SimplySignAndTest -TestDescription "With $($setting.Name) = $($setting.TestValue)"
    $testResult | Add-Member -NotePropertyName "SimplySignStartupWorked" -NotePropertyValue $startupTest
    
    # Check for significant findings
    if ($testResult.TriggeredNewProcess -or (-not $startupTest -and $baselineWorking)) {
        $significantFindings += $testResult
        Write-Host "  SIGNIFICANT FINDING - This setting may affect login behavior!"
    }
    
    Write-Host "    Result summary:"
    Write-Host "    Setting change: $(if ($testResult.ChangeSuccessful) { "YES" } else { "NO" })"
    Write-Host "    New processes: $(if ($testResult.TriggeredNewProcess) { "YES" } else { "NO" })"
    Write-Host "    SimplySign startup: $(if ($startupTest) { "Normal" } else { "Different" })"
    Write-Host "    Restore: $(if ($testResult.RestoreSuccessful) { "YES" } else { "NO" })"
    
    $settingIndex++
}

# ==============================================================================
# SAVE RESULTS AND GENERATE REPORT
# ==============================================================================

Write-Host ""
Write-Host "SAVING TEST RESULTS"
Write-Host "==================="

$testReport = @{
    TestParameters = @{
        AnalysisFile = $SettingsAnalysisFile
        TestDuration = $TestDurationSeconds
        RestoreOriginal = $RestoreOriginal
        Timestamp = Get-Date
    }
    SettingsTested = $settingsToTest.Count
    TestResults = $testResults
    SignificantFindings = $significantFindings
    Summary = @{
        SuccessfulChanges = ($testResults | Where-Object { $_.ChangeSuccessful }).Count
        ProcessTriggered = ($testResults | Where-Object { $_.TriggeredNewProcess }).Count
        StartupAffected = ($testResults | Where-Object { -not $_.SimplySignStartupWorked }).Count
    }
}

# Save detailed results
$resultsFile = "setting_modification_test_results.json"
$testReport | ConvertTo-Json -Depth 10 | Out-File -FilePath $resultsFile -Encoding UTF8
Write-Host "Test results saved to: $resultsFile"

# Generate summary report
$summaryFile = "setting_modification_summary.txt"
$summaryContent = @"
SETTING MODIFICATION TEST SUMMARY
Generated: $(Get-Date)
=================================

TEST PARAMETERS:
- Settings tested: $($testReport.SettingsTested)
- Test duration per setting: $TestDurationSeconds seconds
- Restore original values: $RestoreOriginal

RESULTS OVERVIEW:
- Successful setting changes: $($testReport.Summary.SuccessfulChanges)/$($testReport.SettingsTested)
- Settings that triggered new processes: $($testReport.Summary.ProcessTriggered)
- Settings that affected startup: $($testReport.Summary.StartupAffected)

SIGNIFICANT FINDINGS:
$($significantFindings | ForEach-Object { 
    "- $($_.SettingName): $($_.OriginalValue) → $($_.NewValue)"
    "  Triggered process: $(if ($_.TriggeredNewProcess) { "YES" } else { "No" })"
    "  Affected startup: $(if (-not $_.SimplySignStartupWorked) { "YES" } else { "No" })"
    ""
} | Out-String)

DETAILED RESULTS:
$($testResults | ForEach-Object {
    "Setting: $($_.SettingName)"
    "  Path: $($_.RegPath)" 
    "  Change: $($_.OriginalValue) → $($_.NewValue)"
    "  Success: $(if ($_.ChangeSuccessful) { "YES" } else { "NO" })"
    "  New process: $(if ($_.TriggeredNewProcess) { "YES" } else { "NO" })"
    "  Startup OK: $(if ($_.SimplySignStartupWorked) { "YES" } else { "NO" })"
    "  Restored: $(if ($_.RestoreSuccessful) { "YES" } else { "NO" })"
    ""
} | Out-String)

NEXT STEPS:
$( if ($significantFindings.Count -gt 0) {
    "1. Focus on the $($significantFindings.Count) significant finding(s) above"
    "2. Test these settings in combination"
    "3. Try keeping the promising settings changed and test login detection"
} else {
    "1. No obvious login triggers found in tested settings"
    "2. May need to look for file-based configurations"
    "3. Consider testing with user interaction or different startup modes"
})
"@

$summaryContent | Out-File -FilePath $summaryFile -Encoding UTF8
Write-Host "Summary report saved to: $summaryFile"

Write-Host ""
Write-Host "SETTING MODIFICATION TESTING COMPLETE"

if ($significantFindings.Count -gt 0) {
    Write-Host "Found $($significantFindings.Count) potentially significant setting(s)!"
    Write-Host "Review the summary report for next steps."
} else {
    Write-Host "No obvious login triggers found in registry settings."
    Write-Host "May need alternative approach (config files, user interaction, etc.)"
}

return @{
    SignificantFindings = $significantFindings
    AllResults = $testResults
    Success = $true
}
