#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Comprehensive analysis of all Certum/SimplySign settings to find login dialog controls
.DESCRIPTION
    This script performs a broad search for any settings that might control login dialog behavior,
    rather than searching for a specific macOS setting name. It looks for:
    - All Certum registry entries and their values
    - Settings with boolean patterns (0/1, Yes/No, true/false)
    - Keywords related to login, dialog, startup, cloud, auth
    - Config files in various locations
.PARAMETER OutputPrefix
    Prefix for output files (default: "settings_analysis")
.PARAMETER SearchTerms
    Comma-separated list of terms to search for (default: basic login-related terms)
#>

param(
    [string]$OutputPrefix = "settings_analysis",
    [string]$SearchTerms = "login,logon,dialog,startup,connect,cloud,auth,show,display,auto,trigger"
)

$ErrorActionPreference = "Continue"
$searchTermsArray = $SearchTerms -split ","

Write-Host "=== COMPREHENSIVE CERTUM SETTINGS ANALYSIS ==="
Write-Host "Output prefix: $OutputPrefix"
Write-Host "Search terms: $($searchTermsArray -join ', ')"
Write-Host ""

# Initialize results
$results = @{
    RegistryFindings = @()
    FileFindings = @()
    BooleanSettings = @()
    PotentialTriggers = @()
    AllCertumSettings = @()
    Summary = @{}
}

function Export-RegistrySection {
    param([string]$RegPath, [string]$OutputFile)
    
    try {
        $output = cmd /c "reg export `"$RegPath`" `"$OutputFile`" /y 2>&1"
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-Host "✓ Exported: $RegPath"
            return $true
        } else {
            Write-Host "- Not found: $RegPath"
            return $false
        }
    } catch {
        Write-Host "✗ Error exporting $RegPath : $($_.Exception.Message)"
        return $false
    }
}

function Search-RegistryForTerms {
    param([string]$RootPath, [array]$Terms)
    
    $findings = @()
    
    foreach ($term in $Terms) {
        try {
            Write-Host "  Searching for: $term"
            $output = cmd /c "reg query `"$RootPath`" /s /f `"$term`" 2>nul"
            
            if ($output -and $output.Count -gt 0) {
                $filteredOutput = $output | Where-Object { 
                    $_ -and 
                    $_ -notlike "*searching*" -and 
                    $_ -notlike "*End of search*" -and
                    $_ -notlike "*Error*"
                }
                
                if ($filteredOutput) {
                    $findings += @{
                        Term = $term
                        RootPath = $RootPath
                        Results = $filteredOutput
                    }
                    Write-Host "    ✓ Found matches for '$term'"
                }
            }
        } catch {
            Write-Host "    ✗ Search failed for '$term': $($_.Exception.Message)"
        }
    }
    
    return $findings
}

function Get-AllRegistryValues {
    param([string]$RegPath)
    
    $values = @()
    
    if (Test-Path $RegPath) {
        try {
            $properties = Get-ItemProperty -Path $RegPath -ErrorAction SilentlyContinue
            if ($properties) {
                $properties.PSObject.Properties | ForEach-Object {
                    if ($_.Name -notlike "PS*") {  # Skip PowerShell metadata
                        $values += @{
                            Name = $_.Name
                            Value = $_.Value
                            Type = $_.Value.GetType().Name
                            RegPath = $RegPath
                        }
                    }
                }
            }
        } catch {
            Write-Host "Could not read values from $RegPath"
        }
    }
    
    return $values
}

function Find-BooleanPatterns {
    param([array]$AllValues)
    
    $booleanSettings = @()
    
    foreach ($value in $AllValues) {
        $val = $value.Value
        $name = $value.Name
        
        # Check for boolean patterns
        $isBooleanLike = $false
        $booleanType = ""
        
        if ($val -eq 0 -or $val -eq 1) {
            $isBooleanLike = $true
            $booleanType = "DWORD (0/1)"
        } elseif ($val -eq "Yes" -or $val -eq "No") {
            $isBooleanLike = $true
            $booleanType = "String (Yes/No)"
        } elseif ($val -eq "true" -or $val -eq "false") {
            $isBooleanLike = $true
            $booleanType = "String (true/false)"
        } elseif ($val -eq "True" -or $val -eq "False") {
            $isBooleanLike = $true
            $booleanType = "String (True/False)"
        } elseif ($val -eq "on" -or $val -eq "off") {
            $isBooleanLike = $true
            $booleanType = "String (on/off)"
        }
        
        if ($isBooleanLike) {
            $booleanSettings += @{
                Name = $name
                Value = $val
                Type = $booleanType
                RegPath = $value.RegPath
                PotentialLoginTrigger = ($name -match "($($searchTermsArray -join '|'))")
            }
        }
    }
    
    return $booleanSettings
}

function Search-ConfigFiles {
    param([array]$Paths, [array]$Terms)
    
    $fileFindings = @()
    
    foreach ($basePath in $Paths) {
        if (Test-Path $basePath) {
            Write-Host "Searching config files in: $basePath"
            
            try {
                $configFiles = Get-ChildItem -Path $basePath -Recurse -Include "*.xml", "*.plist", "*.config", "*.ini", "*.json", "*.cfg", "*.properties", "*.txt", "*.dat" -ErrorAction SilentlyContinue
                
                foreach ($file in $configFiles) {
                    try {
                        $content = Get-Content -Path $file.FullName -Raw -ErrorAction SilentlyContinue
                        if ($content) {
                            foreach ($term in $Terms) {
                                if ($content -match $term) {
                                    $fileFindings += @{
                                        File = $file.FullName
                                        Term = $term
                                        Size = $file.Length
                                        LastModified = $file.LastWriteTime
                                    }
                                    Write-Host "  ✓ Found '$term' in: $($file.Name)"
                                    break  # Don't duplicate file entries
                                }
                            }
                        }
                    } catch {
                        # Skip files we can't read
                    }
                }
            } catch {
                Write-Host "  Error searching path: $($_.Exception.Message)"
            }
        } else {
            Write-Host "Path does not exist: $basePath"
        }
    }
    
    return $fileFindings
}

# ==============================================================================
# PHASE 1: COMPLETE REGISTRY DUMP OF CERTUM SECTIONS
# ==============================================================================

Write-Host "PHASE 1: EXPORTING ALL CERTUM REGISTRY SECTIONS"
Write-Host "================================================="

$registryPaths = @(
    "HKCU\Software\Certum",
    "HKCU\Software\SimplySign", 
    "HKCU\Software\SimplySignDesktop",
    "HKCU\Software\Asseco",
    "HKLM\SOFTWARE\Certum",
    "HKLM\SOFTWARE\SimplySign",
    "HKLM\SOFTWARE\SimplySignDesktop", 
    "HKLM\SOFTWARE\Asseco",
    "HKLM\SOFTWARE\WOW6432Node\Certum",
    "HKLM\SOFTWARE\WOW6432Node\SimplySign"
)

$exportedSections = @()
foreach ($regPath in $registryPaths) {
    $filename = "${OutputPrefix}_$(($regPath -replace '\\', '_' -replace ':', '')).reg"
    if (Export-RegistrySection -RegPath $regPath -OutputFile $filename) {
        $exportedSections += $regPath
    }
}

Write-Host ""
Write-Host "Exported $($exportedSections.Count) registry sections"

# ==============================================================================
# PHASE 2: SEARCH FOR LOGIN-RELATED TERMS IN REGISTRY
# ==============================================================================

Write-Host ""
Write-Host "PHASE 2: SEARCHING FOR LOGIN-RELATED TERMS"
Write-Host "============================================"

$searchRoots = @(
    "HKCU\Software\Certum",
    "HKLM\SOFTWARE\Certum"
)

foreach ($root in $searchRoots) {
    Write-Host "Searching under: $root"
    $findings = Search-RegistryForTerms -RootPath $root -Terms $searchTermsArray
    $results.RegistryFindings += $findings
    
    if ($findings.Count -gt 0) {
        Write-Host "  Found $($findings.Count) term matches"
    } else {
        Write-Host "  No matches found"
    }
}

# ==============================================================================
# PHASE 3: ANALYZE ALL CERTUM REGISTRY VALUES
# ==============================================================================

Write-Host ""
Write-Host "PHASE 3: ANALYZING ALL CERTUM REGISTRY VALUES"
Write-Host "=============================================="

$allValues = @()
$certumPaths = @(
    "HKCU:\Software\Certum",
    "HKLM:\SOFTWARE\Certum"
)

foreach ($path in $certumPaths) {
    if (Test-Path $path) {
        Write-Host "Reading all values from: $path"
        $values = Get-AllRegistryValues -RegPath $path
        $allValues += $values
        $results.AllCertumSettings += $values
        Write-Host "  Found $($values.Count) registry values"
        
        # Display interesting values
        foreach ($value in $values) {
            Write-Host "    $($value.Name) = $($value.Value) [$($value.Type)]"
        }
    }
}

# ==============================================================================
# PHASE 4: IDENTIFY BOOLEAN SETTINGS
# ==============================================================================

Write-Host ""
Write-Host "PHASE 4: IDENTIFYING BOOLEAN SETTINGS"
Write-Host "======================================"

$booleanSettings = Find-BooleanPatterns -AllValues $allValues
$results.BooleanSettings = $booleanSettings

Write-Host "Found $($booleanSettings.Count) boolean-like settings:"
foreach ($setting in $booleanSettings) {
    $marker = if ($setting.PotentialLoginTrigger) { " 🎯 POTENTIAL LOGIN TRIGGER" } else { "" }
    Write-Host "  $($setting.Name) = $($setting.Value) [$($setting.Type)]$marker"
}

# ==============================================================================
# PHASE 5: SEARCH CONFIG FILES
# ==============================================================================

Write-Host ""
Write-Host "PHASE 5: SEARCHING CONFIG FILES"
Write-Host "================================"

$searchPaths = @(
    "$env:APPDATA\Certum",
    "$env:APPDATA\SimplySign",
    "$env:LOCALAPPDATA\Certum",
    "$env:LOCALAPPDATA\SimplySign",
    "$env:USERPROFILE\Documents\Certum",
    "$env:USERPROFILE\Documents\SimplySign",
    "$env:ProgramFiles\Certum",
    "$env:ProgramFiles(x86)\Certum",
    "$env:PROGRAMDATA\Certum"
)

$fileFindings = Search-ConfigFiles -Paths $searchPaths -Terms $searchTermsArray
$results.FileFindings = $fileFindings

if ($fileFindings.Count -gt 0) {
    Write-Host "Found $($fileFindings.Count) config files with login-related terms:"
    foreach ($finding in $fileFindings) {
        Write-Host "  $($finding.File) [contains: $($finding.Term)]"
    }
} else {
    Write-Host "No config files found with search terms"
}

# ==============================================================================
# PHASE 6: GENERATE SUMMARY AND RECOMMENDATIONS
# ==============================================================================

Write-Host ""
Write-Host "PHASE 6: ANALYSIS SUMMARY"
Write-Host "=========================="

$potentialTriggers = $booleanSettings | Where-Object { $_.PotentialLoginTrigger }
$results.PotentialTriggers = $potentialTriggers

$results.Summary = @{
    TotalRegistryValues = $allValues.Count
    BooleanSettings = $booleanSettings.Count
    PotentialLoginTriggers = $potentialTriggers.Count
    ConfigFilesFound = $fileFindings.Count
    RegistrySearchHits = ($results.RegistryFindings | Measure-Object).Count
    RecommendedActions = @()
}

Write-Host "ANALYSIS RESULTS:"
Write-Host "  Total registry values found: $($results.Summary.TotalRegistryValues)"
Write-Host "  Boolean-like settings: $($results.Summary.BooleanSettings)"
Write-Host "  Potential login triggers: $($results.Summary.PotentialLoginTriggers)"
Write-Host "  Config files with keywords: $($results.Summary.ConfigFilesFound)"
Write-Host "  Registry search hits: $($results.Summary.RegistrySearchHits)"

Write-Host ""
Write-Host "🎯 RECOMMENDED ACTIONS:"

if ($potentialTriggers.Count -gt 0) {
    Write-Host "  1. TEST BOOLEAN SETTINGS - Found $($potentialTriggers.Count) promising candidates:"
    foreach ($trigger in $potentialTriggers) {
        $opposite = if ($trigger.Value -eq 0 -or $trigger.Value -eq "No" -or $trigger.Value -eq "false") { 
            "Try changing to 1/Yes/true" 
        } else { 
            "Try changing to 0/No/false" 
        }
        Write-Host "     - $($trigger.Name) = $($trigger.Value) → $opposite"
        $results.Summary.RecommendedActions += "Test setting: $($trigger.Name)"
    }
} else {
    Write-Host "  1. No obvious boolean triggers found in registry"
}

if ($fileFindings.Count -gt 0) {
    Write-Host "  2. EXAMINE CONFIG FILES - Found files with relevant keywords"
    $results.Summary.RecommendedActions += "Examine config files for settings"
} else {
    Write-Host "  2. No config files found with login keywords"
}

# Look for the specific "Autostart" setting we know exists
$autostartSetting = $allValues | Where-Object { $_.Name -eq "Autostart" }
if ($autostartSetting) {
    Write-Host "  3. FOUND 'Autostart' SETTING - This might control startup behavior!"
    Write-Host "     Current value: $($autostartSetting.Value)"
    Write-Host "     Location: $($autostartSetting.RegPath)"
    Write-Host "     Try changing from $($autostartSetting.Value) to $(if ($autostartSetting.Value -eq 0) { 1 } else { 0 })"
    $results.Summary.RecommendedActions += "Test Autostart setting"
}

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

Write-Host ""
Write-Host "SAVING ANALYSIS RESULTS..."

# Save detailed results as JSON
$jsonFile = "${OutputPrefix}_detailed_results.json"
$results | ConvertTo-Json -Depth 10 | Out-File -FilePath $jsonFile -Encoding UTF8
Write-Host "Detailed results saved to: $jsonFile"

# Save summary report
$summaryFile = "${OutputPrefix}_summary_report.txt"
$summaryContent = @"
CERTUM SETTINGS ANALYSIS SUMMARY
Generated: $(Get-Date)
=================================

SEARCH PARAMETERS:
- Search terms: $($searchTermsArray -join ', ')
- Registry paths checked: $($registryPaths.Count)
- File system paths checked: $($searchPaths.Count)

RESULTS:
- Total registry values: $($results.Summary.TotalRegistryValues)
- Boolean settings found: $($results.Summary.BooleanSettings)
- Potential login triggers: $($results.Summary.PotentialLoginTriggers)
- Config files with keywords: $($results.Summary.ConfigFilesFound)

POTENTIAL LOGIN TRIGGERS:
$($potentialTriggers | ForEach-Object { "- $($_.Name) = $($_.Value) [$($_.RegPath)]" } | Out-String)

RECOMMENDED NEXT STEPS:
$($results.Summary.RecommendedActions | ForEach-Object { "- $_" } | Out-String)

FILES GENERATED:
- $jsonFile (detailed JSON results)
- $summaryFile (this summary)
- ${OutputPrefix}_*.reg (registry exports)
"@

$summaryContent | Out-File -FilePath $summaryFile -Encoding UTF8
Write-Host "Summary report saved to: $summaryFile"

Write-Host ""
Write-Host "ANALYSIS COMPLETE"
Write-Host "Review the generated files for detailed findings and next steps."

# Return key findings for the workflow
return @{
    PotentialTriggers = $potentialTriggers
    AllSettings = $allValues
    ConfigFiles = $fileFindings
    Success = $true
}
