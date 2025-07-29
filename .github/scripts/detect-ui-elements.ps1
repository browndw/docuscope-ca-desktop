#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Detect UI elements in SimplySign Desktop login dialog
.DESCRIPTION
    This script analyzes the SimplySign Desktop window to find login UI elements
    including username fields, password fields, and buttons for credential injection.
    It also detects web browser controls and HTML elements.
.PARAMETER ProcessId
    Process ID of the SimplySign Desktop process
.PARAMETER TimeoutSeconds
    How long to wait for UI elements to appear
.PARAMETER DebugMode
    Enable verbose debugging output
#>

param(
    [int]$ProcessId = 0,
    [int]$TimeoutSeconds = 60,
    [switch]$DebugMode
)

$ErrorActionPreference = "Continue"

Write-Host "=== SIMPLYSIGN UI ELEMENT DETECTION ==="
Write-Host "Process ID: $ProcessId"
Write-Host "Timeout: $TimeoutSeconds seconds"
Write-Host "Debug mode: $DebugMode"
Write-Host ""

# Add Windows Forms and UI Automation assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Add-Type -AssemblyName System.Drawing

# StringBuilder is part of mscorlib and automatically available

# Windows API declarations for advanced window detection
Add-Type @"
using System;
using System.Runtime.InteropServices;
using System.Text;

public class WindowAPI {
    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern bool EnumWindows(EnumWindowsProc enumProc, IntPtr lParam);
    
    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern bool EnumChildWindows(IntPtr hWndParent, EnumWindowsProc lpEnumFunc, IntPtr lParam);
    
    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
    
    [DllImport("user32.dll", CharSet = CharSet.Auto)]
    public static extern int GetClassName(IntPtr hWnd, StringBuilder lpClassName, int nMaxCount);
    
    [DllImport("user32.dll")]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
    
    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);
    
    [DllImport("user32.dll")]
    public static extern bool GetWindowRect(IntPtr hWnd, out RECT lpRect);
    
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
}
"@

function Get-WindowInfo {
    param([IntPtr]$WindowHandle)
    
    try {
        $title = New-Object System.Text.StringBuilder 256
        $className = New-Object System.Text.StringBuilder 256
        $rect = New-Object WindowAPI+RECT
        
        [WindowAPI]::GetWindowText($WindowHandle, $title, 256) | Out-Null
        [WindowAPI]::GetClassName($WindowHandle, $className, 256) | Out-Null
        [WindowAPI]::GetWindowRect($WindowHandle, [ref]$rect) | Out-Null
        
        # Create hashtable explicitly to avoid any conflicts
        $windowInfo = @{}
        $windowInfo['Handle'] = $WindowHandle.ToInt64()
        $windowInfo['Title'] = $title.ToString()
        $windowInfo['ClassName'] = $className.ToString()
        $windowInfo['Visible'] = [WindowAPI]::IsWindowVisible($WindowHandle)
        $windowInfo['Left'] = $rect.Left
        $windowInfo['Top'] = $rect.Top
        $windowInfo['Right'] = $rect.Right
        $windowInfo['Bottom'] = $rect.Bottom
        $windowInfo['Width'] = $rect.Right - $rect.Left
        $windowInfo['Height'] = $rect.Bottom - $rect.Top
        
        return $windowInfo
        
    } catch {
        if ($DebugMode) {
            Write-Host "Error getting window info for handle $($WindowHandle): $($_.Exception.Message)"
        }
        # Return a clean error hashtable
        $errorInfo = @{}
        $errorInfo['Handle'] = $WindowHandle.ToInt64()
        $errorInfo['Title'] = ""
        $errorInfo['ClassName'] = ""
        $errorInfo['Visible'] = $false
        $errorInfo['Left'] = 0
        $errorInfo['Top'] = 0
        $errorInfo['Right'] = 0
        $errorInfo['Bottom'] = 0
        $errorInfo['Width'] = 0
        $errorInfo['Height'] = 0
        
        return $errorInfo
    }
}

function Find-SimplySignWindows {
    param([int]$TargetProcessId = 0)
    
    $windows = @()
    $callback = {
        param([IntPtr]$hWnd, [IntPtr]$lParam)
        
        $processId = 0
        [WindowAPI]::GetWindowThreadProcessId($hWnd, [ref]$processId) | Out-Null
        
        if ($TargetProcessId -eq 0 -or $processId -eq $TargetProcessId) {
            $windowInfo = Get-WindowInfo -WindowHandle $hWnd
            
            # Look for SimplySign related windows
            if ($windowInfo.Title -like "*SimplySign*" -or 
                $windowInfo.ClassName -like "*SimplySign*" -or
                $processId -eq $TargetProcessId) {
                
                $script:windows += $windowInfo
                
                if ($DebugMode) {
                    Write-Host "Found window: $($windowInfo.Title) [$($windowInfo.ClassName)]"
                }
            }
        }
        
        return $true
    }
    
    [WindowAPI]::EnumWindows($callback, [IntPtr]::Zero) | Out-Null
    return $windows
}

function Find-ChildWindows {
    param([IntPtr]$ParentHandle)
    
    $children = @()
    $callback = {
        param([IntPtr]$hWnd, [IntPtr]$lParam)
        
        $windowInfo = Get-WindowInfo -WindowHandle $hWnd
        $script:children += $windowInfo
        
        if ($DebugMode) {
            Write-Host "  Child window: $($windowInfo.Title) [$($windowInfo.ClassName)]"
        }
        
        return $true
    }
    
    [WindowAPI]::EnumChildWindows($ParentHandle, $callback, [IntPtr]::Zero) | Out-Null
    return $children
}

function Handle-UpdateDialog {
    param([array]$Windows)
    
    Write-Host "Checking for update dialogs..."
    
    foreach ($window in $Windows) {
        # Look for update-related dialogs
        if ($window.Title -like "*update*" -or 
            $window.Title -like "*version*" -or 
            $window.Title -like "*download*" -or
            $window.ClassName -like "*Dialog*" -or
            $window.ClassName -like "*MessageBox*") {
            
            Write-Host "Found potential update dialog: $($window.Title) [$($window.ClassName)]"
            
            try {
                # Try to find and click "No" button to skip update
                $automation = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$window.Handle)
                
                if ($automation) {
                    # Look for "No" button
                    $buttonCondition = New-Object System.Windows.Automation.AndCondition @(
                        [System.Windows.Automation.Condition]::TrueCondition,
                        [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
                    )
                    
                    $buttons = $automation.FindAll([System.Windows.Automation.TreeScope]::Descendants, $buttonCondition)
                    
                    foreach ($button in $buttons) {
                        $buttonName = $button.Current.Name
                        Write-Host "Found button: '$buttonName'"
                        
                        # Look for "No", "Cancel", "Skip", or "Later" buttons
                        if ($buttonName -like "*No*" -or 
                            $buttonName -like "*Cancel*" -or 
                            $buttonName -like "*Skip*" -or 
                            $buttonName -like "*Later*" -or
                            $buttonName -like "*Remind*") {
                            
                            Write-Host "Clicking '$buttonName' button to skip update..."
                            
                            # Get the invoke pattern to click the button
                            $invokePattern = $button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
                            if ($invokePattern) {
                                $invokePattern.Invoke()
                                Write-Host "Successfully clicked '$buttonName' button"
                                Start-Sleep -Seconds 2
                                return $true
                            }
                        }
                    }
                    
                    # If no "No" button found, try pressing Escape key
                    Write-Host "No 'No' button found, trying to press Escape key..."
                    Add-Type -AssemblyName System.Windows.Forms
                    [System.Windows.Forms.SendKeys]::SendWait("{ESC}")
                    Start-Sleep -Seconds 1
                    return $true
                }
                
            } catch {
                Write-Host "Could not interact with update dialog: $($_.Exception.Message)"
            }
        }
    }
    
    return $false
}

function Wait-ForLoginDialog {
    param([int]$ProcessId, [int]$MaxWaitSeconds = 30)
    
    Write-Host "Waiting for login dialog to appear (up to $MaxWaitSeconds seconds)..."
    
    $startTime = Get-Date
    $endTime = $startTime.AddSeconds($MaxWaitSeconds)
    
    while ((Get-Date) -lt $endTime) {
        $windows = Find-SimplySignWindows -TargetProcessId $ProcessId
        
        # Check if we have any dialogs that might be update prompts
        $updateHandled = Handle-UpdateDialog -Windows $windows
        
        if ($updateHandled) {
            Write-Host "Update dialog handled, waiting for login dialog..."
            Start-Sleep -Seconds 3
            continue
        }
        
        # Look for potential login dialogs
        foreach ($window in $windows) {
            if ($window.Title -like "*login*" -or 
                $window.Title -like "*sign*" -or 
                $window.Title -like "*auth*" -or
                $window.Title -like "*connect*" -or
                ($window.Width -gt 300 -and $window.Height -gt 200 -and $window.Visible)) {
                
                Write-Host "Potential login dialog found: $($window.Title)"
                return $windows
            }
        }
        
        Start-Sleep -Seconds 2
    }
    
    Write-Host "Timeout waiting for login dialog"
    return Find-SimplySignWindows -TargetProcessId $ProcessId
}
    param([IntPtr]$WindowHandle)
    
    $uiElements = @()
    
    try {
        # Try UI Automation approach
        $automation = [System.Windows.Automation.AutomationElement]::FromHandle($WindowHandle)
        
        if ($automation) {
            Write-Host "Analyzing UI elements with UI Automation..."
            
            # Find all descendants
            $condition = [System.Windows.Automation.Condition]::TrueCondition
            $elements = $automation.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition)
            
            foreach ($element in $elements) {
                try {
                    $properties = @{
                        ControlType = $element.Current.ControlType.ProgrammaticName
                        Name = $element.Current.Name
                        AutomationId = $element.Current.AutomationId
                        ClassName = $element.Current.ClassName
                        IsEnabled = $element.Current.IsEnabled
                        IsVisible = -not $element.Current.IsOffscreen
                        BoundingRectangle = $element.Current.BoundingRectangle
                    }
                    
                    # Look for input fields
                    if ($properties.ControlType -like "*Edit*" -or 
                        $properties.ControlType -like "*Text*" -or
                        $properties.Name -like "*user*" -or
                        $properties.Name -like "*pass*" -or
                        $properties.Name -like "*login*" -or
                        $properties.AutomationId -like "*user*" -or
                        $properties.AutomationId -like "*pass*" -or
                        $properties.AutomationId -like "*login*") {
                        
                        $properties.ElementType = "InputField"
                        $uiElements += $properties
                        
                        Write-Host "Found input field: $($properties.Name) [$($properties.ControlType)]"
                    }
                    
                    # Look for buttons
                    if ($properties.ControlType -like "*Button*" -or
                        $properties.Name -like "*OK*" -or
                        $properties.Name -like "*Login*" -or
                        $properties.Name -like "*Connect*" -or
                        $properties.Name -like "*Sign*") {
                        
                        $properties.ElementType = "Button"
                        $uiElements += $properties
                        
                        Write-Host "Found button: $($properties.Name) [$($properties.ControlType)]"
                    }
                    
                    # Look for web browser controls
                    if ($properties.ControlType -like "*Document*" -or
                        $properties.ControlType -like "*Pane*" -or
                        $properties.ClassName -like "*WebBrowser*" -or
                        $properties.ClassName -like "*Internet*" -or
                        $properties.ClassName -like "*Chrome*" -or
                        $properties.ClassName -like "*Edge*") {
                        
                        $properties.ElementType = "WebControl"
                        $uiElements += $properties
                        
                        Write-Host "Found web control: $($properties.Name) [$($properties.ClassName)]"
                    }
                    
                } catch {
                    if ($DebugMode) {
                        Write-Host "Could not analyze element: $($_.Exception.Message)"
                    }
                }
            }
        }
        
    } catch {
        Write-Host "UI Automation failed: $($_.Exception.Message)"
    }
    
    return $uiElements
}

function Take-Screenshot {
    param([string]$OutputPath = "screenshots")
    
    try {
        if (-not (Test-Path $OutputPath)) {
            New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
        }
        
        $screen = [System.Windows.Forms.Screen]::PrimaryScreen
        $bitmap = New-Object System.Drawing.Bitmap($screen.Bounds.Width, $screen.Bounds.Height)
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        
        $graphics.CopyFromScreen($screen.Bounds.X, $screen.Bounds.Y, 0, 0, $screen.Bounds.Size)
        
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $filename = "$OutputPath/screenshot_$timestamp.png"
        $bitmap.Save($filename, [System.Drawing.Imaging.ImageFormat]::Png)
        
        $graphics.Dispose()
        $bitmap.Dispose()
        
        Write-Host "Screenshot saved: $filename"
        return $filename
        
    } catch {
        Write-Host "Could not take screenshot: $($_.Exception.Message)"
        return $null
    }
}

# ==============================================================================
# MAIN DETECTION LOGIC
# ==============================================================================

Write-Host "Starting UI element detection..."
$detectionResults = @{
    Timestamp = Get-Date
    ProcessId = $ProcessId
    Windows = @()
    UIElements = @()
    Screenshots = @()
    Summary = @{
        WindowsFound = 0
        InputFieldsFound = 0
        ButtonsFound = 0
        WebControlsFound = 0
    }
}

# Find SimplySign windows and handle update dialogs
Write-Host "Searching for SimplySign windows..."
$windows = Wait-ForLoginDialog -ProcessId $ProcessId -MaxWaitSeconds $TimeoutSeconds

if ($windows.Count -eq 0) {
    Write-Host "No SimplySign windows found after waiting"
    if ($ProcessId -ne 0) {
        Write-Host "Trying to find any windows for process $ProcessId..."
        $windows = Find-SimplySignWindows -TargetProcessId $ProcessId
    }
}

$detectionResults.Windows = $windows
$detectionResults.Summary.WindowsFound = $windows.Count

Write-Host "Found $($windows.Count) SimplySign window(s)"

# Analyze each window
foreach ($window in $windows) {
    Write-Host ""
    Write-Host "=== ANALYZING WINDOW: $($window.Title) ==="
    Write-Host "Class: $($window.ClassName)"
    Write-Host "Size: $($window.Width)x$($window.Height)"
    Write-Host "Position: ($($window.Left),$($window.Top))"
    Write-Host "Visible: $($window.Visible)"
    
    # Find child windows
    Write-Host "Looking for child windows..."
    $childWindows = Find-ChildWindows -ParentHandle ([IntPtr]$window.Handle)
    Write-Host "Found $($childWindows.Count) child window(s)"
    
    # Analyze UI elements
    Write-Host "Analyzing UI elements..."
    $uiElements = Analyze-UIElements -WindowHandle ([IntPtr]$window.Handle)
    $detectionResults.UIElements += $uiElements
    
    # Count element types
    $inputFields = $uiElements | Where-Object { $_.ElementType -eq "InputField" }
    $buttons = $uiElements | Where-Object { $_.ElementType -eq "Button" }
    $webControls = $uiElements | Where-Object { $_.ElementType -eq "WebControl" }
    
    $detectionResults.Summary.InputFieldsFound += $inputFields.Count
    $detectionResults.Summary.ButtonsFound += $buttons.Count
    $detectionResults.Summary.WebControlsFound += $webControls.Count
    
    Write-Host "UI Elements found:"
    Write-Host "  Input fields: $($inputFields.Count)"
    Write-Host "  Buttons: $($buttons.Count)"
    Write-Host "  Web controls: $($webControls.Count)"
}

# Take screenshot
Write-Host ""
Write-Host "Taking screenshot for visual confirmation..."
$screenshot = Take-Screenshot
if ($screenshot) {
    $detectionResults.Screenshots += $screenshot
}

# Save detailed results
Write-Host ""
Write-Host "Saving detection results..."

$detectionResults | ConvertTo-Json -Depth 10 | Out-File -FilePath "ui_detection_results.json" -Encoding UTF8
Write-Host "Results saved to: ui_detection_results.json"

# Save detailed UI elements
$detectionResults.UIElements | ConvertTo-Json -Depth 5 | Out-File -FilePath "ui_elements_detailed.json" -Encoding UTF8
Write-Host "UI elements saved to: ui_elements_detailed.json"

# Save window hierarchy
$windowHierarchy = @{
    MainWindows = $windows
    ChildWindows = @()
}

foreach ($window in $windows) {
    $children = Find-ChildWindows -ParentHandle ([IntPtr]$window.Handle)
    $windowHierarchy.ChildWindows += @{
        ParentHandle = $window.Handle
        ParentTitle = $window.Title
        Children = $children
    }
}

$windowHierarchy | ConvertTo-Json -Depth 5 | Out-File -FilePath "window_hierarchy.json" -Encoding UTF8
Write-Host "Window hierarchy saved to: window_hierarchy.json"

# Generate summary
Write-Host ""
Write-Host "=== DETECTION SUMMARY ==="
Write-Host "Windows found: $($detectionResults.Summary.WindowsFound)"
Write-Host "Input fields found: $($detectionResults.Summary.InputFieldsFound)"
Write-Host "Buttons found: $($detectionResults.Summary.ButtonsFound)"
Write-Host "Web controls found: $($detectionResults.Summary.WebControlsFound)"

if ($detectionResults.Summary.InputFieldsFound -gt 0) {
    Write-Host ""
    Write-Host "SUCCESS: Input fields detected - credential injection possible!"
} elseif ($detectionResults.Summary.WebControlsFound -gt 0) {
    Write-Host ""
    Write-Host "WEB CONTROLS DETECTED: Login dialog may be HTML-based"
    Write-Host "Consider web automation approach for credential injection"
} else {
    Write-Host ""
    Write-Host "NO INPUT FIELDS DETECTED: May need alternative detection approach"
}

Write-Host ""
Write-Host "UI element detection completed"

return @{
    Success = $true
    WindowsFound = $detectionResults.Summary.WindowsFound
    InputFieldsFound = $detectionResults.Summary.InputFieldsFound
    ButtonsFound = $detectionResults.Summary.ButtonsFound
    WebControlsFound = $detectionResults.Summary.WebControlsFound
}
