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
                
                # Create a clean copy of the window info to avoid hashtable conflicts
                $cleanWindowInfo = @{
                    Handle = $windowInfo['Handle']
                    Title = $windowInfo['Title']
                    ClassName = $windowInfo['ClassName']
                    Visible = $windowInfo['Visible']
                    Left = $windowInfo['Left']
                    Top = $windowInfo['Top']
                    Right = $windowInfo['Right']
                    Bottom = $windowInfo['Bottom']
                    Width = $windowInfo['Width']
                    Height = $windowInfo['Height']
                }
                
                $script:windows = $script:windows + @($cleanWindowInfo)
                
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
        
        # Create a clean copy to avoid hashtable conflicts
        $cleanWindowInfo = @{
            Handle = $windowInfo['Handle']
            Title = $windowInfo['Title']
            ClassName = $windowInfo['ClassName']
            Visible = $windowInfo['Visible']
            Left = $windowInfo['Left']
            Top = $windowInfo['Top']
            Right = $windowInfo['Right']
            Bottom = $windowInfo['Bottom']
            Width = $windowInfo['Width']
            Height = $windowInfo['Height']
        }
        
        $script:children = $script:children + @($cleanWindowInfo)
        
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
        if ($DebugMode) {
            Write-Host "Examining window: '$($window.Title)' [$($window.ClassName)] Size: $($window.Width)x$($window.Height)"
        }
        
        # Look for update-related dialogs - be more aggressive in detection
        $isUpdateDialog = $false
        
        # Check title patterns
        if ($window.Title -like "*update*" -or 
            $window.Title -like "*version*" -or 
            $window.Title -like "*download*" -or
            $window.Title -like "*SimplySign*") {
            $isUpdateDialog = $true
        }
        
        # Check class patterns for standard Windows dialogs
        if ($window.ClassName -like "*#32770*" -or 
            $window.ClassName -like "*Dialog*" -or 
            $window.ClassName -like "*MessageBox*") {
            # For standard dialog boxes, assume it might be an update dialog if it's small
            if ($window.Width -lt 500 -and $window.Height -lt 300) {
                $isUpdateDialog = $true
            }
        }
        
        if ($isUpdateDialog) {
            Write-Host "Found potential update dialog: '$($window.Title)' [$($window.ClassName)]"
            Write-Host "  Position: ($($window.Left),$($window.Top)) Size: $($window.Width)x$($window.Height)"
            
            try {
                # Method 1: Try UI Automation to find buttons
                $automation = [System.Windows.Automation.AutomationElement]::FromHandle([IntPtr]$window.Handle)
                
                if ($automation) {
                    Write-Host "Analyzing dialog with UI Automation..."
                    
                    # Get all descendants to see what's in the dialog
                    $allElements = $automation.FindAll([System.Windows.Automation.TreeScope]::Descendants, [System.Windows.Automation.Condition]::TrueCondition)
                    Write-Host "Found $($allElements.Count) UI elements in dialog"
                    
                    # Debug: List all elements found
                    if ($DebugMode) {
                        foreach ($element in $allElements) {
                            try {
                                $controlType = $element.Current.ControlType.ProgrammaticName
                                $name = $element.Current.Name
                                $className = $element.Current.ClassName
                                Write-Host "  Element: $controlType '$name' [$className]"
                            } catch {
                                Write-Host "  Element: (could not read properties)"
                            }
                        }
                    }
                    
                    # Look for buttons specifically
                    $buttonCondition = [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
                    $buttons = $automation.FindAll([System.Windows.Automation.TreeScope]::Descendants, $buttonCondition)
                    
                    Write-Host "Found $($buttons.Count) button(s) in dialog"
                    
                    $buttonClicked = $false
                    foreach ($button in $buttons) {
                        try {
                            $buttonName = $button.Current.Name
                            $buttonId = $button.Current.AutomationId
                            Write-Host "  Button: '$buttonName' [ID: $buttonId]"
                            
                            # Look for "No", "Cancel", "Skip", "Later", or empty buttons (often the second button is "No")
                            if ($buttonName -like "*No*" -or 
                                $buttonName -like "*Cancel*" -or 
                                $buttonName -like "*Skip*" -or 
                                $buttonName -like "*Later*" -or
                                $buttonName -like "*Remind*" -or
                                $buttonName -eq "" -or
                                $buttonId -eq "2" -or
                                $buttonId -eq "7") {  # Common IDs for "No" and "Cancel"
                                
                                Write-Host "Attempting to click '$buttonName' button (ID: $buttonId)..."
                                
                                # Try to get the invoke pattern
                                try {
                                    $invokePattern = $button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
                                    if ($invokePattern) {
                                        $invokePattern.Invoke()
                                        Write-Host "Successfully clicked button via InvokePattern"
                                        $buttonClicked = $true
                                        Start-Sleep -Seconds 2
                                        break
                                    }
                                } catch {
                                    Write-Host "InvokePattern failed: $($_.Exception.Message)"
                                }
                                
                                # Try clicking via bounding rectangle
                                try {
                                    $rect = $button.Current.BoundingRectangle
                                    $centerX = $rect.Left + ($rect.Width / 2)
                                    $centerY = $rect.Top + ($rect.Height / 2)
                                    
                                    Write-Host "Trying to click at coordinates ($centerX, $centerY)..."
                                    
                                    # Use Windows API to click
                                    Add-Type @"
                                        using System;
                                        using System.Runtime.InteropServices;
                                        public class MouseAPI {
                                            [DllImport("user32.dll")]
                                            public static extern bool SetCursorPos(int x, int y);
                                            [DllImport("user32.dll")]
                                            public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
                                            public const uint MOUSEEVENTF_LEFTDOWN = 0x02;
                                            public const uint MOUSEEVENTF_LEFTUP = 0x04;
                                        }
"@
                                    
                                    [MouseAPI]::SetCursorPos([int]$centerX, [int]$centerY)
                                    Start-Sleep -Milliseconds 100
                                    [MouseAPI]::mouse_event([MouseAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
                                    [MouseAPI]::mouse_event([MouseAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
                                    
                                    Write-Host "Clicked at coordinates"
                                    $buttonClicked = $true
                                    Start-Sleep -Seconds 2
                                    break
                                    
                                } catch {
                                    Write-Host "Coordinate click failed: $($_.Exception.Message)"
                                }
                            }
                        } catch {
                            Write-Host "Could not analyze button: $($_.Exception.Message)"
                        }
                    }
                    
                    if ($buttonClicked) {
                        return $true
                    }
                }
                
            } catch {
                Write-Host "UI Automation failed: $($_.Exception.Message)"
            }
            
            # Method 2: Try keyboard shortcuts
            Write-Host "Trying keyboard shortcuts to dismiss dialog..."
            
            # Focus the window first
            try {
                Add-Type @"
                    using System;
                    using System.Runtime.InteropServices;
                    public class WindowAPI2 {
                        [DllImport("user32.dll")]
                        public static extern bool SetForegroundWindow(IntPtr hWnd);
                        [DllImport("user32.dll")]
                        public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
                    }
"@
                [WindowAPI2]::SetForegroundWindow([IntPtr]$window.Handle)
                [WindowAPI2]::ShowWindow([IntPtr]$window.Handle, 1)
                Start-Sleep -Milliseconds 500
                
                # Try different key combinations
                Add-Type -AssemblyName System.Windows.Forms
                
                # Try "N" for No
                Write-Host "Sending 'N' key for No..."
                [System.Windows.Forms.SendKeys]::SendWait("N")
                Start-Sleep -Seconds 1
                
                # Try Escape
                Write-Host "Sending Escape key..."
                [System.Windows.Forms.SendKeys]::SendWait("{ESC}")
                Start-Sleep -Seconds 1
                
                # Try Alt+F4
                Write-Host "Sending Alt+F4..."
                [System.Windows.Forms.SendKeys]::SendWait("%{F4}")
                Start-Sleep -Seconds 1
                
                # Try Tab+Enter (assuming No is second button)
                Write-Host "Sending Tab+Enter..."
                [System.Windows.Forms.SendKeys]::SendWait("{TAB}{ENTER}")
                Start-Sleep -Seconds 1
                
                return $true
                
            } catch {
                Write-Host "Keyboard method failed: $($_.Exception.Message)"
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
    $updateDialogAttempts = 0
    $maxUpdateAttempts = 5
    
    while ((Get-Date) -lt $endTime) {
        $windows = Find-SimplySignWindows -TargetProcessId $ProcessId
        
        if ($windows.Count -eq 0) {
            Write-Host "No windows found, waiting..."
            Start-Sleep -Seconds 2
            continue
        }
        
        # Check if we have any dialogs that might be update prompts
        $updateHandled = Handle-UpdateDialog -Windows $windows
        
        if ($updateHandled) {
            $updateDialogAttempts++
            Write-Host "Update dialog handling attempt $updateDialogAttempts/$maxUpdateAttempts completed"
            
            # Wait a bit longer after handling update dialog
            Start-Sleep -Seconds 5
            
            # Re-check for windows after update dialog dismissal
            $updatedWindows = Find-SimplySignWindows -TargetProcessId $ProcessId
            
            if ($updatedWindows.Count -gt 0) {
                Write-Host "Checking if login dialog appeared after update dismissal..."
                
                foreach ($window in $updatedWindows) {
                    # Look for login dialog characteristics
                    if ($window.Title -like "*login*" -or 
                        $window.Title -like "*sign*" -or 
                        $window.Title -like "*auth*" -or
                        $window.Title -like "*connect*" -or
                        # If it's a different dialog (larger than typical update dialog)
                        ($window.Width -gt 400 -or $window.Height -gt 300)) {
                        
                        Write-Host "Potential login dialog found after update dismissal: '$($window.Title)'"
                        Write-Host "  Size: $($window.Width)x$($window.Height)"
                        return $updatedWindows
                    }
                }
            }
            
            # If we've tried multiple times and still see update dialogs, continue waiting
            if ($updateDialogAttempts -ge $maxUpdateAttempts) {
                Write-Host "Reached maximum update dialog attempts, proceeding with current windows..."
                return $windows
            }
            
            continue
        }
        
        # Look for potential login dialogs
        foreach ($window in $windows) {
            if ($window.Title -like "*login*" -or 
                $window.Title -like "*sign*" -or 
                $window.Title -like "*auth*" -or
                $window.Title -like "*connect*" -or
                # Consider larger dialogs as potential login windows
                ($window.Width -gt 400 -and $window.Height -gt 300 -and $window.Visible)) {
                
                Write-Host "Potential login dialog found: '$($window.Title)' (Size: $($window.Width)x$($window.Height))"
                return $windows
            }
        }
        
        # If we only have small dialogs, they might still be update dialogs
        $allSmallDialogs = $true
        foreach ($window in $windows) {
            if ($window.Visible -and ($window.Width -gt 400 -or $window.Height -gt 300)) {
                $allSmallDialogs = $false
                break
            }
        }
        
        if ($allSmallDialogs) {
            Write-Host "Only small dialogs found, likely still update dialogs. Continuing to wait..."
        } else {
            Write-Host "Found larger dialogs, assuming login dialog appeared"
            return $windows
        }
        
        Start-Sleep -Seconds 2
    }
    
    Write-Host "Timeout waiting for login dialog"
    return Find-SimplySignWindows -TargetProcessId $ProcessId
}

function Analyze-UIElements {
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

function Start-NetworkMonitoring {
    param([int]$TimeoutSeconds = 120)
    
    Write-Host "Starting network monitoring for OAuth2 activity..."
    
    # Create a background job to monitor network activity
    $monitoringJob = Start-Job -ScriptBlock {
        param($timeout)
        
        $endTime = (Get-Date).AddSeconds($timeout)
        $dnsCount = 0
        $oauth2Activity = @()
        
        while ((Get-Date) -lt $endTime) {
            try {
                # Monitor DNS resolution for OAuth2 endpoint
                $dnsResult = nslookup cloudsign.webnotarius.pl 2>$null
                if ($dnsResult -and $dnsResult -notlike "*can't find*") {
                    $dnsCount++
                    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                    $oauth2Activity += "[$timestamp] DNS resolution successful for OAuth2 endpoint"
                }
                
                # Check for network connections (if netstat is available)
                try {
                    $connections = netstat -an 2>$null | Select-String "cloudsign" -Quiet
                    if ($connections) {
                        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                        $oauth2Activity += "[$timestamp] OAuth2 connection detected"
                    }
                } catch {
                    # netstat might not be available, continue with DNS monitoring
                }
                
            } catch {
                # Continue monitoring even if individual checks fail
            }
            
            Start-Sleep -Seconds 2
        }
        
        # Return results
        return @{
            DNSCount = $dnsCount
            Activity = $oauth2Activity
            TotalTime = $timeout
        }
        
    } -ArgumentList $TimeoutSeconds
    
    Write-Host "Network monitoring started in background (Job ID: $($monitoringJob.Id))"
    return $monitoringJob
}

function Get-NetworkMonitoringResults {
    param([System.Management.Automation.Job]$MonitoringJob)
    
    if (-not $MonitoringJob) {
        Write-Host "No network monitoring job provided"
        return $null
    }
    
    Write-Host "Collecting network monitoring results..."
    
    # Wait a bit for the job to complete if it's still running
    if ($MonitoringJob.State -eq "Running") {
        Write-Host "Network monitoring still running, waiting up to 10 seconds..."
        Wait-Job $MonitoringJob -Timeout 10 | Out-Null
    }
    
    try {
        if ($MonitoringJob.State -eq "Completed") {
            $results = Receive-Job $MonitoringJob
            Remove-Job $MonitoringJob
            
            # Save results to file for workflow compatibility
            $networkResults = @{
                Timestamp = Get-Date
                DNSResolutions = $results.DNSCount
                OAuth2Activity = $results.Activity
                MonitoringDuration = $results.TotalTime
            }
            
            $networkResults | ConvertTo-Json -Depth 3 | Out-File -FilePath "network_monitor_results.log" -Encoding UTF8
            
            Write-Host "Network monitoring completed:"
            Write-Host "  DNS resolutions: $($results.DNSCount)"
            Write-Host "  OAuth2 activities: $($results.Activity.Count)"
            
            if ($results.DNSCount -gt 0 -or $results.Activity.Count -gt 0) {
                Write-Host "  OAuth2 network activity detected!"
                foreach ($activity in $results.Activity) {
                    Write-Host "    $activity"
                }
            } else {
                Write-Host "  No OAuth2 network activity detected"
            }
            
            return $results
            
        } else {
            Write-Host "Network monitoring job did not complete successfully (State: $($MonitoringJob.State))"
            Remove-Job $MonitoringJob -Force
            return $null
        }
        
    } catch {
        Write-Host "Error collecting network monitoring results: $($_.Exception.Message)"
        Remove-Job $MonitoringJob -Force
        return $null
    }
}

function Take-Screenshot {
    param([string]$OutputPath = "screenshots", [string]$Suffix = "")
    
    try {
        if (-not (Test-Path $OutputPath)) {
            New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
        }
        
        $screen = [System.Windows.Forms.Screen]::PrimaryScreen
        $bitmap = New-Object System.Drawing.Bitmap($screen.Bounds.Width, $screen.Bounds.Height)
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        
        $graphics.CopyFromScreen($screen.Bounds.X, $screen.Bounds.Y, 0, 0, $screen.Bounds.Size)
        
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $filename = if ($Suffix) { 
            "$OutputPath/screenshot_${timestamp}_${Suffix}.png" 
        } else { 
            "$OutputPath/screenshot_$timestamp.png" 
        }
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

# Start network monitoring first
$networkMonitoringJob = Start-NetworkMonitoring -TimeoutSeconds $TimeoutSeconds

$detectionResults = @{
    Timestamp = Get-Date
    ProcessId = $ProcessId
    Windows = @()
    UIElements = @()
    Screenshots = @()
    NetworkResults = $null
    Summary = @{
        WindowsFound = 0
        InputFieldsFound = 0
        ButtonsFound = 0
        WebControlsFound = 0
        OAuth2ActivityDetected = $false
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

# Take multiple screenshots to capture dialog progression
Write-Host ""
Write-Host "Taking screenshots to capture dialog progression..."

# First screenshot - immediate state (likely shows update dialog)
$screenshot1 = Take-Screenshot -Suffix "initial_state"
if ($screenshot1) {
    $detectionResults.Screenshots += $screenshot1
}

# Try to dismiss any update dialogs one more time before waiting
Write-Host "Making additional attempt to dismiss update dialogs..."
$currentWindows = Find-SimplySignWindows -TargetProcessId $ProcessId
$updateDismissed = Handle-UpdateDialog -Windows $currentWindows

if ($updateDismissed) {
    Write-Host "Additional update dialog dismissal attempted"
    Start-Sleep -Seconds 3
    
    # Screenshot after update dismissal
    $screenshot2 = Take-Screenshot -Suffix "after_update_dismissal"
    if ($screenshot2) {
        $detectionResults.Screenshots += $screenshot2
    }
}

# Wait longer for login dialog to fully appear
Write-Host "Waiting 8 seconds for login dialog to fully appear..."
Start-Sleep -Seconds 8

# Screenshot after longer wait
$screenshot3 = Take-Screenshot -Suffix "after_extended_wait"
if ($screenshot3) {
    $detectionResults.Screenshots += $screenshot3
}

# Check for any new dialogs that appeared
Write-Host "Checking for new dialogs after extended wait..."
$finalWindows = Find-SimplySignWindows -TargetProcessId $ProcessId

if ($finalWindows.Count -ne $windows.Count) {
    Write-Host "Window count changed from $($windows.Count) to $($finalWindows.Count) - analyzing new state..."
    
    # Screenshot showing the final state
    $screenshot4 = Take-Screenshot -Suffix "final_window_state"
    if ($screenshot4) {
        $detectionResults.Screenshots += $screenshot4
    }
}

# Check if we have new windows after the extended wait
Write-Host "Re-checking for new login dialog windows..."
$updatedWindows = Find-SimplySignWindows -TargetProcessId $ProcessId

if ($updatedWindows.Count -gt $windows.Count) {
    Write-Host "New windows detected after extended wait - analyzing updated windows..."
    
    # Analyze any new windows that appeared
    foreach ($window in $updatedWindows) {
        $existingWindow = $windows | Where-Object { $_.Handle -eq $window.Handle }
        if (-not $existingWindow) {
            Write-Host "Analyzing new window: '$($window.Title)' [$($window.ClassName)]"
            Write-Host "  Size: $($window.Width)x$($window.Height) Position: ($($window.Left),$($window.Top))"
            
            $newUIElements = Analyze-UIElements -WindowHandle ([IntPtr]$window.Handle)
            $detectionResults.UIElements += $newUIElements
            
            # Update counts
            $newInputFields = $newUIElements | Where-Object { $_.ElementType -eq "InputField" }
            $newButtons = $newUIElements | Where-Object { $_.ElementType -eq "Button" }
            $newWebControls = $newUIElements | Where-Object { $_.ElementType -eq "WebControl" }
            
            $detectionResults.Summary.InputFieldsFound += $newInputFields.Count
            $detectionResults.Summary.ButtonsFound += $newButtons.Count
            $detectionResults.Summary.WebControlsFound += $newWebControls.Count
            
            Write-Host "  New UI Elements found:"
            Write-Host "    Input fields: $($newInputFields.Count)"
            Write-Host "    Buttons: $($newButtons.Count)"
            Write-Host "    Web controls: $($newWebControls.Count)"
        }
    }
    
    # Take final screenshot after analyzing new windows
    $screenshotFinal = Take-Screenshot -Suffix "final_analysis_complete"
    if ($screenshotFinal) {
        $detectionResults.Screenshots += $screenshotFinal
    }
    
    # Update windows list
    $detectionResults.Windows = $updatedWindows
    $detectionResults.Summary.WindowsFound = $updatedWindows.Count
    
} elseif ($finalWindows.Count -ne $windows.Count) {
    # Update to the final windows state if it changed
    $detectionResults.Windows = $finalWindows
    $detectionResults.Summary.WindowsFound = $finalWindows.Count
    
    Write-Host "Window state updated to final count: $($finalWindows.Count)"
}

# Collect network monitoring results
Write-Host ""
$networkResults = Get-NetworkMonitoringResults -MonitoringJob $networkMonitoringJob
if ($networkResults) {
    $detectionResults.NetworkResults = $networkResults
    $detectionResults.Summary.OAuth2ActivityDetected = ($networkResults.DNSCount -gt 0 -or $networkResults.Activity.Count -gt 0)
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
Write-Host "OAuth2 activity detected: $($detectionResults.Summary.OAuth2ActivityDetected)"

if ($detectionResults.NetworkResults) {
    Write-Host "Network monitoring results:"
    Write-Host "  DNS resolutions: $($detectionResults.NetworkResults.DNSCount)"
    Write-Host "  Network activities: $($detectionResults.NetworkResults.Activity.Count)"
}

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

if ($detectionResults.Summary.OAuth2ActivityDetected) {
    Write-Host ""
    Write-Host "NETWORK ACTIVITY CONFIRMED: OAuth2 communication detected during session"
} else {
    Write-Host ""
    Write-Host "NO OAUTH2 ACTIVITY: Login dialog may not have appeared or no cloud connection"
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
