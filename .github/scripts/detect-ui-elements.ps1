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
    [switch]$DebugMode = $true  # Enable debug mode by default
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

# Windows API declarations for advanced window detection and system tray interaction
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
    
    // System tray and taskbar APIs
    [DllImport("user32.dll")]
    public static extern IntPtr FindWindow(string lpClassName, string lpWindowName);
    
    [DllImport("user32.dll")]
    public static extern IntPtr FindWindowEx(IntPtr hwndParent, IntPtr hwndChildAfter, string lpszClass, string lpszWindow);
    
    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int x, int y);
    
    [DllImport("user32.dll")]
    public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
    
    [DllImport("user32.dll")]
    public static extern bool GetCursorPos(out POINT lpPoint);
    
    [DllImport("shell32.dll")]
    public static extern uint SHAppBarMessage(uint dwMessage, ref APPBARDATA pData);
    
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
    
    public const uint MOUSEEVENTF_LEFTDOWN = 0x02;
    public const uint MOUSEEVENTF_LEFTUP = 0x04;
    public const uint MOUSEEVENTF_RIGHTDOWN = 0x08;
    public const uint MOUSEEVENTF_RIGHTUP = 0x10;
    public const uint ABM_GETTASKBARPOS = 0x00000005;
    
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT {
        public int Left;
        public int Top;
        public int Right;
        public int Bottom;
    }
    
    [StructLayout(LayoutKind.Sequential)]
    public struct POINT {
        public int X;
        public int Y;
    }
    
    [StructLayout(LayoutKind.Sequential)]
    public struct APPBARDATA {
        public uint cbSize;
        public IntPtr hWnd;
        public uint uCallbackMessage;
        public uint uEdge;
        public RECT rc;
        public IntPtr lParam;
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

function Find-SystemTrayApplications {
    param([int]$TargetProcessId = 0)
    
    Write-Host "Searching for system tray applications..."
    
    $trayInfo = @{
        TaskbarFound = $false
        NotificationAreaFound = $false
        TrayIcons = @()
        TaskbarRect = @{}
        NotificationAreaRect = @{}
    }
    
    try {
        # Find the taskbar window
        $taskbarHandle = [WindowAPI]::FindWindow("Shell_TrayWnd", $null)
        if ($taskbarHandle -ne [IntPtr]::Zero) {
            $trayInfo.TaskbarFound = $true
            Write-Host "Found taskbar window: Handle $($taskbarHandle.ToInt64())"
            
            # Get taskbar position using SHAppBarMessage
            $appBarData = New-Object WindowAPI+APPBARDATA
            $appBarData.cbSize = [System.Runtime.InteropServices.Marshal]::SizeOf($appBarData)
            
            $result = [WindowAPI]::SHAppBarMessage([WindowAPI]::ABM_GETTASKBARPOS, [ref]$appBarData)
            if ($result -ne 0) {
                $trayInfo.TaskbarRect = @{
                    Left = $appBarData.rc.Left
                    Top = $appBarData.rc.Top
                    Right = $appBarData.rc.Right
                    Bottom = $appBarData.rc.Bottom
                    Width = $appBarData.rc.Right - $appBarData.rc.Left
                    Height = $appBarData.rc.Bottom - $appBarData.rc.Top
                }
                Write-Host "Taskbar position: ($($trayInfo.TaskbarRect.Left),$($trayInfo.TaskbarRect.Top)) Size: $($trayInfo.TaskbarRect.Width)x$($trayInfo.TaskbarRect.Height)"
            }
            
            # Find the notification area (system tray)
            $trayNotifyHandle = [WindowAPI]::FindWindowEx($taskbarHandle, [IntPtr]::Zero, "TrayNotifyWnd", $null)
            if ($trayNotifyHandle -ne [IntPtr]::Zero) {
                $trayInfo.NotificationAreaFound = $true
                Write-Host "Found notification area: Handle $($trayNotifyHandle.ToInt64())"
                
                # Get notification area rectangle
                $notifyRect = New-Object WindowAPI+RECT
                [WindowAPI]::GetWindowRect($trayNotifyHandle, [ref]$notifyRect) | Out-Null
                
                $trayInfo.NotificationAreaRect = @{
                    Left = $notifyRect.Left
                    Top = $notifyRect.Top
                    Right = $notifyRect.Right
                    Bottom = $notifyRect.Bottom
                    Width = $notifyRect.Right - $notifyRect.Left
                    Height = $notifyRect.Bottom - $notifyRect.Top
                }
                Write-Host "Notification area position: ($($trayInfo.NotificationAreaRect.Left),$($trayInfo.NotificationAreaRect.Top)) Size: $($trayInfo.NotificationAreaRect.Width)x$($trayInfo.NotificationAreaRect.Height)"
                
                # Find the SysPager (contains the actual tray icons)
                $sysPagerHandle = [WindowAPI]::FindWindowEx($trayNotifyHandle, [IntPtr]::Zero, "SysPager", $null)
                if ($sysPagerHandle -ne [IntPtr]::Zero) {
                    Write-Host "Found SysPager: Handle $($sysPagerHandle.ToInt64())"
                    
                    # Find ToolbarWindow32 (contains individual tray icons)
                    $toolbarHandle = [WindowAPI]::FindWindowEx($sysPagerHandle, [IntPtr]::Zero, "ToolbarWindow32", $null)
                    if ($toolbarHandle -ne [IntPtr]::Zero) {
                        Write-Host "Found tray toolbar: Handle $($toolbarHandle.ToInt64())"
                        
                        # Try to enumerate tray icon information
                        try {
                            $automation = [System.Windows.Automation.AutomationElement]::FromHandle($toolbarHandle)
                            if ($automation) {
                                $buttonCondition = [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
                                $trayButtons = $automation.FindAll([System.Windows.Automation.TreeScope]::Children, $buttonCondition)
                                
                                Write-Host "Found $($trayButtons.Count) tray icon(s)"
                                
                                foreach ($button in $trayButtons) {
                                    try {
                                        $buttonInfo = @{
                                            Name = $button.Current.Name
                                            AutomationId = $button.Current.AutomationId
                                            ClassName = $button.Current.ClassName
                                            BoundingRectangle = $button.Current.BoundingRectangle
                                            IsEnabled = $button.Current.IsEnabled
                                        }
                                        
                                        # Check if this might be SimplySign based on name patterns
                                        $isSimplySign = ($buttonInfo.Name -like "*SimplySign*" -or 
                                                        $buttonInfo.Name -like "*Sign*" -or 
                                                        $buttonInfo.AutomationId -like "*SimplySign*")
                                        
                                        $buttonInfo.IsSimplySign = $isSimplySign
                                        $trayInfo.TrayIcons += $buttonInfo
                                        
                                        if ($isSimplySign) {
                                            Write-Host "POTENTIAL SIMPLYSIGN TRAY ICON FOUND: '$($buttonInfo.Name)'"
                                            Write-Host "  Position: ($($buttonInfo.BoundingRectangle.Left),$($buttonInfo.BoundingRectangle.Top))"
                                            Write-Host "  Size: $($buttonInfo.BoundingRectangle.Width)x$($buttonInfo.BoundingRectangle.Height)"
                                        } else {
                                            Write-Host "Tray icon: '$($buttonInfo.Name)' [ID: $($buttonInfo.AutomationId)]"
                                        }
                                        
                                    } catch {
                                        Write-Host "Could not analyze tray button: $($_.Exception.Message)"
                                    }
                                }
                            }
                        } catch {
                            Write-Host "Could not enumerate tray icons via UI Automation: $($_.Exception.Message)"
                        }
                    }
                }
            }
        }
        
    } catch {
        Write-Host "Error detecting system tray: $($_.Exception.Message)"
    }
    
    return $trayInfo
}

function Interact-WithSystemTray {
    param([hashtable]$TrayInfo, [string]$OutputPath = "screenshots")
    
    Write-Host "Attempting to interact with system tray..."
    
    if (-not $TrayInfo.NotificationAreaFound) {
        Write-Host "No notification area found - cannot interact with system tray"
        return $false
    }
    
    $interactions = @()
    
    try {
        # Take screenshot before interaction
        $beforeScreenshot = Take-Screenshot -OutputPath $OutputPath -Suffix "before_tray_interaction"
        if ($beforeScreenshot) {
            $interactions += "Screenshot taken before tray interaction: $beforeScreenshot"
        }
        
        # Check if we found any potential SimplySign tray icons
        $simplySignIcons = $TrayInfo.TrayIcons | Where-Object { $_.IsSimplySign -eq $true }
        
        if ($simplySignIcons.Count -gt 0) {
            Write-Host "Found $($simplySignIcons.Count) potential SimplySign tray icon(s)"
            
            foreach ($icon in $simplySignIcons) {
                Write-Host "Attempting to click SimplySign tray icon: '$($icon.Name)'"
                
                # Calculate center of the tray icon
                $centerX = $icon.BoundingRectangle.Left + ($icon.BoundingRectangle.Width / 2)
                $centerY = $icon.BoundingRectangle.Top + ($icon.BoundingRectangle.Height / 2)
                
                # Left click on the tray icon
                Write-Host "Left-clicking tray icon at ($centerX, $centerY)"
                [WindowAPI]::SetCursorPos([int]$centerX, [int]$centerY)
                Start-Sleep -Milliseconds 200
                [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
                Start-Sleep -Milliseconds 50
                [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
                
                $interactions += "Left-clicked SimplySign tray icon '$($icon.Name)' at ($centerX, $centerY)"
                
                # Wait for potential dialog to appear
                Start-Sleep -Seconds 2
                
                # Take screenshot after left click
                $leftClickScreenshot = Take-Screenshot -OutputPath $OutputPath -Suffix "after_tray_left_click"
                if ($leftClickScreenshot) {
                    $interactions += "Screenshot after left click: $leftClickScreenshot"
                }
                
                # Also try right-click to see context menu
                Write-Host "Right-clicking tray icon for context menu"
                [WindowAPI]::SetCursorPos([int]$centerX, [int]$centerY)
                Start-Sleep -Milliseconds 200
                [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, [UIntPtr]::Zero)
                Start-Sleep -Milliseconds 50
                [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_RIGHTUP, 0, 0, 0, [UIntPtr]::Zero)
                
                $interactions += "Right-clicked SimplySign tray icon '$($icon.Name)' for context menu"
                
                # Wait for context menu
                Start-Sleep -Seconds 2
                
                # Take screenshot after right click
                $rightClickScreenshot = Take-Screenshot -OutputPath $OutputPath -Suffix "after_tray_right_click"
                if ($rightClickScreenshot) {
                    $interactions += "Screenshot after right click: $rightClickScreenshot"
                }
                
                # Wait a bit more and dismiss any context menu by clicking elsewhere
                Start-Sleep -Seconds 1
                [WindowAPI]::SetCursorPos(100, 100)  # Click away from tray
                [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
                [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
                
                Start-Sleep -Seconds 1
            }
            
        } else {
            Write-Host "No SimplySign tray icons found, attempting generic tray interaction"
            
            # Try clicking in the general notification area to reveal hidden icons
            $notifyRect = $TrayInfo.NotificationAreaRect
            $centerX = $notifyRect.Left + ($notifyRect.Width / 2)
            $centerY = $notifyRect.Top + ($notifyRect.Height / 2)
            
            Write-Host "Clicking notification area center at ($centerX, $centerY)"
            [WindowAPI]::SetCursorPos($centerX, $centerY)
            Start-Sleep -Milliseconds 200
            [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
            Start-Sleep -Milliseconds 50
            [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
            
            $interactions += "Clicked notification area center at ($centerX, $centerY)"
            
            Start-Sleep -Seconds 2
            
            # Take screenshot after generic click
            $genericClickScreenshot = Take-Screenshot -OutputPath $OutputPath -Suffix "after_generic_tray_click"
            if ($genericClickScreenshot) {
                $interactions += "Screenshot after generic tray click: $genericClickScreenshot"
            }
            
            # Try clicking the "Show hidden icons" button (usually a small arrow)
            $showHiddenX = $notifyRect.Left + 10  # Usually near the left edge
            $showHiddenY = $centerY
            
            Write-Host "Attempting to click 'Show hidden icons' at ($showHiddenX, $showHiddenY)"
            [WindowAPI]::SetCursorPos($showHiddenX, $showHiddenY)
            Start-Sleep -Milliseconds 200
            [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
            Start-Sleep -Milliseconds 50
            [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
            
            $interactions += "Attempted to click 'Show hidden icons' at ($showHiddenX, $showHiddenY)"
            
            Start-Sleep -Seconds 2
            
            # Take screenshot after trying to show hidden icons
            $hiddenIconsScreenshot = Take-Screenshot -OutputPath $OutputPath -Suffix "after_show_hidden_icons"
            if ($hiddenIconsScreenshot) {
                $interactions += "Screenshot after attempting to show hidden icons: $hiddenIconsScreenshot"
            }
        }
        
        # Take final screenshot
        Start-Sleep -Seconds 2
        $finalScreenshot = Take-Screenshot -OutputPath $OutputPath -Suffix "final_tray_interaction"
        if ($finalScreenshot) {
            $interactions += "Final screenshot after tray interaction: $finalScreenshot"
        }
        
    } catch {
        Write-Host "Error during tray interaction: $($_.Exception.Message)"
        $interactions += "Error during tray interaction: $($_.Exception.Message)"
    }
    
    # Save interaction log
    $interactionLog = @{
        Timestamp = Get-Date
        TrayInfo = $TrayInfo
        Interactions = $interactions
    }
    
    $interactionLog | ConvertTo-Json -Depth 5 | Out-File -FilePath "$OutputPath/tray_interaction_log.json" -Encoding UTF8
    Write-Host "Tray interaction log saved to: $OutputPath/tray_interaction_log.json"
    
    return $interactions.Count -gt 0
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
        Write-Host "Examining window: '$($window.Title)' [$($window.ClassName)] Size: $($window.Width)x$($window.Height)"
        Write-Host "  Handle: $($window.Handle) Visible: $($window.Visible)"
        
        # Target the EXACT update dialog pattern we know:
        # - Title: "SimplySign Desktop"
        # - Class: "#32770" 
        # - Size: ~328x166 (small dialog)
        # - Has "Yes" and "No" buttons
        $isUpdateDialog = $false
        
        if ($window.Title -eq "SimplySign Desktop" -and 
            $window.ClassName -eq "#32770" -and 
            $window.Width -gt 300 -and $window.Width -lt 400 -and
            $window.Height -gt 150 -and $window.Height -lt 200 -and
            $window.Visible) {
            
            $isUpdateDialog = $true
            Write-Host "  CONFIRMED: This matches the exact update dialog pattern!"
        }
        
        if ($isUpdateDialog) {
            Write-Host "  TARGETING UPDATE DIALOG: '$($window.Title)' [$($window.ClassName)]"
            Write-Host "  Position: ($($window.Left),$($window.Top)) Size: $($window.Width)x$($window.Height)"
            
            $dialogDismissed = $false
            $windowHandle = [IntPtr]$window.Handle
            
            try {
                # Method 0: WM_CLOSE message (most reliable for dialogs)
                Write-Host "Method 0: Sending WM_CLOSE message to dialog..."
                Add-Type @"
using System;
using System.Runtime.InteropServices;
public class CloseDialogAPI {
    [DllImport("user32.dll")]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")]
    public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool BringWindowToTop(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool SetActiveWindow(IntPtr hWnd);
    [DllImport("user32.dll")]
    public static extern bool IsWindow(IntPtr hWnd);
    public const uint WM_CLOSE = 0x0010;
    public const uint WM_DESTROY = 0x0002;
    public const uint WM_SYSCOMMAND = 0x0112;
    public const uint SC_CLOSE = 0xF060;
}
"@
                # First activate the dialog
                Write-Host "  Activating dialog at handle $($window.Handle)..."
                $activated = [CloseDialogAPI]::BringWindowToTop($windowHandle)
                Write-Host "    BringWindowToTop result: $activated"
                
                $foreground = [CloseDialogAPI]::SetForegroundWindow($windowHandle)
                Write-Host "    SetForegroundWindow result: $foreground"
                
                $active = [CloseDialogAPI]::SetActiveWindow($windowHandle)
                Write-Host "    SetActiveWindow result: $active"
                
                Start-Sleep -Milliseconds 500
                
                # Check if window is still valid
                $isValid = [CloseDialogAPI]::IsWindow($windowHandle)
                Write-Host "    Window still valid: $isValid"
                
                if ($isValid) {
                    # Try multiple close methods
                    Write-Host "  Sending WM_CLOSE..."
                    $closeResult = [CloseDialogAPI]::SendMessage($windowHandle, [CloseDialogAPI]::WM_CLOSE, [IntPtr]::Zero, [IntPtr]::Zero)
                    Write-Host "    WM_CLOSE result: $closeResult"
                    
                    Start-Sleep -Milliseconds 500
                    
                    # Check if still valid after WM_CLOSE
                    $stillValid = [CloseDialogAPI]::IsWindow($windowHandle)
                    Write-Host "    Window still valid after WM_CLOSE: $stillValid"
                    
                    if ($stillValid) {
                        Write-Host "  Sending WM_SYSCOMMAND SC_CLOSE..."
                        $sysclose = [CloseDialogAPI]::SendMessage($windowHandle, [CloseDialogAPI]::WM_SYSCOMMAND, [IntPtr][CloseDialogAPI]::SC_CLOSE, [IntPtr]::Zero)
                        Write-Host "    WM_SYSCOMMAND result: $sysclose"
                        
                        Start-Sleep -Milliseconds 500
                        
                        $finalValid = [CloseDialogAPI]::IsWindow($windowHandle)
                        Write-Host "    Window still valid after SC_CLOSE: $finalValid"
                        
                        if (-not $finalValid) {
                            Write-Host "  SUCCESS: Dialog dismissed via WM_SYSCOMMAND"
                            $dialogDismissed = $true
                        }
                    } else {
                        Write-Host "  SUCCESS: Dialog dismissed via WM_CLOSE"
                        $dialogDismissed = $true
                    }
                } else {
                    Write-Host "  Window handle became invalid during activation"
                }
                
                Start-Sleep -Seconds 2
                
            } catch {
                Write-Host "  WM_CLOSE method failed: $($_.Exception.Message)"
            }
            
            if (-not $dialogDismissed) {
                # Method 1: Click the "X" close button (calculated coordinates)
                Write-Host "Method 1: Clicking 'X' close button at calculated coordinates..."
                
                try {
                    # Calculate close button position (typically 20px from right edge, 10px from top)
                    $closeButtonX = $window.Right - 20
                    $closeButtonY = $window.Top + 10
                    
                    Write-Host "  Calculated close button position: ($closeButtonX, $closeButtonY)"
                    Write-Host "  Dialog bounds: Left=$($window.Left) Top=$($window.Top) Right=$($window.Right) Bottom=$($window.Bottom)"
                    
                    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class CloseButtonAPI {
    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")]
    public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
    [DllImport("user32.dll")]
    public static extern bool GetCursorPos(out POINT lpPoint);
    public const uint MOUSEEVENTF_LEFTDOWN = 0x02;
    public const uint MOUSEEVENTF_LEFTUP = 0x04;
    
    [StructLayout(LayoutKind.Sequential)]
    public struct POINT {
        public int X;
        public int Y;
    }
}
"@
                    
                    # Move cursor and verify position
                    [CloseButtonAPI]::SetCursorPos($closeButtonX, $closeButtonY)
                    Start-Sleep -Milliseconds 200
                    
                    # Verify cursor position
                    $cursorPos = New-Object CloseButtonAPI+POINT
                    [CloseButtonAPI]::GetCursorPos([ref]$cursorPos)
                    Write-Host "  Cursor moved to: ($($cursorPos.X), $($cursorPos.Y))"
                    
                    # Perform click
                    [CloseButtonAPI]::mouse_event([CloseButtonAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
                    Start-Sleep -Milliseconds 50
                    [CloseButtonAPI]::mouse_event([CloseButtonAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
                    
                    Write-Host "  Performed click at close button coordinates"
                    $dialogDismissed = $true
                    Start-Sleep -Seconds 2
                    
                } catch {
                    Write-Host "  Close button click failed: $($_.Exception.Message)"
                }
            }
            
            if (-not $dialogDismissed) {
                # Method 2: Click the "No" button (calculated coordinates)
                Write-Host "Method 2: Clicking 'No' button at calculated coordinates..."
                
                try {
                    # Calculate "No" button position (same distance from corner as close button)
                    $noButtonX = $window.Right - 20
                    $noButtonY = $window.Bottom - 20
                    
                    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class NoButtonCoordAPI {
    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")]
    public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
    public const uint MOUSEEVENTF_LEFTDOWN = 0x02;
    public const uint MOUSEEVENTF_LEFTUP = 0x04;
}
"@
                    
                    [NoButtonCoordAPI]::SetCursorPos($noButtonX, $noButtonY)
                    Start-Sleep -Milliseconds 100
                    [NoButtonCoordAPI]::mouse_event([NoButtonCoordAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
                    [NoButtonCoordAPI]::mouse_event([NoButtonCoordAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
                    
                    Write-Host "  Clicked 'No' button at coordinates ($noButtonX, $noButtonY)"
                    $dialogDismissed = $true
                    Start-Sleep -Seconds 2
                    
                } catch {
                    Write-Host "  No button coordinate click failed: $($_.Exception.Message)"
                }
            }
            
            if (-not $dialogDismissed) {
                # Method 3: Direct "No" button targeting via UI Automation
                try {
                    Write-Host "Method 3: Searching for 'No' button via UI Automation..."
                    $automation = [System.Windows.Automation.AutomationElement]::FromHandle($windowHandle)
                    
                    if ($automation) {
                        # Find all buttons
                        $buttonCondition = [System.Windows.Automation.PropertyCondition]::new([System.Windows.Automation.AutomationElement]::ControlTypeProperty, [System.Windows.Automation.ControlType]::Button)
                        $buttons = $automation.FindAll([System.Windows.Automation.TreeScope]::Descendants, $buttonCondition)
                        
                        Write-Host "  Found $($buttons.Count) button(s) in update dialog"
                        
                        foreach ($button in $buttons) {
                            try {
                                $buttonName = $button.Current.Name
                                $buttonId = $button.Current.AutomationId
                                Write-Host "    Button found: '$buttonName' [ID: $buttonId]"
                                
                                # Look specifically for "No" button
                                if ($buttonName -eq "No" -or $buttonName -eq "&No" -or $buttonId -eq "7") {
                                    Write-Host "    Found 'No' button - clicking it!"
                                    
                                    # Try InvokePattern
                                    try {
                                        $invokePattern = $button.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern)
                                        if ($invokePattern) {
                                            $invokePattern.Invoke()
                                            Write-Host "    Successfully clicked 'No' button via InvokePattern"
                                            $dialogDismissed = $true
                                            Start-Sleep -Seconds 2
                                            break
                                        }
                                    } catch {
                                        Write-Host "    InvokePattern failed: $($_.Exception.Message)"
                                    }
                                    
                                    # Try coordinate click as backup
                                    try {
                                        $rect = $button.Current.BoundingRectangle
                                        $centerX = $rect.Left + ($rect.Width / 2)
                                        $centerY = $rect.Top + ($rect.Height / 2)
                                        
                                        Add-Type @"
using System;
using System.Runtime.InteropServices;
public class NoButtonClickAPI {
    [DllImport("user32.dll")]
    public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")]
    public static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, UIntPtr dwExtraInfo);
    public const uint MOUSEEVENTF_LEFTDOWN = 0x02;
    public const uint MOUSEEVENTF_LEFTUP = 0x04;
}
"@
                                        
                                        [NoButtonClickAPI]::SetCursorPos([int]$centerX, [int]$centerY)
                                        Start-Sleep -Milliseconds 100
                                        [NoButtonClickAPI]::mouse_event([NoButtonClickAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
                                        [NoButtonClickAPI]::mouse_event([NoButtonClickAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
                                        
                                        Write-Host "    Clicked 'No' button at coordinates ($centerX, $centerY)"
                                        $dialogDismissed = $true
                                        Start-Sleep -Seconds 2
                                        break
                                        
                                    } catch {
                                        Write-Host "    Coordinate click failed: $($_.Exception.Message)"
                                    }
                                }
                            } catch {
                                Write-Host "    Could not analyze button: $($_.Exception.Message)"
                            }
                        }
                    }
                    
                } catch {
                    Write-Host "  UI Automation failed: $($_.Exception.Message)"
                }
            }
            
            if (-not $dialogDismissed) {
                # Method 4: Keyboard shortcut for "No" button
                Write-Host "Method 4: Using keyboard to select 'No' button..."
                
                try {
                    Add-Type -AssemblyName System.Windows.Forms
                    
                    # Since "Yes" is default and focused, Tab once to get to "No", then press Enter
                    Write-Host "  Pressing Tab to move to 'No' button..."
                    [System.Windows.Forms.SendKeys]::SendWait("{TAB}")
                    Start-Sleep -Milliseconds 300
                    
                    Write-Host "  Pressing Enter to click 'No' button..."
                    [System.Windows.Forms.SendKeys]::SendWait("{ENTER}")
                    Start-Sleep -Seconds 1
                    
                    Write-Host "    Tab+Enter sequence sent to select 'No'"
                    $dialogDismissed = $true
                    
                } catch {
                    Write-Host "  Keyboard method failed: $($_.Exception.Message)"
                }
            }
            
            if (-not $dialogDismissed) {
                # Method 5: Direct Windows message to simulate "No" button (ID 7)
                Write-Host "Method 5: Sending WM_COMMAND for 'No' button (ID 7)..."
                
                try {
                    Add-Type @"
using System;
using System.Runtime.InteropServices;
public class NoButtonMessageAPI {
    [DllImport("user32.dll")]
    public static extern IntPtr SendMessage(IntPtr hWnd, uint Msg, IntPtr wParam, IntPtr lParam);
    public const uint WM_COMMAND = 0x0111;
    public const int IDNO = 7;
}
"@
                    
                    # Send WM_COMMAND with IDNO (7) to simulate clicking "No" button
                    [NoButtonMessageAPI]::SendMessage($windowHandle, [NoButtonMessageAPI]::WM_COMMAND, [IntPtr][NoButtonMessageAPI]::IDNO, [IntPtr]::Zero)
                    Write-Host "  Sent WM_COMMAND for 'No' button"
                    $dialogDismissed = $true
                    Start-Sleep -Seconds 2
                    
                } catch {
                    Write-Host "  Windows message method failed: $($_.Exception.Message)"
                }
            }
            
            if (-not $dialogDismissed) {
                # Method 6: Try "N" key (mnemonic for "No")
                Write-Host "Method 6: Trying 'N' key (mnemonic for No)..."
                
                try {
                    Add-Type -AssemblyName System.Windows.Forms
                    [System.Windows.Forms.SendKeys]::SendWait("N")
                    Write-Host "  Sent 'N' key"
                    $dialogDismissed = $true
                    Start-Sleep -Seconds 1
                    
                } catch {
                    Write-Host "  'N' key method failed: $($_.Exception.Message)"
                }
            }
            
            if ($dialogDismissed) {
                Write-Host "UPDATE DIALOG DISMISSAL ATTEMPTED!"
                Write-Host "  Waiting 3 seconds for dialog to close..."
                Start-Sleep -Seconds 3
                
                # Verify the dialog was actually dismissed
                try {
                    $checkWindow = Get-WindowInfo -WindowHandle $windowHandle
                    if (-not $checkWindow.Visible) {
                        Write-Host "  CONFIRMED: Update dialog successfully dismissed!"
                        return $true
                    } else {
                        Write-Host "  Dialog still visible, but dismissal was attempted"
                        return $true  # Still return true to avoid infinite retry
                    }
                } catch {
                    Write-Host "  Dialog handle no longer valid - likely dismissed!"
                    return $true
                }
            } else {
                Write-Host "  All dismissal methods failed for this dialog"
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
    SystemTrayInfo = $null
    TrayInteractionResults = @()
    NetworkResults = $null
    Summary = @{
        WindowsFound = 0
        InputFieldsFound = 0
        ButtonsFound = 0
        WebControlsFound = 0
        OAuth2ActivityDetected = $false
        SystemTrayDetected = $false
        SimplySignTrayIconsFound = 0
    }
}

# FIRST: Check for system tray applications (this might be where SimplySign is hiding)
Write-Host ""
Write-Host "=== SYSTEM TRAY DETECTION ==="
$systemTrayInfo = Find-SystemTrayApplications -TargetProcessId $ProcessId
$detectionResults.SystemTrayInfo = $systemTrayInfo
$detectionResults.Summary.SystemTrayDetected = $systemTrayInfo.NotificationAreaFound

if ($systemTrayInfo.NotificationAreaFound) {
    $simplySignTrayIcons = $systemTrayInfo.TrayIcons | Where-Object { $_.IsSimplySign -eq $true }
    $detectionResults.Summary.SimplySignTrayIconsFound = $simplySignTrayIcons.Count
    
    Write-Host "System tray detection completed:"
    Write-Host "  Taskbar found: $($systemTrayInfo.TaskbarFound)"
    Write-Host "  Notification area found: $($systemTrayInfo.NotificationAreaFound)"
    Write-Host "  Total tray icons: $($systemTrayInfo.TrayIcons.Count)"
    Write-Host "  SimplySign tray icons: $($simplySignTrayIcons.Count)"
    
    if ($simplySignTrayIcons.Count -gt 0) {
        Write-Host ""
        Write-Host "THEORY CONFIRMED: SimplySign appears to be running in system tray!"
        Write-Host "This explains why we see network activity but no visible windows."
        
        # Interact with the system tray to try to trigger the login dialog
        Write-Host ""
        Write-Host "=== SYSTEM TRAY INTERACTION ==="
        $trayInteracted = Interact-WithSystemTray -TrayInfo $systemTrayInfo -OutputPath "screenshots"
        
        if ($trayInteracted) {
            Write-Host "System tray interaction completed - checking for new windows..."
            
            # Wait for potential dialogs to appear after tray interaction
            Start-Sleep -Seconds 5
            
            # Re-scan for windows after tray interaction
            Write-Host "Re-scanning for windows after tray interaction..."
            $postTrayWindows = Find-SimplySignWindows -TargetProcessId $ProcessId
            
            if ($postTrayWindows.Count -gt 0) {
                Write-Host "SUCCESS: Found $($postTrayWindows.Count) window(s) after tray interaction!"
                $detectionResults.Windows = $postTrayWindows
                $detectionResults.Summary.WindowsFound = $postTrayWindows.Count
                
                # Analyze the new windows
                foreach ($window in $postTrayWindows) {
                    Write-Host ""
                    Write-Host "=== ANALYZING POST-TRAY WINDOW: $($window.Title) ==="
                    Write-Host "Class: $($window.ClassName)"
                    Write-Host "Size: $($window.Width)x$($window.Height)"
                    Write-Host "Position: ($($window.Left),$($window.Top))"
                    Write-Host "Visible: $($window.Visible)"
                    
                    # Analyze UI elements in the new window
                    $uiElements = Analyze-UIElements -WindowHandle ([IntPtr]$window.Handle)
                    $detectionResults.UIElements += $uiElements
                    
                    # Count element types
                    $inputFields = $uiElements | Where-Object { $_.ElementType -eq "InputField" }
                    $buttons = $uiElements | Where-Object { $_.ElementType -eq "Button" }
                    $webControls = $uiElements | Where-Object { $_.ElementType -eq "WebControl" }
                    
                    $detectionResults.Summary.InputFieldsFound += $inputFields.Count
                    $detectionResults.Summary.ButtonsFound += $buttons.Count
                    $detectionResults.Summary.WebControlsFound += $webControls.Count
                    
                    Write-Host "UI Elements found in post-tray window:"
                    Write-Host "  Input fields: $($inputFields.Count)"
                    Write-Host "  Buttons: $($buttons.Count)"
                    Write-Host "  Web controls: $($webControls.Count)"
                }
                
            } else {
                Write-Host "No new windows appeared after tray interaction"
                Write-Host "SimplySign may require different interaction or may be using web-based login"
            }
        }
        
    } else {
        Write-Host "No SimplySign-specific tray icons detected"
        Write-Host "SimplySign may be using a generic icon or different identification"
    }
    
} else {
    Write-Host "Could not detect system tray - falling back to traditional window detection"
}

# SECOND: Traditional window detection (fallback or additional detection)
Write-Host ""
Write-Host "=== TRADITIONAL WINDOW DETECTION ==="
Write-Host "Searching for SimplySign windows..."
$traditionalWindows = Wait-ForLoginDialog -ProcessId $ProcessId -MaxWaitSeconds $TimeoutSeconds

if ($traditionalWindows.Count -eq 0 -and $ProcessId -ne 0) {
    Write-Host "No SimplySign windows found after waiting"
    Write-Host "Trying to find any windows for process $ProcessId..."
    $traditionalWindows = Find-SimplySignWindows -TargetProcessId $ProcessId
}

# Combine results if we haven't found windows via tray interaction
if ($detectionResults.Summary.WindowsFound -eq 0 -and $traditionalWindows.Count -gt 0) {
    $detectionResults.Windows = $traditionalWindows
    $detectionResults.Summary.WindowsFound = $traditionalWindows.Count
    
    Write-Host "Found $($traditionalWindows.Count) SimplySign window(s) via traditional detection"
    
    # Analyze each traditionally found window
    foreach ($window in $traditionalWindows) {
        Write-Host ""
        Write-Host "=== ANALYZING TRADITIONAL WINDOW: $($window.Title) ==="
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
} elseif ($detectionResults.Summary.WindowsFound -gt 0) {
    Write-Host "Using windows found via tray interaction (skipping traditional detection)"
} else {
    Write-Host "No windows found via either tray interaction or traditional detection"
}

# Take additional screenshots to capture final state
Write-Host ""
Write-Host "Taking final screenshots to capture complete session state..."

# Screenshot showing current desktop state
$finalDesktopScreenshot = Take-Screenshot -Suffix "final_desktop_state"
if ($finalDesktopScreenshot) {
    $detectionResults.Screenshots += $finalDesktopScreenshot
}

# Try one more interaction with system tray if we found it but no windows
if ($detectionResults.Summary.SystemTrayDetected -and $detectionResults.Summary.WindowsFound -eq 0) {
    Write-Host "Attempting additional system tray exploration..."
    
    # Take screenshot of notification area specifically
    if ($systemTrayInfo.NotificationAreaFound) {
        $notifyRect = $systemTrayInfo.NotificationAreaRect
        
        # Double-click in notification area to explore further
        $centerX = $notifyRect.Left + ($notifyRect.Width / 2)
        $centerY = $notifyRect.Top + ($notifyRect.Height / 2)
        
        Write-Host "Double-clicking notification area at ($centerX, $centerY)"
        [WindowAPI]::SetCursorPos($centerX, $centerY)
        Start-Sleep -Milliseconds 200
        
        # Double-click
        [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
        [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
        Start-Sleep -Milliseconds 100
        [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTDOWN, 0, 0, 0, [UIntPtr]::Zero)
        [WindowAPI]::mouse_event([WindowAPI]::MOUSEEVENTF_LEFTUP, 0, 0, 0, [UIntPtr]::Zero)
        
        Start-Sleep -Seconds 3
        
        $doubleClickScreenshot = Take-Screenshot -Suffix "after_tray_double_click"
        if ($doubleClickScreenshot) {
            $detectionResults.Screenshots += $doubleClickScreenshot
        }
        
        # Check once more for new windows
        $finalWindowCheck = Find-SimplySignWindows -TargetProcessId $ProcessId
        if ($finalWindowCheck.Count -gt $detectionResults.Summary.WindowsFound) {
            Write-Host "FOUND NEW WINDOWS after additional tray interaction!"
            $detectionResults.Windows += $finalWindowCheck
            $detectionResults.Summary.WindowsFound = $finalWindowCheck.Count
        }
    }
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
Write-Host "System tray detected: $($detectionResults.Summary.SystemTrayDetected)"
Write-Host "SimplySign tray icons found: $($detectionResults.Summary.SimplySignTrayIconsFound)"

if ($detectionResults.NetworkResults) {
    Write-Host "Network monitoring results:"
    Write-Host "  DNS resolutions: $($detectionResults.NetworkResults.DNSCount)"
    Write-Host "  Network activities: $($detectionResults.NetworkResults.Activity.Count)"
}

# Analysis and recommendations
Write-Host ""
Write-Host "=== ANALYSIS AND THEORY VALIDATION ==="

if ($detectionResults.Summary.SimplySignTrayIconsFound -gt 0) {
    Write-Host "THEORY CONFIRMED: SimplySign Desktop is running in system tray!"
    Write-Host "This explains the network activity without visible windows."
    
    if ($detectionResults.Summary.WindowsFound -gt 0) {
        Write-Host "SUCCESS: Tray interaction triggered login dialog appearance!"
        Write-Host "Your theory was correct - newer versions avoid update prompts by running in tray."
    } else {
        Write-Host "Tray icons found but no login dialog appeared after interaction."
        Write-Host "SimplySign may require different activation method or uses web-based login."
    }
    
} elseif ($detectionResults.Summary.SystemTrayDetected) {
    Write-Host "System tray detected but no SimplySign-specific icons identified."
    Write-Host "SimplySign may be using generic icon names or different identification."
    Write-Host "The network activity suggests the application is running but hidden."
    
} else {
    Write-Host "Could not detect system tray functionality."
    Write-Host "Falling back to traditional window detection methods."
}

if ($detectionResults.Summary.OAuth2ActivityDetected) {
    Write-Host ""
    Write-Host "NETWORK ACTIVITY CONFIRMED: OAuth2 communication detected during session"
    Write-Host "This proves SimplySign Desktop is running and attempting to connect to cloud services."
    
    if ($detectionResults.Summary.WindowsFound -eq 0) {
        Write-Host "Network activity without visible windows suggests:"
        Write-Host "  1. Application is running in background (system tray)"
        Write-Host "  2. Login dialog may be web-based or browser-embedded"
        Write-Host "  3. Application may require specific trigger to show login UI"
    }
} else {
    Write-Host ""
    Write-Host "NO OAUTH2 ACTIVITY: Login dialog may not have appeared or no cloud connection"
}

# Specific recommendations based on findings
Write-Host ""
Write-Host "=== RECOMMENDATIONS ==="

if ($detectionResults.Summary.SimplySignTrayIconsFound -gt 0) {
    Write-Host "RECOMMENDED APPROACH: System tray interaction"
    Write-Host "  - Focus on automating tray icon clicks"
    Write-Host "  - Monitor for dialog appearance after tray interaction"
    Write-Host "  - This approach bypasses update dialog issues completely"
    
} elseif ($detectionResults.Summary.OAuth2ActivityDetected -and $detectionResults.Summary.WindowsFound -eq 0) {
    Write-Host "RECOMMENDED APPROACH: Web-based login detection"
    Write-Host "  - Application may be using embedded browser for login"
    Write-Host "  - Consider browser automation instead of Windows UI automation"
    Write-Host "  - Monitor network traffic for OAuth2 login pages"
    
} elseif ($detectionResults.Summary.InputFieldsFound -gt 0) {
    Write-Host "RECOMMENDED APPROACH: Traditional UI automation"
    Write-Host "  - Input fields detected - credential injection possible"
    Write-Host "  - Use existing Windows UI automation methods"
    
} else {
    Write-Host "RECOMMENDED APPROACH: Hybrid detection"
    Write-Host "  - Combine system tray monitoring with network activity detection"
    Write-Host "  - Try multiple activation methods (tray clicks, keyboard shortcuts, etc.)"
    Write-Host "  - Consider that newer versions may have different UI patterns"
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
    Write-Host "Focus on system tray interaction or web-based login methods"
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
