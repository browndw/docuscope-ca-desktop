use log::{error, info, warn};
use std::env;

use reqwest::Client;
use std::time::Duration;
use tauri::{Manager, WindowEvent};
use tauri_plugin_shell::ShellExt;
use tokio::time::sleep;
use sysinfo::{System, Signal};

// Global variable to track the sidecar PID
static mut SIDECAR_PID: Option<u32> = None;

pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let splash_window = app.get_webview_window("splashscreen").unwrap();
            let main_window = app.get_webview_window("main").unwrap();
            let sidecar = app.shell().sidecar("docuscope").unwrap();
            
            // Only clean up processes by name on startup (safer)
            cleanup_startup_processes();

            // Clone the splash window for use in the async task
            let splash_window_clone = splash_window.clone();

            tauri::async_runtime::spawn(async move {
                // Stage 1: Starting services (0-25%)
                let _ = splash_window_clone.eval("updateProgress(5, 1, 'Starting DocuScope services...')");
                
                let (_rx, child) = sidecar.spawn().expect("Failed to spawn sidecar");
                let client = Client::new();
                
                // Try to get the PID of the spawned process
                let pid = child.pid();
                info!("Spawned sidecar process with PID: {}", pid);
                
                // Store the PID globally so we can kill it later
                unsafe {
                    SIDECAR_PID = Some(pid);
                }

                let _ = splash_window_clone.eval("updateProgress(15, 1, 'DocuScope process started')");
                
                // Stage 2: Wait for server to be available (15-25%)
                let _ = splash_window_clone.eval("updateProgress(20, 2, 'Starting Streamlit server...')");
                
                let mut attempt = 0;
                let max_server_attempts = 30; // Reduced to 15 seconds max wait
                let mut server_ready = false;
                
                // Wait for server to become available before loading URL
                loop {
                    attempt += 1;
                    
                    // Update progress during server startup (20% to 25%)
                    let progress = std::cmp::min(20 + (attempt * 5 / max_server_attempts), 25);
                    let _ = splash_window_clone.eval(&format!("updateProgress({}, 2, 'Waiting for server...')", progress));
                    
                    match client.get("http://localhost:8501").send().await {
                        Ok(response) if response.status().is_success() => {
                            info!("Streamlit server is ready - loading URL now");
                            server_ready = true;
                            break;
                        }
                        _ => {
                            if attempt >= max_server_attempts {
                                error!("Streamlit server failed to start after {} attempts", max_server_attempts);
                                let _ = splash_window_clone.eval("updateProgress(25, 2, 'Error: Server startup timeout')");
                                break;
                            }
                            
                            if attempt % 5 == 0 {
                                info!("Waiting for server... (attempt {}/{})", attempt, max_server_attempts);
                            }
                        }
                    }
                    
                    sleep(Duration::from_millis(500)).await;
                }
                
                if server_ready {
                    // Stage 3: Load URL as soon as server is ready (25-30%)
                    let _ = splash_window_clone.eval("updateProgress(25, 3, 'Server ready, loading application...')");
                    
                    // Load the Streamlit URL in the main window
                    main_window
                        .eval("window.location.replace('http://localhost:8501');")
                        .expect("Failed to load the URL in the main window");
                    
                    let _ = splash_window_clone.eval("updateProgress(30, 3, 'Application loading...')");
                    
                    // Show progress while Streamlit loads (30% to 95% over 50 seconds)
                    let total_load_time = 50;
                    
                    for i in 0..total_load_time {
                        let progress = 30 + (i * 65 / total_load_time); // 30% to 95%
                        
                        let stage_msg = if progress < 45 {
                            "Initializing NLP components..."
                        } else if progress < 60 {
                            "Loading corpus analysis models..."
                        } else if progress < 75 {
                            "Preparing user interface..."
                        } else if progress < 90 {
                            "Loading application components..."
                        } else {
                            "Finalizing application startup..."
                        };
                        
                        let _ = splash_window_clone.eval(&format!("updateProgress({}, 3, '{}')", progress, stage_msg));
                        sleep(Duration::from_millis(1000)).await;
                    }
                    
                    let _ = splash_window_clone.eval("updateProgress(95, 3, 'Application ready')");
                } else {
                    // Server failed to start, show error but continue
                    let _ = splash_window_clone.eval("updateProgress(95, 3, 'Server startup failed')");
                }
                
                // Stage 4: Final preparations (95-100%)
                let _ = splash_window_clone.eval("updateProgress(100, 4, 'DocuScope CA Ready!')");
                
                // Wait a bit to show completion
                sleep(Duration::from_millis(1500)).await;
                
                // Hide splash and show main window (main window should already be loaded by now)
                splash_window.hide().unwrap();
                main_window.show().unwrap();       
            });

            Ok(())
        })
        .on_window_event(|window, event| match event {
            WindowEvent::CloseRequested { .. } => {
                if window.label() == "splashscreen" || window.label() == "main" {
                    info!("Close requested - cleaning up docuscope processes");
                    cleanup_docuscope_processes();
                    window.app_handle().exit(0);
                }
            }
            WindowEvent::Destroyed => {
                if window.label() == "main" {
                    info!("Main window destroyed - cleaning up docuscope processes");
                    cleanup_docuscope_processes();
                    window.app_handle().exit(0);
                }
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

/// Comprehensive cleanup of docuscope processes using multiple strategies
fn cleanup_docuscope_processes() {
    info!("Starting comprehensive docuscope process cleanup");
    
    // Strategy 0: Kill the tracked sidecar PID and its children
    kill_tracked_sidecar_process();
    
    // Strategy 1: Kill processes by port (most reliable)
    kill_processes_by_port(8501);
    
    // Strategy 2: Kill processes by name (backup)
    kill_processes_by_name("docuscope");
    
    // Strategy 3: macOS-specific PyInstaller bootloader cleanup
    kill_pyinstaller_bootloader_processes();
    
    // Give processes time to terminate
    std::thread::sleep(Duration::from_millis(200));
    
    // Strategy 4: Force kill any remaining processes
    force_kill_remaining_processes();
    
    // Strategy 5: Final nuclear option - pkill everything
    nuclear_cleanup();
    
    info!("Docuscope process cleanup completed");
}

/// Safe startup cleanup - only kills processes by name
fn cleanup_startup_processes() {
    info!("Starting safe startup cleanup");
    
    let s = System::new_all();
    let processes: Vec<_> = s.processes_by_exact_name("docuscope".as_ref()).collect();
    
    if processes.is_empty() {
        info!("No existing docuscope processes found");
        return;
    }
    
    info!("Found {} existing docuscope processes, cleaning up", processes.len());
    
    for process in processes {
        let process_name = process.name().to_string_lossy();
        info!("Terminating existing process: {} (PID: {})", process_name, process.pid());
        
        // Try graceful kill first
        if process.kill_with(Signal::Term).is_some() {
            info!("Gracefully terminated process {} (PID: {})", process_name, process.pid());
        } else {
            warn!("Failed to terminate process {} (PID: {})", process_name, process.pid());
        }
    }
    
    // Give processes time to terminate
    std::thread::sleep(Duration::from_millis(100));
    
    info!("Startup cleanup completed");
}

/// Kill the tracked sidecar process and its children
fn kill_tracked_sidecar_process() {
    unsafe {
        if let Some(pid) = SIDECAR_PID {
            info!("Killing tracked sidecar process with PID: {}", pid);
            
            // First try to kill the process group (negative PID)
            let kill_group_result = std::process::Command::new("kill")
                .args(&["-TERM", &format!("-{}", pid)])
                .output();
            
            match kill_group_result {
                Ok(output) => {
                    if output.status.success() {
                        info!("Successfully terminated process group for PID: {}", pid);
                    } else {
                        info!("Process group termination failed, trying individual process");
                        
                        // Try individual process kill
                        let kill_result = std::process::Command::new("kill")
                            .args(&["-KILL", &pid.to_string()])
                            .output();
                        
                        match kill_result {
                            Ok(kill_output) => {
                                if kill_output.status.success() {
                                    info!("Successfully killed individual process PID: {}", pid);
                                } else {
                                    error!("Failed to kill tracked process PID: {}", pid);
                                }
                            }
                            Err(e) => error!("Error killing tracked process {}: {}", pid, e),
                        }
                    }
                }
                Err(e) => error!("Error killing tracked process group {}: {}", pid, e),
            }
            
            // Also try to find and kill child processes
            let pgrep_result = std::process::Command::new("pgrep")
                .args(&["-P", &pid.to_string()])
                .output();
            
            match pgrep_result {
                Ok(output) => {
                    if output.status.success() {
                        let children_str = String::from_utf8_lossy(&output.stdout);
                        let child_pids: Vec<&str> = children_str.trim().split('\n').filter(|s| !s.is_empty()).collect();
                        
                        if !child_pids.is_empty() {
                            info!("Found {} child processes of PID {}", child_pids.len(), pid);
                            
                            for child_pid_str in child_pids {
                                if let Ok(child_pid) = child_pid_str.parse::<u32>() {
                                    info!("Killing child process: {}", child_pid);
                                    
                                    let kill_child_result = std::process::Command::new("kill")
                                        .args(&["-KILL", &child_pid.to_string()])
                                        .output();
                                    
                                    match kill_child_result {
                                        Ok(kill_output) => {
                                            if kill_output.status.success() {
                                                info!("Successfully killed child process PID: {}", child_pid);
                                            } else {
                                                error!("Failed to kill child process PID: {}", child_pid);
                                            }
                                        }
                                        Err(e) => error!("Error killing child process {}: {}", child_pid, e),
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => error!("Error finding child processes: {}", e),
            }
            
            // Reset the PID after cleanup
            SIDECAR_PID = None;
        } else {
            info!("No tracked sidecar PID found");
        }
    }
}

/// Kill processes using the specified port via system commands
fn kill_processes_by_port(port: u16) {
    info!("Killing processes on port {}", port);
    
    // Use lsof to find processes using the port
    let lsof_output = std::process::Command::new("lsof")
        .args(&["-ti", &format!(":{}", port)])
        .output();
    
    match lsof_output {
        Ok(output) => {
            if output.status.success() {
                let pids_str = String::from_utf8_lossy(&output.stdout);
                let pids: Vec<&str> = pids_str.trim().split('\n').filter(|s| !s.is_empty()).collect();
                
                if pids.is_empty() {
                    info!("No processes found on port {}", port);
                    return;
                }
                
                info!("Found {} processes on port {}", pids.len(), port);
                
                for pid_str in pids {
                    if let Ok(pid) = pid_str.parse::<u32>() {
                        // Try graceful termination first
                        let term_result = std::process::Command::new("kill")
                            .args(&["-TERM", &pid.to_string()])
                            .output();
                        
                        match term_result {
                            Ok(term_output) => {
                                if term_output.status.success() {
                                    info!("Gracefully terminated process with PID: {}", pid);
                                    std::thread::sleep(Duration::from_millis(50));
                                    
                                    // Check if process is still running
                                    let check_result = std::process::Command::new("kill")
                                        .args(&["-0", &pid.to_string()])
                                        .output();
                                    
                                    if check_result.map_or(false, |o| o.status.success()) {
                                        // Process still running, force kill
                                        let kill_result = std::process::Command::new("kill")
                                            .args(&["-KILL", &pid.to_string()])
                                            .output();
                                        
                                        match kill_result {
                                            Ok(kill_output) => {
                                                if kill_output.status.success() {
                                                    info!("Force killed process with PID: {}", pid);
                                                } else {
                                                    error!("Failed to force kill process with PID: {}", pid);
                                                }
                                            }
                                            Err(e) => error!("Error force killing process {}: {}", pid, e),
                                        }
                                    }
                                } else {
                                    warn!("Failed to gracefully terminate process with PID: {}", pid);
                                }
                            }
                            Err(e) => error!("Error terminating process {}: {}", pid, e),
                        }
                    }
                }
            } else {
                warn!("lsof failed to find processes on port {}", port);
            }
        }
        Err(e) => error!("Error running lsof: {}", e),
    }
}

/// Kill processes by exact name using sysinfo
fn kill_processes_by_name(name: &str) {
    info!("Killing processes by name: {}", name);
    
    let s = System::new_all();
    let processes: Vec<_> = s.processes_by_exact_name(name.as_ref()).collect();
    
    if processes.is_empty() {
        info!("No processes found with name: {}", name);
        return;
    }
    
    for process in processes {
        let process_name = process.name().to_string_lossy();
        info!("Killing process: {} (PID: {})", process_name, process.pid());
        
        // Try graceful kill first
        if process.kill_with(Signal::Term).is_some() {
            info!("Gracefully killed process {} (PID: {})", process_name, process.pid());
            std::thread::sleep(Duration::from_millis(50));
            
            // Check if process is still running by trying to refresh
            let mut new_system = System::new_all();
            new_system.refresh_all();
            
            if new_system.process(process.pid()).is_some() {
                // Process still running, force kill
                if process.kill_with(Signal::Kill).is_some() {
                    info!("Force killed process {} (PID: {})", process_name, process.pid());
                } else {
                    error!("Failed to force kill process {} (PID: {})", process_name, process.pid());
                }
            }
        } else {
            warn!("Failed to gracefully kill process {} (PID: {})", process_name, process.pid());
            
            // Try force kill
            if process.kill_with(Signal::Kill).is_some() {
                info!("Force killed process {} (PID: {})", process_name, process.pid());
            } else {
                error!("Failed to force kill process {} (PID: {})", process_name, process.pid());
            }
        }
    }
}

/// Final cleanup - force kill any remaining docuscope processes
fn force_kill_remaining_processes() {
    info!("Performing final cleanup check");
    
    let s = System::new_all();
    let remaining_processes: Vec<_> = s.processes_by_exact_name("docuscope".as_ref()).collect();
    
    if remaining_processes.is_empty() {
        info!("No remaining docuscope processes found");
        return;
    }
    
    warn!("Found {} remaining docuscope processes, force killing", remaining_processes.len());
    
    for process in remaining_processes {
        let process_name = process.name().to_string_lossy();
        if process.kill_with(Signal::Kill).is_some() {
            info!("Force killed remaining process {} (PID: {})", process_name, process.pid());
        } else {
            error!("Failed to force kill remaining process {} (PID: {})", process_name, process.pid());
        }
    }
}

/// macOS-specific PyInstaller bootloader cleanup
fn kill_pyinstaller_bootloader_processes() {
    info!("Performing macOS-specific PyInstaller bootloader cleanup");
    
    // Find processes with docuscope in the command line (catches bootloader processes)
    let ps_output = std::process::Command::new("ps")
        .args(&["aux"])
        .output();
    
    match ps_output {
        Ok(output) => {
            if output.status.success() {
                let ps_str = String::from_utf8_lossy(&output.stdout);
                let lines: Vec<&str> = ps_str.lines().collect();
                
                for line in lines {
                    if line.contains("docuscope") && !line.contains("grep") {
                        // Extract PID (second column)
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(pid) = parts[1].parse::<u32>() {
                                info!("Found docuscope process: PID {} - {}", pid, line);
                                
                                // Kill the process group to catch child processes
                                let kill_group_result = std::process::Command::new("kill")
                                    .args(&["-TERM", &format!("-{}", pid)])
                                    .output();
                                
                                match kill_group_result {
                                    Ok(term_output) => {
                                        if term_output.status.success() {
                                            info!("Terminated process group for PID: {}", pid);
                                        } else {
                                            // Try individual process kill
                                            let kill_result = std::process::Command::new("kill")
                                                .args(&["-KILL", &pid.to_string()])
                                                .output();
                                            
                                            match kill_result {
                                                Ok(kill_output) => {
                                                    if kill_output.status.success() {
                                                        info!("Force killed individual process PID: {}", pid);
                                                    } else {
                                                        error!("Failed to kill process PID: {}", pid);
                                                    }
                                                }
                                                Err(e) => error!("Error killing process {}: {}", pid, e),
                                            }
                                        }
                                    }
                                    Err(e) => error!("Error killing process group {}: {}", pid, e),
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => error!("Error running ps command: {}", e),
    }
}

/// Nuclear option - use pkill to kill all docuscope processes
fn nuclear_cleanup() {
    info!("Performing nuclear cleanup with pkill");
    
    // First try pkill with signal
    let pkill_result = std::process::Command::new("pkill")
        .args(&["-f", "docuscope"])
        .output();
    
    match pkill_result {
        Ok(output) => {
            if output.status.success() {
                info!("Successfully ran pkill -f docuscope");
            } else {
                warn!("pkill command failed or no processes found");
            }
        }
        Err(e) => error!("Error running pkill: {}", e),
    }
    
    // Give processes time to die
    std::thread::sleep(Duration::from_millis(100));
    
    // Force kill anything still running
    let pkill_force_result = std::process::Command::new("pkill")
        .args(&["-9", "-f", "docuscope"])
        .output();
    
    match pkill_force_result {
        Ok(output) => {
            if output.status.success() {
                info!("Successfully ran pkill -9 -f docuscope");
            } else {
                info!("pkill -9 command completed (may have found no processes)");
            }
        }
        Err(e) => error!("Error running pkill -9: {}", e),
    }
}
