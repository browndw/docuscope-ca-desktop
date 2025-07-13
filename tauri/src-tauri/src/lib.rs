use log::{info, warn};
use std::env;

use reqwest::Client;
use std::time::Duration;
use tauri::{Manager, WindowEvent};
use tauri_plugin_shell::ShellExt;
use tokio::time::sleep;
use sysinfo::System;

pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let splash_window = app.get_webview_window("splashscreen").unwrap();
            let main_window = app.get_webview_window("main").unwrap();
            let sidecar = app.shell().sidecar("docuscope").unwrap();
            let s = System::new_all();

            for process in s.processes_by_exact_name("docuscope".as_ref()) {
              process.kill();
            }

            tauri::async_runtime::spawn(async move {
                let (_rx, _child) = sidecar.spawn().expect("Failed to spawn sidecar");
                let client = Client::new();

                loop {
                    match client.get("http://localhost:8501").send().await {
                        Ok(response) if response.status().is_success() => {
                            info!("Streamlit server loaded");
                            sleep(Duration::from_millis(500)).await;
                            break;
                        }
                        _ => {
                            warn!("Streamlit server not available, retrying...");
                            sleep(Duration::from_millis(500)).await;
                        }
                    }
                }
                main_window
                    .eval("window.location.replace('http://localhost:8501');")
                    .expect("Failed to load the URL in the main window");

                sleep(Duration::from_millis(250)).await;
                splash_window.hide().unwrap();
                main_window.show().unwrap();       
            });

            Ok(())
        })
        .on_window_event(|window, event| match event {
            WindowEvent::CloseRequested { .. } => {
                if window.label() == "splashscreen" || window.label() == "main" {
                    info!("Close requested - killing docuscope processes");
                    let s = System::new_all();
                    for process in s.processes_by_exact_name("docuscope".as_ref()) {
                        process.kill();
                    }
                    window.app_handle().exit(0);
                }
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
