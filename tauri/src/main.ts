import { invoke } from "@tauri-apps/api/core";

let appDiv: HTMLElement | null;

async function initializeApp() {
  // Get reference to the app div
  appDiv = document.querySelector("#app");
  
  if (appDiv) {
    appDiv.innerHTML = `
      <div style="padding: 20px; text-align: center; font-family: Arial, sans-serif;">
        <h1>DocuScope CA Desktop</h1>
        <p>Initializing Streamlit application...</p>
        <div style="margin-top: 20px;">
          <div style="display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #a617a5; border-radius: 50%; animation: spin 1s linear infinite;"></div>
        </div>
        <p style="margin-top: 20px; color: #666; font-size: 14px;">
          This may take a few moments on first launch...
        </p>
      </div>
    `;
  }

  try {
    // Initialize the Streamlit backend
    await invoke("initialize_streamlit");
    
    // Once initialized, you can embed the Streamlit app
    // For now, just show a success message
    if (appDiv) {
      appDiv.innerHTML = `
        <div style="padding: 20px; text-align: center; font-family: Arial, sans-serif;">
          <h1>DocuScope CA Desktop</h1>
          <p style="color: green;">✓ Application initialized successfully!</p>
          <p>Loading interface...</p>
        </div>
      `;
    }
  } catch (error) {
    console.error("Failed to initialize application:", error);
    if (appDiv) {
      appDiv.innerHTML = `
        <div style="padding: 20px; text-align: center; font-family: Arial, sans-serif;">
          <h1>DocuScope CA Desktop</h1>
          <p style="color: red;">✗ Failed to initialize application</p>
          <p style="font-size: 14px; color: #666;">Error: ${error}</p>
        </div>
      `;
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  initializeApp();
});

// Add CSS for spinner animation
const style = document.createElement('style');
style.textContent = `
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;
document.head.appendChild(style);
