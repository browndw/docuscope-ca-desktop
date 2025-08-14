let appDiv: HTMLElement | null;

async function initializeApp() {
  // Get reference to the app div
  appDiv = document.querySelector("#app");
  
  if (appDiv) {
    appDiv.innerHTML = `
      <div style="padding: 20px; text-align: center; font-family: Arial, sans-serif;">
        <h1>DocuScope CA Desktop</h1>
        <p>Preparing application...</p>
        <div style="margin-top: 20px;">
          <div style="display: inline-block; width: 20px; height: 20px; border: 3px solid #f3f3f3; border-top: 3px solid #a617a5; border-radius: 50%; animation: spin 1s linear infinite;"></div>
        </div>
        <p style="margin-top: 20px; color: #666; font-size: 14px;">
          The Rust backend is handling Streamlit initialization...
        </p>
      </div>
    `;
  }

  // No need to invoke any commands - the Rust backend automatically
  // handles sidecar spawning, server startup, and URL loading via the splash screen system
  console.log("Main window ready - Rust backend will handle Streamlit initialization");
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
