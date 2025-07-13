# Desktop Build Instructions

This document provides instructions for building the desktop version of DocuScope CA using Tauri.

## Prerequisites

- Node.js 18+ and npm
- Rust toolchain
- Python 3.10+
- This repository's codebase

## Repository Structure for Desktop Fork

When forking this repository for desktop development, you'll need to add:

```
docuscope-ca-desktop/
├── webapp/                    # Core application (from this repo)
├── src-tauri/                 # Tauri configuration
│   ├── Cargo.toml
│   ├── tauri.conf.json
│   └── src/
│       └── main.rs
├── package.json               # Node.js dependencies
├── tauri-build.yml           # GitHub Actions for builds
└── desktop-specific files...
```

## Key Integration Points

### 1. Python Environment Setup
The Tauri app needs to:
- Bundle Python runtime
- Install requirements.txt dependencies  
- Set up proper paths for the webapp/ directory

### 2. Configuration Changes
- Copy `.streamlit/secrets.toml.template` to `.streamlit/secrets.toml`
- Ensure `desktop_mode = true` in options.toml
- Configure local data directories

### 3. Path Resolution
The current `index.py` already includes Tauri-compatible path setup:

```python
# Tauri-compatible path setup - finds project root reliably
project_root = pathlib.Path(__file__).resolve()
for _ in range(10):  # Search up to 10 levels
    if (project_root / 'webapp').exists() or (project_root / 'pyproject.toml').exists():
        break
    project_root = project_root.parent
```

This should work correctly when the webapp/ directory is bundled in a Tauri app.

## Build Process

### Development Build
```bash
# Install Tauri CLI
npm install -g @tauri-apps/cli

# Install dependencies
npm install

# Set up Python environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Copy configuration
cp .streamlit/secrets.toml.template .streamlit/secrets.toml

# Run in development mode
npm run tauri dev
```

### Production Build
```bash
# Build for current platform
npm run tauri build

# Cross-platform builds (requires GitHub Actions)
# See tauri-build.yml workflow
```

## Release Integration

The main repository's release workflow creates `docuscope-ca-webapp-{version}.tar.gz` specifically for desktop integration. The desktop repository can:

1. Listen for `repository_dispatch` events from the main repo
2. Download the latest webapp archive
3. Extract it to the correct location
4. Trigger desktop builds

## GitHub Actions Integration

### In Main Repository (this repo)
- Release workflow triggers desktop build via repository dispatch
- Creates webapp-only archive for easy integration

### In Desktop Repository
```yaml
on:
  repository_dispatch:
    types: [core_release]
```

This allows automatic desktop builds when the core webapp is updated.

## Configuration Files Needed in Desktop Fork

### tauri.conf.json
```json
{
  "package": {
    "productName": "DocuScope CA Desktop",
    "version": "1.0.0"
  },
  "build": {
    "distDir": "../webapp",
    "devPath": "http://localhost:8501",
    "beforeDevCommand": "python -m streamlit run webapp/index.py",
    "beforeBuildCommand": ""
  },
  "tauri": {
    "allowlist": {
      "all": false,
      "fs": {
        "all": false,
        "readFile": true,
        "writeFile": true,
        "createDir": true,
        "scope": ["$APPDATA/*", "$HOME/.docuscope-ca/*"]
      },
      "path": {
        "all": true
      },
      "shell": {
        "all": false,
        "execute": false
      }
    },
    "bundle": {
      "active": true,
      "targets": "all",
      "identifier": "com.docuscope.ca",
      "icon": [
        "icons/32x32.png",
        "icons/128x128.png",
        "icons/icon.icns",
        "icons/icon.ico"
      ]
    },
    "security": {
      "csp": null
    },
    "windows": [
      {
        "fullscreen": false,
        "height": 800,
        "resizable": true,
        "title": "DocuScope CA Desktop",
        "width": 1200,
        "minWidth": 800,
        "minHeight": 600
      }
    ]
  }
}
```

### Cargo.toml
```toml
[package]
name = "docuscope-ca-desktop"
version = "1.0.0"
description = "Desktop version of DocuScope Corpus Analysis & Concordancer"
authors = ["David West Brown <dwb2@andrew.cmu.edu>"]
license = "Apache-2.0"
repository = "https://github.com/browndw/docuscope-ca-desktop"
edition = "2021"

[build-dependencies]
tauri-build = { version = "1.0", features = [] }

[dependencies]
serde_json = "1.0"
serde = { version = "1.0", features = ["derive"] }
tauri = { version = "1.0", features = ["api-all"] }

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]
```

## Testing Desktop Integration

### Local Testing
1. Ensure the webapp runs correctly in desktop mode
2. Test file I/O operations work correctly
3. Verify all features work without internet connection
4. Test on all target platforms

### Automated Testing
The desktop repository should include:
- Cross-platform build tests
- Application startup tests  
- Core functionality tests
- Installation/packaging tests

## Deployment Strategy

### Distribution Channels
- GitHub Releases (cross-platform)
- Platform-specific stores (optional)
- Direct download from website

### Update Mechanism
- Tauri's built-in updater
- Check main repository releases
- Download and apply webapp updates

This architecture allows the desktop version to stay synchronized with the main codebase while maintaining platform-specific optimizations and build processes.
