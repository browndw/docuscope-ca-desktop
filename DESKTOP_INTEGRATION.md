# Integration Checklist for Desktop Fork

This checklist ensures smooth integration when forking this repository for desktop development.

## Pre-Fork Setup

- [ ] Review `DESKTOP_BUILD.md` for complete build instructions
- [ ] Ensure you have Rust and Node.js installed for Tauri development
- [ ] Understand the repository structure and key integration points

## Fork Setup Steps

### 1. Initial Fork Configuration
- [ ] Fork this repository to `docuscope-ca-desktop`
- [ ] Clone the forked repository
- [ ] Add this repository as upstream: `git remote add upstream https://github.com/browndw/docuscope-ca-online.git`

### 2. Add Tauri Components
- [ ] Initialize Tauri: `npm create tauri-app`
- [ ] Copy configuration templates from `DESKTOP_BUILD.md`
- [ ] Configure `tauri.conf.json` for proper webapp integration
- [ ] Set up `src-tauri/` directory with Rust components

### 3. Configuration Setup
- [ ] Copy `.streamlit/secrets.toml.template` to `.streamlit/secrets.toml`
- [ ] Set `desktop_mode = true` in `webapp/config/options.toml`
- [ ] Configure local data directories
- [ ] Test configuration with `streamlit run webapp/index.py`

### 4. GitHub Actions Setup
- [ ] Create `.github/workflows/tauri-build.yml` for cross-platform builds
- [ ] Set up `repository_dispatch` listener for upstream releases
- [ ] Configure GitHub secrets for code signing (if needed)
- [ ] Test automated builds

### 5. Release Integration
- [ ] Configure automatic webapp updates from upstream releases
- [ ] Set up version synchronization
- [ ] Test release pipeline

## Ongoing Maintenance

### Staying Synchronized with Upstream
```bash
# Regular sync process
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Handling Upstream Updates
- [ ] Monitor upstream releases
- [ ] Test new versions in desktop environment
- [ ] Update Tauri configuration if needed
- [ ] Release new desktop version

### Testing Checklist
- [ ] Application starts correctly
- [ ] All core features work offline
- [ ] File I/O operations work correctly
- [ ] Cross-platform compatibility maintained
- [ ] Performance acceptable on target platforms

## Key Files to Monitor

These files from upstream may require desktop-specific adaptations:

- `webapp/index.py` - Entry point (path resolution)
- `webapp/config/options.toml` - Configuration settings
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Project metadata
- `.streamlit/secrets.toml.template` - Configuration template

## Troubleshooting Common Issues

### Path Resolution Issues
- Ensure the path resolution logic in `index.py` works with Tauri bundling
- Test with different bundle configurations

### Permission Issues
- Configure Tauri allowlist for file system access
- Test on all target platforms

### Performance Issues
- Monitor memory usage with large corpora
- Optimize Python bundling for size
- Consider lazy loading for better startup times

### Cross-Platform Issues
- Test on Windows, macOS, and Linux
- Handle platform-specific path separators
- Ensure consistent behavior across platforms

## Contact and Support

- Main repository: https://github.com/browndw/docuscope-ca-online
- Desktop repository: https://github.com/browndw/docuscope-ca-desktop (after fork)
- Issues: Use repository-specific issue trackers
- Documentation: https://browndw.github.io/docuscope-docs/
