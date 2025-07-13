"""
Desktop-specific build and configuration tests.

This test suite focuses on validating the build process and desktop-specific
functionality rather than duplicating tests from the template repository.
"""

import os
from pathlib import Path

import pytest

from webapp.utilities.configuration.config_manager import ConfigurationManager


class TestDesktopBuild:
    """Test desktop build process and PyInstaller compatibility."""

    def test_pyinstaller_path_resolution(self):
        """Test that path resolution works in PyInstaller bundle context."""
        config_manager = ConfigurationManager()

        # Test project root detection
        project_root = config_manager.get_project_root()
        assert isinstance(project_root, Path)
        assert project_root.exists()

        # Verify essential directories exist
        webapp_dir = project_root / "webapp"
        assert webapp_dir.exists(), f"webapp directory not found at {webapp_dir}"

    def test_config_file_accessibility(self):
        """Test that config files are accessible in bundled environment."""
        config_manager = ConfigurationManager()

        # Test config.toml accessibility
        config_path = config_manager.get_config_path()
        assert config_path.exists(), f"Config file not found at {config_path}"

        # Test config loading
        config = config_manager.load_config()
        assert isinstance(config, dict)
        assert 'global' in config

    def test_static_file_paths(self):
        """Test that static file paths resolve correctly."""
        config_manager = ConfigurationManager()

        # Test essential static files
        static_files = [
            config_manager.docuscope_logo_path,
            config_manager.porpoise_badge_path,
            config_manager.user_guide_badge_path,
        ]

        for file_path in static_files:
            assert os.path.exists(file_path), f"Static file not found: {file_path}"

    def test_model_paths(self):
        """Test that model paths are correctly resolved."""
        config_manager = ConfigurationManager()

        # Test model directory paths
        models_dir = config_manager.models_dir_path
        assert os.path.exists(models_dir), f"Models directory not found: {models_dir}"

        # Test specific model paths
        large_model = config_manager.model_large_path
        small_model = config_manager.model_small_path

        # At least one model should exist
        assert (os.path.exists(large_model) or os.path.exists(small_model)), \
            "No spaCy models found"


class TestDesktopConfiguration:
    """Test desktop-specific configuration management."""

    def test_desktop_mode_default(self):
        """Test that desktop mode is properly detected/set."""
        config_manager = ConfigurationManager()

        # Desktop mode should be True for desktop builds
        assert config_manager.desktop_mode is True

    def test_cache_mode_disabled_in_desktop(self):
        """Test that cache mode is disabled in desktop mode."""
        config_manager = ConfigurationManager()

        # Cache should be disabled in desktop mode
        assert config_manager.cache_mode is False

    def test_runtime_sanity_checks(self):
        """Test that runtime sanity checks work correctly."""
        config_manager = ConfigurationManager()

        # Load config and verify sanity checks applied
        config = config_manager.load_config()

        # If desktop mode is true, cache should be false
        if config['global']['desktop_mode']:
            assert config['cache']['cache_mode'] is False

    def test_version_extraction(self):
        """Test that version can be extracted from pyproject.toml."""
        from webapp.utilities.configuration.config_manager import get_version_from_pyproject

        version = get_version_from_pyproject()
        assert isinstance(version, str)
        assert version != "0.0.0"  # Should find actual version


class TestEntrypointConfiguration:
    """Test the entrypoint.py configuration."""

    def test_entrypoint_file_exists(self):
        """Test that entrypoint.py exists and is readable."""
        project_root = Path(__file__).resolve().parents[1]
        entrypoint_path = project_root / "entrypoint.py"

        assert entrypoint_path.exists(), "entrypoint.py not found"

        # Test it's readable and contains expected content
        content = entrypoint_path.read_text()
        assert "streamlit" in content.lower()
        assert "webapp/index.py" in content

    def test_streamlit_args_configuration(self):
        """Test that Streamlit arguments are properly configured."""
        project_root = Path(__file__).resolve().parents[1]
        entrypoint_path = project_root / "entrypoint.py"

        content = entrypoint_path.read_text()

        # Check for essential Streamlit arguments
        required_args = [
            "--client.showSidebarNavigation=false",
            "--client.toolbarMode=minimal", 
            "--server.headless=true",
            "--global.developmentMode=false"
        ]

        for arg in required_args:
            assert arg in content, f"Missing Streamlit argument: {arg}"


class TestTauriIntegration:
    """Test Tauri-specific integration points."""

    def test_tauri_directory_structure(self):
        """Test that Tauri directory structure is correct."""
        project_root = Path(__file__).resolve().parents[1]
        tauri_dir = project_root / "tauri"

        assert tauri_dir.exists(), "Tauri directory not found"

        # Check essential Tauri files
        essential_files = [
            tauri_dir / "package.json",
            tauri_dir / "src-tauri" / "Cargo.toml",
            tauri_dir / "src-tauri" / "tauri.conf.json",
            tauri_dir / "src-tauri" / "src" / "lib.rs",
        ]

        for file_path in essential_files:
            assert file_path.exists(), f"Essential Tauri file missing: {file_path}"

    def test_tauri_config_validity(self):
        """Test that Tauri configuration is valid."""
        import json

        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "tauri" / "src-tauri" / "tauri.conf.json"

        # Load and validate JSON
        with open(config_path) as f:
            config = json.load(f)

        # Check essential configuration
        assert "productName" in config
        assert "identifier" in config
        assert "app" in config
        assert "bundle" in config

        # Check binary configuration
        assert "externalBin" in config["bundle"]
        assert "docuscope" in str(config["bundle"]["externalBin"])

    def test_binary_directory_exists(self):
        """Test that binary directory exists for Tauri."""
        project_root = Path(__file__).resolve().parents[1]
        binaries_dir = project_root / "tauri" / "src-tauri" / "binaries"

        assert binaries_dir.exists(), "Tauri binaries directory not found"


class TestBuildArtifacts:
    """Test build artifacts and requirements."""

    def test_requirements_files_exist(self):
        """Test that requirements files exist and are readable."""
        project_root = Path(__file__).resolve().parents[1]

        requirements_files = [
            project_root / "requirements.txt",
            project_root / "requirements-deploy.txt",
        ]

        for req_file in requirements_files:
            assert req_file.exists(), f"Requirements file missing: {req_file}"

            # Check it's not empty
            content = req_file.read_text().strip()
            assert len(content) > 0, f"Requirements file is empty: {req_file}"

    def test_pyproject_toml_validity(self):
        """Test that pyproject.toml is valid and contains required fields."""
        import tomllib

        project_root = Path(__file__).resolve().parents[1]
        pyproject_path = project_root / "pyproject.toml"

        assert pyproject_path.exists(), "pyproject.toml not found"

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        # Check essential project metadata
        assert "project" in config
        assert "name" in config["project"]
        assert "version" in config["project"]
        assert "docuscope-ca-desktop" in config["project"]["name"]

    def test_pyinstaller_spec_exists(self):
        """Test that PyInstaller spec file exists."""
        project_root = Path(__file__).resolve().parents[1]
        spec_file = project_root / "entrypoint.spec"

        assert spec_file.exists(), "entrypoint.spec not found"

        # Check it contains expected content
        content = spec_file.read_text()
        assert "entrypoint.py" in content
        assert "Analysis" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
