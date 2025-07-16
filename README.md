# DocuScope CA Desktop

<div class="image" align="center">
    <img width="150" height="auto" src="webapp/_static/docuscope-logo.png" alt="DocuScope logo">
    <br>
</div>

---

[![License][license]](https://github.com/browndw/docuscope-ca-desktop/blob/main/LICENSE) [![Python][python]](https://www.python.org/downloads/) [![Streamlit][streamlit]](https://streamlit.io) [![spaCy][spacy]](https://spacy.io) [![Build][build]](https://github.com/browndw/docuscope-ca-desktop/actions)

## Download DocuScope CA Desktop

Get the latest version of DocuScope CA Desktop for your operating system:

### Latest Release

[Download DocuScope CA for Windows (.exe)][windows]

[Download DocuScope CA for macOS Intel (.dmg)][mac-intel]

[Download DocuScope CA for macOS Apple Silicon (.dmg)][mac-arm64]

[Download DocuScope CA for Linux (.deb)][linux-deb]

[Download DocuScope CA for Linux (.AppImage)][linux-appimage]

---

## About DocuScope CA Desktop

DocuScope CA Desktop is a standalone desktop application for corpus analysis and concordancing, combining part-of-speech tagging with DocuScope rhetorical analysis.

With the desktop application users can:

1. process small to medium-sized corpora
2. create frequency tables of words, phrases, and tags
3. calculate associations around node words
4. generate key word in context (KWIC) tables
5. compare corpora or sub-corpora
6. explore single texts
7. practice advanced plotting

## Installation

### System Requirements

- **Windows**: Windows 10 or later (64-bit)
- **macOS**: macOS 10.15 (Catalina) or later
- **Linux**: Ubuntu 20.04 or equivalent (64-bit)

### Quick Installation

1. Download the appropriate installer for your operating system from the links above
2. Run the installer and follow the setup instructions
3. Launch DocuScope CA Desktop from your applications folder or start menu

### Manual Installation

If you prefer to build from source or need to customize the installation:

1. Clone this repository
2. Install Python 3.11 or higher
3. Install Poetry for dependency management
4. Install Rust and Tauri CLI for building the desktop application

```bash
# Clone the repository
git clone https://github.com/browndw/docuscope-ca-desktop.git
cd docuscope-ca-desktop

# Install Python dependencies
poetry install

# Build the desktop application
cd tauri
pnpm install
pnpm tauri build
```

## Features

This desktop application provides a comprehensive suite of tools for corpus analysis:

- **Corpus Processing**: Upload and process small to medium-sized text corpora
- **Dual Tagging**: Combines part-of-speech tagging with DocuScope rhetorical analysis
- **Frequency Analysis**: Generate detailed frequency tables for words, phrases, and rhetorical tags
- **Collocation Analysis**: Calculate statistical associations around target words
- **KWIC Tables**: Create keyword-in-context concordances for detailed text examination
- **Comparative Analysis**: Compare different corpora or sub-sections of the same corpus
- **Single Document Explorer**: In-depth analysis of individual texts
- **Advanced Visualization**: Interactive plotting tools for data exploration
- **Offline Operation**: Works completely offline - no internet connection required
- **Cross-Platform**: Available for Windows, macOS, and Linux

## Configuration

The application behavior can be customized through the configuration files. The desktop version is pre-configured for optimal standalone use with:

- Desktop mode enabled by default
- Simplified interface for individual researchers
- Local file processing without external dependencies
- Optimized performance for desktop environments

## Usage Examples

### Basic Corpus Analysis

1. Launch DocuScope CA Desktop
2. Load your corpus using the "Manage Corpora" page
3. Generate frequency tables for initial exploration
4. Use KWIC tables to examine specific terms in context
5. Create visualizations to identify patterns

### Comparative Studies

1. Upload multiple corpora or define sub-corpora
2. Use "Compare Corpora" tools to identify statistical differences
3. Generate comparative visualizations
4. Export results for further analysis

## Development

### Building from Source

To build the desktop application from source:

```bash
# Install dependencies
poetry install

# Build Python binary
poetry run pyinstaller entrypoint.spec

# Build desktop application
cd tauri
pnpm install
pnpm tauri build
```

### Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code style requirements
- Testing procedures
- Pull request process
- Issue reporting

## License

Code licensed under `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
See `LICENSE <https://github.com/browndw/docuscope-ca-desktop/blob/main/LICENSE>`_ file.

## Citation

If you use this software in your research, please cite:

```bibtex
[Citation details to be added upon JOSS publication]
```

## Acknowledgments

- **DocuScope**: This project builds upon the DocuScope rhetorical analysis framework developed at Carnegie Mellon University
- **spaCy**: Natural language processing capabilities provided by the spaCy library
- **Streamlit**: Web application framework enabling accessible deployment
- **Tauri**: Desktop application framework for building native apps

## Support

For questions, bug reports, or feature requests:

- Open an issue on [GitHub Issues](https://github.com/browndw/docuscope-ca-desktop/issues)
- Check our [documentation](https://browndw.github.io/docuscope-docs/) for detailed guides

---

<!-- Download links - automatically updated by release workflow -->
[windows]: https://github.com/browndw/docuscope-ca-desktop/releases/latest/download/docuscope-ca-desktop-x86_64-pc-windows-msvc.exe
[mac-intel]: https://github.com/browndw/docuscope-ca-desktop/releases/latest/download/docuscope-ca-desktop-x86_64-apple-darwin.dmg
[mac-arm64]: https://github.com/browndw/docuscope-ca-desktop/releases/latest/download/docuscope-ca-desktop-aarch64-apple-darwin.dmg
[linux-deb]: https://github.com/browndw/docuscope-ca-desktop/releases/latest/download/docuscope-ca-desktop-x86_64-unknown-linux-gnu.deb
[linux-appimage]: https://github.com/browndw/docuscope-ca-desktop/releases/latest/download/docuscope-ca-desktop-x86_64-unknown-linux-gnu.AppImage

[license]: https://img.shields.io/github/license/browndw/docuscope-ca-desktop
[python]: https://img.shields.io/badge/python-3.11%2B-blue
[streamlit]: https://static.streamlit.io/badges/streamlit_badge_black_white.svg
[spacy]: https://img.shields.io/badge/made%20with%20❤%20and-spaCy-09a3d5.svg
[build]: https://github.com/browndw/docuscope-ca-desktop/actions/workflows/build-tauri.yml/badge.svg
