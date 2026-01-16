# afterseed

A zero-dependency Python post-processing script for qBittorrent that handles hardlinking media files to library directories without breaking seeding.

## Features

- **Zero Dependencies**: Uses only Python standard library
- **Safe**: No deletions or moves of source files
- **Idempotent**: Can be run multiple times safely
- **BDMV Support**: Handles Blu-ray disc structures
- **Multi-file Support**: Processes single files and multi-file torrents
- **API Integration**: Notifies Radarr, Sonarr, and Plex
- **Cross-host Support**: Works with NFS mounts for remote libraries
- **Retry Logic**: Robust API calls with exponential backoff

## Installation

1. Clone the repository
2. Install Python 3.14+
3. Copy `config.example.json` to `config.json` and configure
4. Set up as qBittorrent external program

## Configuration

See `config.example.json` for configuration options.

## Testing

Run the test suite:

```bash
pip install pytest
python -m pytest tests/
```

Run internal validation tests:

```bash
python afterseed3.py --test
```

## Development

This project uses:
- **black** for code formatting
- **isort** for import sorting
- **pytest** for testing
- **pre-commit** for code quality hooks

Install development dependencies:

```bash
pip install pre-commit pytest
pre-commit install
```

## License

See LICENSE file.
