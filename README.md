# Project Setup

This repository provides analysis utilities for vegetation indices. Use the provided setup script to configure a local Python environment.

## Prerequisites
- Python 3.12 available on your PATH
- Access to a POSIX-compatible shell (macOS, Linux, or WSL)

## Initial Setup
1. Clone the repository.
2. Make the setup script executable (first time only):
   ```bash
   chmod +x setup.sh
   ```
3. Run the setup script to create the `venv` virtual environment and install dependencies:
   ```bash
   ./setup.sh
   ```

The script will:
- Verify that `python3.12` is available
- Create a virtual environment named `venv`
- Upgrade `pip` inside the environment
- Install packages listed in `requirements.txt`

## Activating the Environment
After setup, activate the environment whenever you work on the project:
```bash
source venv/bin/activate
```

When finished, deactivate it with:
```bash
deactivate
```

## Updating Dependencies
If you add or update packages:
1. Modify `requirements.txt` accordingly.
2. Re-run `./setup.sh` to install the new dependencies.
