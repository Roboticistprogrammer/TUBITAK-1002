# TUBITAK-1002
Early Fire Detection with Multi-source Dataset Integrated in Simulation Environment

## Overview
This project focuses on early fire detection using multiple datasets and simulation environments as part of the TUBITAK-1002 research program.

## Dataset Management

The SciDB URLs require manual download due to authentication requirements. This script helps organize the datasets once downloaded.

### Quick Setup

1. **Create dataset structure:**
   ```bash
   python download_dataset.py --setup
   ```

2. **Manually download datasets:**
   - Visit each URL in your browser
   - Register/login to SciDB if needed
   - Download files to respective dataset directories

3. **Extract and organize:**
   ```bash
   python download_dataset.py --extract
   ```

### Dataset Information

The project uses three fire detection datasets:

| Dataset | Description | Manual Download Required |
|---------|-------------|-------------------------|
| FASDD_UAV | Fire and Smoke Detection Dataset - UAV | ✅ |
| Fire_Detection_CV | Fire Detection Dataset - Computer Vision | ✅ |
| Fire_Remote_Sensing | Fire Detection Dataset - Remote Sensing | ✅ |

| Dataset | Description | Command |
|---------|-------------|---------|
| Dataset 1 | Fire Detection Dataset 1(UAV) | `python download_dataset.py --dataset 1` |
| Dataset 2 | Fire Detection Dataset 2(CV) | `python download_dataset.py --dataset 2` |
| Dataset 3 | Fire Detection Dataset 3(RS) | `python download_dataset.py --dataset 3` |

### Usage Examples

```bash
# Show information about all datasets
python download_dataset.py --info

# Download all datasets with automatic extraction and organization
python download_dataset.py --all

# Download dataset 2 only, without extracting
python download_dataset.py --dataset 2 --no-extract

# Download and organize to custom directory
python download_dataset.py --all --base-dir my_datasets

# Clean up archive files after extraction
python download_dataset.py --clean
```

### Dataset Organization

The script automatically organizes downloaded datasets into a standardized structure:

```
datasets/
├── fire_dataset_1/
│   ├── images/          # Image files (.jpg, .png, etc.)
│   ├── annotations/     # Annotation files (.xml, .json, .txt)
│   ├── train/          # Training data
│   ├── val/            # Validation data
│   └── test/           # Test data
├── fire_dataset_2/
│   └── ...
└── fire_dataset_3/
    └── ...
```

### Features

- **Automatic Download**: Downloads datasets from official sources
- **Progress Tracking**: Shows download progress with progress bars
- **Checksum Verification**: Calculates SHA256 checksums for integrity
- **Smart Organization**: Automatically categorizes files by type and purpose
- **Resume Support**: Skips already downloaded files
- **Logging**: Detailed logs in `dataset_download.log`
- **Cleanup Tools**: Remove archive files after extraction

### Requirements

- Python 3.6+
- Internet connection for downloading
- Sufficient disk space (datasets can be large)

See `requirements.txt` for Python package dependencies.
