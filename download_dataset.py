"""
Fire Detection Dataset Setup
============================
Simple script to set up fire detection datasets for TUBITAK-1002 project.
Since the SciDB URLs require authentication, this script provides manual instructions.
"""

import os
from pathlib import Path

# Dataset information
DATASETS = {
    1: {
        "name": "FASDD_UAV",
        "description": "Fire and Smoke Detection Dataset - UAV",
        "url": "https://www.scidb.cn/en/file?fid=9456106d26c5fc6b74143c3707115d39&mode=front",
        "filename": "FASDD_UAV.zip"
    },
    2: {
        "name": "Fire_Detection_CV",
        "description": "Fire Detection Dataset - Computer Vision", 
        "url": "https://www.scidb.cn/en/file?fid=85bad3972ed60e0a20a790032c0d85fb&mode=front",
        "filename": "fire_detection_cv.zip"
    },
    3: {
        "name": "Fire_Remote_Sensing",
        "description": "Fire Detection Dataset - Remote Sensing",
        "url": "https://www.scidb.cn/en/file?fid=0fefc7486bf648a0ce754e093f3a56f2&mode=front", 
        "filename": "fire_remote_sensing.zip"
    }
}

def create_dataset_structure():
    """Create the dataset directory structure."""
    base_dir = Path("datasets")
    base_dir.mkdir(exist_ok=True)
    
    for dataset_id, info in DATASETS.items():
        dataset_dir = base_dir / info["name"]
        dataset_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ["images", "annotations", "train", "val", "test"]:
            (dataset_dir / subdir).mkdir(exist_ok=True)
        
        # Create README
        readme_path = dataset_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {info['description']}\n\n")
            f.write(f"Dataset ID: {dataset_id}\n")
            f.write(f"Source: {info['url']}\n")
            f.write(f"Expected filename: {info['filename']}\n\n")
            f.write("## Manual Download Instructions:\n")
            f.write("1. Visit the URL above in your browser\n")
            f.write("2. Register/login to SciDB if required\n") 
            f.write(f"3. Download the file as '{info['filename']}'\n")
            f.write(f"4. Place the file in this directory: {dataset_dir.absolute()}\n")
            f.write("5. Run: python download_dataset.py --extract\n\n")
            f.write("## Directory Structure:\n")
            f.write("```\n")
            f.write(f"{info['name']}/\n")
            f.write("|-- images/       # Image files\n")
            f.write("|-- annotations/  # Label files\n")
            f.write("|-- train/        # Training data\n")
            f.write("|-- val/          # Validation data\n")
            f.write("`-- test/         # Test data\n")
            f.write("```\n")
    
    print("✅ Dataset structure created!")
    print("\n📁 Created directories:")
    for dataset_id, info in DATASETS.items():
        print(f"   {dataset_id}. {info['name']} - {info['description']}")

def extract_datasets():
    """Extract downloaded dataset files."""
    import zipfile
    import tarfile
    
    base_dir = Path("datasets")
    extracted_count = 0
    
    for dataset_id, info in DATASETS.items():
        dataset_dir = base_dir / info["name"]
        
        # Look for archive files
        archive_files = []
        for ext in ['*.zip', '*.tar.gz', '*.tar', '*.rar']:
            archive_files.extend(dataset_dir.glob(ext))
        
        for archive_path in archive_files:
            print(f"\n📦 Extracting {archive_path.name}...")
            
            try:
                if archive_path.suffix.lower() == '.zip':
                    with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_dir)
                        
                elif '.tar' in archive_path.name.lower():
                    with tarfile.open(archive_path, 'r:*') as tar_ref:
                        tar_ref.extractall(dataset_dir)
                
                print(f"✅ Extracted: {archive_path.name}")
                extracted_count += 1
                
                # Organize files
                organize_files(dataset_dir)
                
            except Exception as e:
                print(f"❌ Failed to extract {archive_path.name}: {e}")
    
    if extracted_count == 0:
        print("❌ No archive files found to extract.")
        print("   Please download the datasets manually first.")
    else:
        print(f"\n✅ Successfully extracted {extracted_count} datasets!")

def organize_files(dataset_dir):
    """Organize extracted files into proper structure."""
    
    for file_path in dataset_dir.rglob("*"):
        if not file_path.is_file():
            continue
            
        # Skip files already in organized subdirectories
        if file_path.parent.name in ["images", "annotations", "train", "val", "test"]:
            continue
            
        file_ext = file_path.suffix.lower()
        file_name = file_path.name.lower()
        
        # Determine target directory
        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            target_dir = dataset_dir / "images"
        elif file_ext in ['.xml', '.json', '.txt', '.csv']:
            target_dir = dataset_dir / "annotations"  
        elif any(keyword in file_name for keyword in ['train', 'training']):
            target_dir = dataset_dir / "train"
        elif any(keyword in file_name for keyword in ['val', 'valid', 'validation']):
            target_dir = dataset_dir / "val"
        elif any(keyword in file_name for keyword in ['test', 'testing']):
            target_dir = dataset_dir / "test"
        else:
            continue  # Skip organizing this file
        
        # Move file
        target_path = target_dir / file_path.name
        if target_path.exists():
            counter = 1
            while target_path.exists():
                stem = file_path.stem
                target_path = target_dir / f"{stem}_{counter}{file_path.suffix}"
                counter += 1
        
        try:
            file_path.rename(target_path)
        except Exception as e:
            print(f"Warning: Could not move {file_path.name}: {e}")

def show_info():
    """Display dataset information and instructions."""
    print("\n🔥 Fire Detection Datasets for TUBITAK-1002")
    print("=" * 50)
    
    for dataset_id, info in DATASETS.items():
        print(f"\n📊 Dataset {dataset_id}: {info['description']}")
        print(f"   URL: {info['url']}")
        print(f"   Expected file: {info['filename']}")
        
        dataset_dir = Path("datasets") / info["name"]
        if dataset_dir.exists():
            files = list(dataset_dir.rglob("*"))
            file_count = len([f for f in files if f.is_file()])
            print(f"   Status: Directory exists ({file_count} files)")
        else:
            print(f"   Status: Not set up")
    
    print(f"\n📋 Instructions:")
    print("   1. Run: python download_dataset.py --setup")
    print("   2. Manually download files from the URLs above")
    print("   3. Place files in respective dataset directories")
    print("   4. Run: python download_dataset.py --extract")

def clean_up():
    """Remove downloaded archive files after extraction."""
    import shutil
    
    base_dir = Path("datasets")
    if not base_dir.exists():
        print("❌ No datasets directory found.")
        return
    
    removed_count = 0
    for archive_path in base_dir.rglob("*.zip"):
        archive_path.unlink()
        print(f"🗑️ Removed: {archive_path}")
        removed_count += 1
    
    for archive_path in base_dir.rglob("*.tar*"):
        archive_path.unlink()  
        print(f"🗑️ Removed: {archive_path}")
        removed_count += 1
    
    if removed_count == 0:
        print("✨ No archive files to clean up.")
    else:
        print(f"✅ Cleaned up {removed_count} archive files.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        show_info()
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "--setup":
        create_dataset_structure()
    elif command == "--extract":
        extract_datasets()
    elif command == "--info":
        show_info()
    elif command == "--clean":
        clean_up()
    else:
        print("❌ Unknown command. Use: --setup, --extract, --info, or --clean")
        show_info()