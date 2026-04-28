"""
Quick test script to verify dataset loading before training
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config

def test_datasets():
    """Test loading all datasets"""
    config = Config()
    
    print("="*70)
    print("TESTING DATASET LOADING")
    print("="*70)
    
    # Test if multi-dataset mode is enabled
    print(f"\nMulti-dataset mode: {config.USE_MULTI_DATASETS}")
    
    if config.USE_MULTI_DATASETS:
        print(f"\nDataset sources configured:")
        for name, spec in config.DATASET_SOURCES.items():
            root = Path(spec['root'])
            exists = root.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {name}: {root}")
            if exists:
                img_dir = root / 'images'
                ann_dir = root / 'annotations'
                print(f"    - Images: {img_dir.exists()}")
                print(f"    - Annotations: {ann_dir.exists()}")
                if ann_dir.exists():
                    coco_dirs = list(ann_dir.glob('COCO_*'))
                    if coco_dirs:
                        print(f"    - COCO format: {coco_dirs[0].name}")
    
    print("\n" + "="*70)
    print("Testing dataloader creation...")
    print("="*70)
    
    try:
        from swin_transformer_pytorch.data_utils import create_multi_source_dataloaders
        
        train_loader, val_loader, test_loader, per_dataset_tests = create_multi_source_dataloaders(
            dataset_sources=config.DATASET_SOURCES,
            train_sources=config.TRAIN_DATASETS,
            val_sources=config.VAL_DATASETS,
            test_sources=config.TEST_DATASETS,
            batch_size=4,
            num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
            img_size=config.IMG_SIZE,
            annotation_format=config.ANNOTATION_FORMAT,
            task=config.TASK,
            pin_memory=False,
            cache_images=False
        )
        
        print(f"\n✓ Dataloaders created successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Combined test batches: {len(test_loader)}")
        
        if per_dataset_tests:
            print(f"\nPer-dataset test loaders:")
            for name, loader in per_dataset_tests.items():
                print(f"  - {name}: {len(loader)} batches")
        
        # Try to get one batch
        print("\nTesting batch retrieval...")
        images, labels = next(iter(train_loader))
        print(f"✓ Batch shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"  Sample labels: {labels[:5]}")
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nYou can now run: python train.py")
        
    except Exception as e:
        print(f"\n✗ Error during testing:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_datasets()
    sys.exit(0 if success else 1)
