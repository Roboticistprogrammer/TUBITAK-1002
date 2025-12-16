"""
Optimized data utilities for FASDD_RS Fire and Smoke Detection Dataset
Supports COCO, VOC, and YOLO annotation formats with efficient loading
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np


class FASSDDataset(Dataset):
    """
    FASDD_RS Fire and Smoke Detection Dataset
    Supports multiple annotation formats: COCO, VOC, YOLO
    """
    
    # Class mapping
    CLASS_NAMES = ['fire', 'smoke', 'neitherFireNorSmoke']
    CLASS_TO_IDX = {'fire': 0, 'smoke': 1, 'neitherFireNorSmoke': 2}
    IDX_TO_CLASS = {0: 'fire', 1: 'smoke', 2: 'neitherFireNorSmoke'}
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        annotation_format: str = 'coco',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        task: str = 'detection',  # 'detection' or 'classification'
        return_raw_image: bool = False,
        cache_images: bool = False
    ):
        """
        Args:
            root_dir: Root directory of FASDD_RS dataset
            split: 'train', 'val', or 'test'
            annotation_format: 'coco', 'voc', or 'yolo'
            transform: Image transformations
            target_transform: Target transformations
            task: 'detection' (bboxes) or 'classification' (image-level labels)
            return_raw_image: Return original image along with transformed
            cache_images: Cache images in memory (faster but uses more RAM)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.annotation_format = annotation_format.lower()
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.return_raw_image = return_raw_image
        self.cache_images = cache_images
        
        # Paths
        self.image_dir = self.root_dir / 'images'
        self.annotation_dir = self.root_dir / 'annotations'
        
        # Load annotations based on format
        self.samples = self._load_annotations()
        
        # Cache for images
        self._image_cache = {} if cache_images else None
        
        print(f"Loaded {len(self.samples)} samples for {split} split using {annotation_format} format")
    
    def _load_annotations(self) -> List[Dict]:
        """Load annotations based on specified format"""
        if self.annotation_format == 'coco':
            return self._load_coco_annotations()
        elif self.annotation_format == 'voc':
            return self._load_voc_annotations()
        elif self.annotation_format == 'yolo':
            return self._load_yolo_annotations()
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")
    
    def _load_coco_annotations(self) -> List[Dict]:
        """Load COCO format annotations"""
        # Detect the correct COCO subfolder (COCO_RS_RGB, COCO_CV, COCO_UAV, etc.)
        coco_subdirs = list(self.annotation_dir.glob('COCO_*'))
        if not coco_subdirs:
            raise FileNotFoundError(f"No COCO annotation directory found in {self.annotation_dir}")
        coco_dir = coco_subdirs[0]
        
        # Check for Annotations subfolder or direct JSON files
        if (coco_dir / 'Annotations').exists():
            json_file = coco_dir / 'Annotations' / f'{self.split}.json'
        else:
            json_file = coco_dir / f'{self.split}.json'
        
        if not json_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image id to filename mapping
        images_info = {img['id']: img for img in coco_data['images']}
        
        # Build category mapping
        categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image
        image_annotations = defaultdict(list)
        for ann in coco_data['annotations']:
            image_annotations[ann['image_id']].append(ann)
        
        samples = []
        for img_id, img_info in images_info.items():
            image_path = self.image_dir / img_info['file_name']
            
            # Get annotations for this image
            anns = image_annotations.get(img_id, [])
            
            # Extract bounding boxes and labels
            boxes = []
            labels = []
            areas = []
            
            for ann in anns:
                # COCO bbox format: [x, y, width, height]
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]
                labels.append(categories[ann['category_id']])
                areas.append(ann['area'])
            
            # For classification task, get image-level label
            if self.task == 'classification':
                if labels:
                    # Use the most prominent object (largest area)
                    if areas:
                        max_idx = np.argmax(areas)
                        image_label = labels[max_idx]
                    else:
                        image_label = labels[0]
                else:
                    image_label = 'neitherFireNorSmoke'
            else:
                image_label = None
            
            sample = {
                'image_path': str(image_path),
                'image_id': img_id,
                'width': img_info['width'],
                'height': img_info['height'],
                'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
                'labels': labels,
                'areas': areas,
                'image_label': image_label
            }
            samples.append(sample)
        
        return samples
    
    def _load_voc_annotations(self) -> List[Dict]:
        """Load VOC format annotations"""
        # Detect the correct VOC and YOLO subdirectories
        voc_subdirs = list(self.annotation_dir.glob('VOC_*'))
        yolo_subdirs = list(self.annotation_dir.glob('YOLO_*'))
        
        if not voc_subdirs:
            raise FileNotFoundError(f"No VOC annotation directory found in {self.annotation_dir}")
        if not yolo_subdirs:
            raise FileNotFoundError(f"No YOLO annotation directory found in {self.annotation_dir}")
        
        voc_base = voc_subdirs[0]
        yolo_base = yolo_subdirs[0]
        
        # Check for Annotations subfolder
        voc_dir = voc_base / 'Annotations' if (voc_base / 'Annotations').exists() else voc_base
        yolo_file = yolo_base / f'{self.split}.txt'
        
        # Read image list from YOLO split file
        with open(yolo_file, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        
        samples = []
        for img_path in image_paths:
            # Extract filename
            filename = Path(img_path).name
            image_full_path = self.image_dir / filename
            
            # Get corresponding XML file
            xml_filename = filename.replace('.tif', '.xml')
            xml_path = voc_dir / xml_filename
            
            if not xml_path.exists():
                continue
            
            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image size
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # Get all objects
            boxes = []
            labels = []
            areas = []
            
            for obj in root.findall('object'):
                label = obj.find('name').text
                bndbox = obj.find('bndbox')
                
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
                areas.append((xmax - xmin) * (ymax - ymin))
            
            # For classification task
            if self.task == 'classification':
                if labels:
                    if areas:
                        max_idx = np.argmax(areas)
                        image_label = labels[max_idx]
                    else:
                        image_label = labels[0]
                else:
                    image_label = 'neitherFireNorSmoke'
            else:
                image_label = None
            
            sample = {
                'image_path': str(image_full_path),
                'image_id': filename,
                'width': width,
                'height': height,
                'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
                'labels': labels,
                'areas': areas,
                'image_label': image_label
            }
            samples.append(sample)
        
        return samples
    
    def _load_yolo_annotations(self) -> List[Dict]:
        """
        Load using YOLO split files but extract full annotations from VOC
        This is a hybrid approach for efficiency
        """
        return self._load_voc_annotations()
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load image with caching support"""
        if self._image_cache is not None:
            if image_path not in self._image_cache:
                self._image_cache[image_path] = Image.open(image_path).convert('RGB')
            return self._image_cache[image_path].copy()
        else:
            return Image.open(image_path).convert('RGB')
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple, Dict]:
        """
        Returns:
            For classification: (image, label) or dict
            For detection: dict with image, boxes, labels
        """
        sample = self.samples[idx]
        
        # Load image
        image = self._load_image(sample['image_path'])
        raw_image = image.copy() if self.return_raw_image else None
        
        if self.task == 'classification':
            # Classification task: return image and label
            label = self.CLASS_TO_IDX.get(sample['image_label'], 2)
            
            if self.transform:
                image = self.transform(image)
            
            if self.return_raw_image:
                return {'image': image, 'label': label, 'raw_image': raw_image, 'image_path': sample['image_path']}
            return image, label
        
        else:
            # Detection task: return image with bounding boxes
            boxes = sample['boxes'].copy()
            labels = [self.CLASS_TO_IDX.get(lbl, 0) for lbl in sample['labels']]
            
            # Apply transforms
            if self.transform:
                # For detection, transform needs to handle boxes too
                transformed = self.transform(image=np.array(image), bboxes=boxes, labels=labels)
                image = transformed['image']
                boxes = np.array(transformed.get('bboxes', boxes), dtype=np.float32)
                labels = transformed.get('labels', labels)
            
            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': torch.as_tensor(sample['areas'], dtype=torch.float32) if sample['areas'] else torch.tensor([]),
                'iscrowd': torch.zeros((len(labels),), dtype=torch.int64)
            }
            
            if self.return_raw_image:
                return {
                    'image': image,
                    'target': target,
                    'raw_image': raw_image,
                    'image_path': sample['image_path']
                }
            
            return image, target
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        distribution = defaultdict(int)
        
        for sample in self.samples:
            if self.task == 'classification':
                distribution[sample['image_label']] += 1
            else:
                for label in sample['labels']:
                    distribution[label] += 1
        
        return dict(distribution)
    
    def visualize_sample(self, idx: int, save_path: Optional[str] = None):
        """Visualize a sample with bounding boxes"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        sample = self.samples[idx]
        image = self._load_image(sample['image_path'])
        
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image)
        
        # Draw bounding boxes
        colors = {'fire': 'red', 'smoke': 'yellow', 'neitherFireNorSmoke': 'green'}
        for box, label in zip(sample['boxes'], sample['labels']):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=colors.get(label, 'blue'),
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, label, color=colors.get(label, 'blue'),
                   fontsize=12, fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7))
        
        ax.axis('off')
        plt.title(f"Sample {idx}: {Path(sample['image_path']).name}")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()


def get_transforms(split: str = 'train', img_size: int = 224, task: str = 'classification'):
    """
    Get appropriate transforms for train/val/test splits
    
    Args:
        split: 'train', 'val', or 'test'
        img_size: Target image size
        task: 'classification' or 'detection'
    """
    if task == 'classification':
        # Standard image classification transforms
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    else:
        # Detection transforms (using albumentations for bbox handling)
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            if split == 'train':
                return A.Compose([
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
            else:
                return A.Compose([
                    A.Resize(img_size, img_size),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        except ImportError:
            print("Warning: albumentations not installed. Using basic transforms.")
            return get_transforms(split, img_size, task='classification')


def collate_fn_detection(batch):
    """Custom collate function for detection task"""
    images = []
    targets = []
    
    for item in batch:
        if isinstance(item, dict):
            images.append(item['image'])
            targets.append(item['target'])
        else:
            images.append(item[0])
            targets.append(item[1])
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    return images, targets


def create_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    annotation_format: str = 'coco',
    task: str = 'classification',
    pin_memory: bool = True,
    cache_images: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        root_dir: Root directory of FASDD_RS dataset
        batch_size: Batch size for training
        num_workers: Number of worker processes
        img_size: Target image size
        annotation_format: 'coco', 'voc', or 'yolo'
        task: 'classification' or 'detection'
        pin_memory: Pin memory for faster GPU transfer
        cache_images: Cache images in RAM
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = FASSDDataset(
        root_dir=root_dir,
        split='train',
        annotation_format=annotation_format,
        transform=get_transforms('train', img_size, task),
        task=task,
        cache_images=cache_images
    )
    
    val_dataset = FASSDDataset(
        root_dir=root_dir,
        split='val',
        annotation_format=annotation_format,
        transform=get_transforms('val', img_size, task),
        task=task,
        cache_images=cache_images
    )
    
    test_dataset = FASSDDataset(
        root_dir=root_dir,
        split='test',
        annotation_format=annotation_format,
        transform=get_transforms('test', img_size, task),
        task=task,
        cache_images=cache_images
    )
    
    # Select collate function
    collate_fn = collate_fn_detection if task == 'detection' else None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def create_multi_source_dataloaders(
    dataset_sources: Dict[str, Dict],
    train_sources: List[str],
    val_sources: List[str],
    test_sources: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
    annotation_format: str = 'coco',
    task: str = 'classification',
    pin_memory: bool = True,
    cache_images: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, DataLoader]]:
    """Create dataloaders by combining multiple dataset sources."""

    if not dataset_sources:
        raise ValueError("dataset_sources cannot be empty when using multi-source loading.")

    def _get_spec(name: str) -> Dict:
        if name not in dataset_sources:
            raise KeyError(f"Dataset source '{name}' is not defined in dataset_sources.")
        spec = dataset_sources[name]
        if 'root' not in spec:
            raise KeyError(f"Dataset source '{name}' must define a 'root' path.")
        return spec

    active_names = set(train_sources + val_sources + test_sources)
    active_tasks = { _get_spec(name).get('task', task) for name in active_names }
    if len(active_tasks) > 1:
        raise ValueError("All dataset sources must share the same task type for multi-source loading.")
    active_task = active_tasks.pop() if active_tasks else task
    collate_fn = collate_fn_detection if active_task == 'detection' else None

    def _build_dataset(name: str, split: str) -> FASSDDataset:
        spec = _get_spec(name)
        fmt = spec.get('annotation_format', annotation_format)
        ds_task = spec.get('task', active_task)
        ds_cache = spec.get('cache_images', cache_images)
        return FASSDDataset(
            root_dir=str(spec['root']),
            split=split,
            annotation_format=fmt,
            transform=get_transforms(split, img_size, ds_task),
            task=ds_task,
            cache_images=ds_cache
        )

    def _build_pairs(source_names: List[str], split: str) -> List[Tuple[str, FASSDDataset]]:
        return [(name, _build_dataset(name, split)) for name in source_names]

    def _make_loader(pairs: List[Tuple[str, FASSDDataset]], split: str, shuffle: bool) -> DataLoader:
        datasets = [ds for _, ds in pairs]
        if not datasets:
            raise ValueError(f"No datasets found for split '{split}'.")
        combined_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
        drop_last = (split == 'train')
        return DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            drop_last=drop_last
        )

    train_pairs = _build_pairs(train_sources, 'train')
    val_pairs = _build_pairs(val_sources, 'val')
    test_pairs = _build_pairs(test_sources, 'test')

    print("\nCombined dataset breakdown:")
    for split_name, pairs in [('Train', train_pairs), ('Val', val_pairs), ('Test', test_pairs)]:
        for name, dataset in pairs:
            print(f"  {split_name} - {name}: {len(dataset)} samples")

    train_loader = _make_loader(train_pairs, 'train', shuffle=True)
    val_loader = _make_loader(val_pairs, 'val', shuffle=False)
    combined_test_loader = _make_loader(test_pairs, 'test', shuffle=False)

    per_dataset_tests = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn
        )
        for name, dataset in test_pairs
    }

    return train_loader, val_loader, combined_test_loader, per_dataset_tests


def get_dataset_stats(root_dir: str, annotation_format: str = 'coco'):
    """
    Get comprehensive statistics about the dataset
    """
    print("=" * 60)
    print("FASDD_RS Dataset Statistics")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        dataset = FASSDDataset(
            root_dir=root_dir,
            split=split,
            annotation_format=annotation_format,
            task='detection'
        )
        
        print(f"\n{split.upper()} Split:")
        print(f"  Total images: {len(dataset)}")
        
        # Class distribution
        distribution = dataset.get_class_distribution()
        print(f"  Class distribution:")
        for cls, count in sorted(distribution.items()):
            print(f"    {cls}: {count}")
        
        # Average boxes per image
        total_boxes = sum(len(s['boxes']) for s in dataset.samples)
        avg_boxes = total_boxes / len(dataset) if len(dataset) > 0 else 0
        print(f"  Average boxes per image: {avg_boxes:.2f}")
        
        # Images with no annotations
        no_annotation = sum(1 for s in dataset.samples if len(s['boxes']) == 0)
        print(f"  Images with no annotations: {no_annotation}")
    
    print("\n" + "=" * 60)


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    root_dir = "/mnt/storage/Dataset/FASDD_RS"
    
    print("Testing FASDD Dataset Loader...\n")
    
    # Get dataset statistics
    get_dataset_stats(root_dir, annotation_format='coco')
    
    # Test classification task
    print("\n" + "="*60)
    print("Testing Classification Task")
    print("="*60)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=root_dir,
        batch_size=8,
        num_workers=2,
        img_size=224,
        annotation_format='coco',
        task='classification'
    )
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label examples: {labels[:5]}")
    
    # Test detection task
    print("\n" + "="*60)
    print("Testing Detection Task")
    print("="*60)
    
    try:
        train_loader_det, val_loader_det, test_loader_det = create_dataloaders(
            root_dir=root_dir,
            batch_size=4,
            num_workers=2,
            img_size=224,
            annotation_format='coco',
            task='detection'
        )
        
        images, targets = next(iter(train_loader_det))
        print(f"Images shape: {images.shape}")
        print(f"Number of targets: {len(targets)}")
        print(f"First target keys: {targets[0].keys()}")
        print(f"First target boxes shape: {targets[0]['boxes'].shape}")
        print(f"First target labels: {targets[0]['labels']}")
    except Exception as e:
        print(f"Detection test failed (might need albumentations): {e}")
    
    # Visualize a sample
    print("\n" + "="*60)
    print("Visualizing sample...")
    print("="*60)
    
    dataset = FASSDDataset(
        root_dir=root_dir,
        split='train',
        annotation_format='coco',
        task='detection'
    )
    
    # Uncomment to visualize
    dataset.visualize_sample(0, save_path='sample_visualization.png')
    
    print("\nDataset utilities test completed successfully!")
