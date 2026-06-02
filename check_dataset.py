import json
from collections import defaultdict
from pathlib import Path

# Check all three dataset splits
base_path = Path('datasets/FASDD_RS/annotations/COCO_RS_RGB/Annotations')

for split in ['train', 'val', 'test']:
    json_file = base_path / f'{split}.json'
    print(f"\n{'='*50}")
    print(f"Dataset: {split.upper()}")
    print('='*50)
    
    with open(json_file) as f:
        data = json.load(f)
    
    # Count images by category
    category_counts = defaultdict(int)
    image_categories = {}  # Track which category each image belongs to
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Create image to category mapping
    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_name = categories[ann['category_id']]
        category_counts[cat_name] += 1
        image_categories[img_id] = cat_name
    
    # Count unique images per category
    unique_images_per_cat = defaultdict(set)
    for img_id, cat in image_categories.items():
        unique_images_per_cat[cat].add(img_id)
    
    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")
    print('\nAnnotations by category:')
    for cat in sorted(categories.values()):
        img_count = len(unique_images_per_cat[cat])
        ann_count = category_counts[cat]
        print(f'  {cat}: {img_count} images, {ann_count} annotations')
