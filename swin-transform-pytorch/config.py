"""
Configuration file for FASDD_RS Fire and Smoke Detection Training
"""

import os
from pathlib import Path


class Config:
    """Training configuration"""
    
    # ============== Paths ==============
    ROOT_DIR = Path(__file__).parent
    DATASETS_BASE_DIR = ROOT_DIR.parent / "datasets"
    DATASET_ROOT = DATASETS_BASE_DIR / "FASDD_RS"
    CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
    LOGS_DIR = ROOT_DIR / "logs"
    RESULTS_DIR = ROOT_DIR / "results"

    # Multi-source dataset support
    USE_MULTI_DATASETS = True
    DATASET_SOURCES = {
        'FASDD_RS': {
            'root': DATASETS_BASE_DIR / 'FASDD_RS',
            'annotation_format': 'coco',
            'task': 'classification'
        },
        'FASDD_CV': {
            'root': DATASETS_BASE_DIR / 'FASDD_CV' / 'FASDD_CV',
            'annotation_format': 'coco',
            'task': 'classification'
        },
        'FASDD_UAV': {
            'root': DATASETS_BASE_DIR / 'FASDD_UAV',
            'annotation_format': 'coco',
            'task': 'classification'
        }
    }
    TRAIN_DATASETS = ['FASDD_RS', 'FASDD_CV', 'FASDD_UAV']
    VAL_DATASETS = ['FASDD_RS', 'FASDD_CV', 'FASDD_UAV']
    TEST_DATASETS = ['FASDD_RS', 'FASDD_CV', 'FASDD_UAV']
    COMBINED_TEST_NAME = 'Combined'
    
    # ============== Dataset ==============
    ANNOTATION_FORMAT = 'coco'  # 'coco', 'voc', or 'yolo'
    TASK = 'classification'  # 'classification' or 'detection'
    NUM_CLASSES = 3  # fire, smoke, neitherFireNorSmoke
    CLASS_NAMES = ['fire', 'smoke', 'neitherFireNorSmoke']
    
    # ============== Data Loading ==============
    BATCH_SIZE = 16
    NUM_WORKERS = 8
    PIN_MEMORY = True
    CACHE_IMAGES = False  # Set to True if you have enough RAM
    IMG_SIZE = 224  # 224, 384, or 512
    
    # ============== Model Architecture ==============
    # Swin Transformer variants: 'tiny', 'small', 'base', 'large', or 'custom'
    MODEL_VARIANT = 'base'
    
    # Swin-T (Tiny) configuration
    # HIDDEN_DIM = 96
    # LAYERS = (2, 2, 6, 2)
    # HEADS = (3, 6, 12, 24)
    # HEAD_DIM = 32
    # WINDOW_SIZE = 7
    # DOWNSCALING_FACTORS = (4, 2, 2, 2)
    # RELATIVE_POS_EMBEDDING = True
    # CHANNELS = 3
    
    # For other variants, uncomment the desired configuration:
    
    # # Swin-S (Small)
    # HIDDEN_DIM = 96
    # LAYERS = (2, 2, 18, 2)
    # HEADS = (3, 6, 12, 24)
    
    # # Swin-B (Base)
    HIDDEN_DIM = 128
    LAYERS = (2, 2, 18, 2)
    HEADS = (4, 8, 16, 32)
    HEAD_DIM = 32
    WINDOW_SIZE = 7
    DOWNSCALING_FACTORS = (4, 2, 2, 2)
    RELATIVE_POS_EMBEDDING = True
    CHANNELS = 3
    
    # # Swin-L (Large)
    # HIDDEN_DIM = 192
    # LAYERS = (2, 2, 18, 2)
    # HEADS = (6, 12, 24, 48)
    
    # ============== Training Hyperparameters ==============
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    
    # Learning rate scheduler
    SCHEDULER = 'cosine'  # 'cosine', 'step', or 'none'
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # Step scheduler (if SCHEDULER='step')
    STEP_SIZE = 30
    GAMMA = 0.1
    
    # ============== Optimizer ==============
    OPTIMIZER = 'adamw'  # 'adamw', 'adam', or 'sgd'
    MOMENTUM = 0.9  # For SGD
    BETAS = (0.9, 0.999)  # For Adam/AdamW
    
    # ============== Regularization ==============
    DROPOUT = 0.0
    LABEL_SMOOTHING = 0.1
    GRADIENT_CLIP = 1.0  # Set to None to disable
    
    # ============== Training Strategy ==============
    MIXED_PRECISION = True  # Use automatic mixed precision
    COMPILE_MODEL = False  # Use torch.compile (PyTorch 2.0+)
    
    # ============== Checkpoint & Logging ==============
    SAVE_EVERY = 5  # Save checkpoint every N epochs
    SAVE_BEST_ONLY = True
    EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for N epochs
    
    LOG_INTERVAL = 10  # Log every N batches
    EVAL_INTERVAL = 1  # Evaluate every N epochs
    
    # ============== Resume Training ==============
    RESUME = False
    RESUME_CHECKPOINT = None  # Path to checkpoint file
    
    # ============== Pretrained Weights ==============
    PRETRAINED = False
    PRETRAINED_PATH = None  # Path to pretrained weights
    FREEZE_BACKBONE = False  # Freeze backbone layers initially
    UNFREEZE_AFTER_EPOCHS = 0  # Unfreeze backbone after N epochs
    
    # ============== Device ==============
    DEVICE = 'cuda'  # 'cuda' or 'cpu'
    USE_CUDNN_BENCHMARK = True  # Auto-tune CUDNN kernels for faster training
    
    # ============== Seed ==============
    SEED = 42
    
    # ============== Weights & Biases ==============
    USE_WANDB = False
    WANDB_PROJECT = "fasdd-fire-smoke-detection"
    WANDB_ENTITY = None  # Your wandb username
    WANDB_RUN_NAME = None  # Auto-generated if None
    
    # ============== TensorBoard ==============
    USE_TENSORBOARD = True
    
    # ============== Class Weights ==============
    USE_CLASS_WEIGHTS = False  # Balance classes if dataset is imbalanced
    CLASS_WEIGHTS = None  # Will be computed automatically if USE_CLASS_WEIGHTS=True
    
    # ============== Validation ==============
    VAL_METRIC = 'accuracy'  # 'accuracy' or 'loss'
    VAL_MODE = 'max'  # 'max' for accuracy, 'min' for loss
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def display(cls):
        """Display configuration"""
        print("\n" + "="*70)
        print("TRAINING CONFIGURATION")
        print("="*70)
        
        print("\n[DATASET]")
        print(f"  Dataset Root: {cls.DATASET_ROOT}")
        print(f"  Task: {cls.TASK}")
        print(f"  Number of Classes: {cls.NUM_CLASSES}")
        print(f"  Annotation Format: {cls.ANNOTATION_FORMAT}")
        print(f"  Image Size: {cls.IMG_SIZE}")
        if cls.USE_MULTI_DATASETS:
            print(f"  Using Combined Sources: {', '.join(cls.TRAIN_DATASETS)}")
            for name, spec in cls.DATASET_SOURCES.items():
                root = Path(spec['root'])
                print(f"    - {name}: {root}")
        
        print("\n[MODEL]")
        print(f"  Variant: {cls.MODEL_VARIANT}")
        print(f"  Hidden Dim: {cls.HIDDEN_DIM}")
        print(f"  Layers: {cls.LAYERS}")
        print(f"  Heads: {cls.HEADS}")
        print(f"  Window Size: {cls.WINDOW_SIZE}")
        
        print("\n[TRAINING]")
        print(f"  Epochs: {cls.EPOCHS}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"  Optimizer: {cls.OPTIMIZER}")
        print(f"  Scheduler: {cls.SCHEDULER}")
        print(f"  Mixed Precision: {cls.MIXED_PRECISION}")
        
        print("\n[PATHS]")
        print(f"  Checkpoints: {cls.CHECKPOINT_DIR}")
        print(f"  Logs: {cls.LOGS_DIR}")
        print(f"  Results: {cls.RESULTS_DIR}")
        
        print("\n" + "="*70 + "\n")


# Alternative: Function-based config for easy override
def get_config(**kwargs):
    """Get configuration with optional overrides"""
    config = Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config
