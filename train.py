EPOCHS = 300
MOSAIC = 0.8
OPTIMIZER = 'AdamW'
MOMENTUM = 0.95
LR0 = 0.001
LRF = 0.0001
SINGLE_CLS = False
BATCH_SIZE = 16
IMG_SIZE = 1024  # Higher resolution for better accuracy

import argparse
from ultralytics import YOLO
import os
import sys
import torch
import time

# Enable CUDA optimizations
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True
if hasattr(torch.backends, 'cuda'):
    torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    # mosaic
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    # optimizer
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    # momentum
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    # lr0
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    # lrf
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    # single_cls
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    # batch size
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size')
    # image size
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='Image size')
    # model
    parser.add_argument('--model', type=str, default='yolov8x.pt', help='Model to use (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)')
    # device
    parser.add_argument('--device', type=str, default='', help='Device to use (empty for auto, cpu, 0, 1, etc.)')
    
    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    print("Starting high-accuracy YOLOv8 training...")
    start_time = time.time()
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {'CUDA' if torch.cuda.is_available() and device != 'cpu' else 'CPU'}")
    if torch.cuda.is_available() and device != 'cpu':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Download and use YOLOv8x model for higher accuracy
    if not os.path.exists(os.path.join(this_dir, args.model)):
        print(f"Downloading {args.model}...")
    
    # Load the model
    model = YOLO(os.path.join(this_dir, args.model))
    print(f"Model loaded: {args.model}")
    
    # Train with optimized parameters for maximum accuracy
    print(f"Training with {args.epochs} epochs, image size {args.imgsz}...")
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=args.epochs,
        device=device,
        single_cls=args.single_cls, 
        mosaic=args.mosaic,
        optimizer=args.optimizer, 
        lr0=args.lr0, 
        lrf=args.lrf, 
        momentum=args.momentum,
        batch=args.batch,
        imgsz=args.imgsz,
        amp=True,                 # Automatic mixed precision
        cos_lr=True,              # Use cosine learning rate
        warmup_epochs=10,         # Longer warmup period
        weight_decay=0.0005,      # L2 regularization
        
        # Loss gains
        box=7.5,                  # Box loss gain
        cls=0.6,                  # Cls loss gain
        dfl=1.5,                  # DFL loss gain
        
        # Advanced augmentations
        close_mosaic=100,         # Disable mosaic later in training
        mixup=0.3,                # Image mixup
        copy_paste=0.3,           # Copy-paste augmentation
        degrees=10.0,             # Random rotation
        translate=0.2,            # Random translation
        scale=0.5,                # Random scaling
        shear=0.5,                # Random shear
        perspective=0.0005,       # Random perspective
        flipud=0.1,               # Random flip up-down
        fliplr=0.5,               # Random flip left-right
        hsv_h=0.015,              # HSV-Hue augmentation
        hsv_s=0.7,                # HSV-Saturation augmentation
        hsv_v=0.4,                # HSV-Value augmentation
        
        # Training efficiency
        cache='ram',              # Cache images in RAM
        workers=8,                # Number of workers
        
        # Mask parameters
        overlap_mask=True,        # Masks should overlap during training
        mask_ratio=4,             # Mask downsample ratio
        
        # Validation and saving
        patience=100,             # Early stopping patience
        save=True,                # Save checkpoints
        save_period=10,           # Save checkpoint every X epochs
        project='runs/train',
        name='high_accuracy_model',
        exist_ok=True,
        plots=True,               # Generate plots for analysis
        val=True,                 # Validate during training
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time/3600:.2f} hours ({elapsed_time/60:.2f} minutes)")
    
    # Path to best model
    best_model_path = os.path.join('runs/train/high_accuracy_model/weights/best.pt')
    print(f"Best model saved at: {best_model_path}")
    
    # Validate the model
    print("\nValidating model on test dataset...")
    try:
        val_results = model.val(data=os.path.join(this_dir, "yolo_params.yaml"), split="test")
        print(f"Validation results:")
        print(f"  mAP@0.5     : {val_results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")
        print(f"  Precision   : {val_results.box.p:.4f}")
        print(f"  Recall      : {val_results.box.r:.4f}")
    except Exception as e:
        print(f"Error during validation: {e}")
    
    print("\nTraining complete. Use predict.py to test the model on new images.")
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''