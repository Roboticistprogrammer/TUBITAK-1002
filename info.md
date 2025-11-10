DATA:
FASDD_RS dataset is a fire and smoke detection dataset with:

2,223 total images (1000x1000, .tif format, 3 channels RGB)
3 classes: fire, smoke, neitherFireNorSmoke
Split: Train (1,111), Val (740), Test (369)
Annotations: Available in COCO, VOC, YOLO, and TDML formats
Task: Object detection (bounding boxes for fire/smoke)



Possible Modifications:
Option A: Classification approach (image-level)

3 classes: fire, smoke, neither
Use existing Swin architecture as-is
Option B: Object Detection approach (recommended)

Integrate Swin as backbone
Add detection head (Faster R-CNN, RetinaNet, or FCOS)
Consider using frameworks like Detectron2 or MMDetection