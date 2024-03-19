python  thirdparty/yolov9/val_dual.py \
    --data data/taco.yaml \
    --img 640 \
    --batch 32 \
    --device 0 \
    --name yolov9-c-taco-from-scratch \
    --weights /home/vladimir/yolo9/runs/train/yolov9-c-taco-from-scratch/weights/best.pt \
    --conf 0.001 \
    --iou 0.7