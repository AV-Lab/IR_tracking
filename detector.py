import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import sys
sys.path.append('yolov7/')


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Detector:


    def __init__(self, weights, min_confidence, device):
        self.weights = weights
        self.min_confidence = min_confidence
        self.device = select_device('')
        self.iou_threshold = 0.45 #IOU threshold for NMS

        with torch.no_grad():
            self.model = attempt_load([self.weights], map_location=self.device)  # load FP32 model

    def detect(self, source, imgsz):
        
        with torch.no_grad():
            
            stride = int(self.model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
                
            # Set Dataloader
            vid_path, vid_writer = None, None
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        
            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            
            # Run inference
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
            old_img_w = old_img_h = imgsz
            old_img_b = 1
            
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(self.device)
                img = img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Warmup
                if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        self.model(img, augment=False)[0]

                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                    pred = self.model(img, augment=False)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.min_confidence, self.iou_threshold, classes=None, agnostic=True)
                results = []
                
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)        
                    p = Path(p)  # to Path
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        
                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh)  # label format
                            with open('detctions.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
    
                            label = f'{names[int(cls)]} {conf:.2f}'
                            results.append((xyxy, names[int(cls)], conf))
                            
                return results
        