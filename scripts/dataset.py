import cv2
import copy
import numpy as np
import os.path

import torch
from torchvision.datasets import CocoDetection

class CocoSegmentationAlb(CocoDetection):
    def __init__(self, root, annFile, depthDir, transform):
        super().__init__(root=root, annFile=annFile, transform=transform)

        categories = self.coco.dataset['categories']
        self.category_id_to_class_label = {}
        for label, category in enumerate(categories):
            self.category_id_to_class_label[category['id']] = label

        # Depth image directory for RGB-D dataset
        self.depth_dir = depthDir

    def __getitem__(self, index):

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        # Read image with OpenCV
        path = coco.loadImgs(img_id)[0]['file_name']
        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert annotations to binary masks
        masks = []
        for t in target:
            mask = self.coco.annToMask(t)
            masks.append(mask)

        # Append depth channel to masks to apply transform
        if self.depth_dir is not None:
            basename = os.path.basename(path)
            basename = os.path.splitext(basename)[0] + '.png'
            path = os.path.join(self.depth_dir, basename)
            depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / 1000 # mm to m
            masks.append(depth)

        if self.transform is not None:
            # Get class labels
            labels = [self.category_id_to_class_label[t['category_id']] for t in target]

            # Apply transform
            transformed = self.transform(image=img, masks=masks, class_labels=labels)

            # Get transformed results
            labels = transformed['class_labels']
            img = transformed['image']
            masks = transformed['masks']

            # Create new targets
            target = []
            for label, mask in zip(labels, masks):
                t = {
                    'class_labels': label,
                    'segmentation': mask
                }
                target.append(t)

        # Extract depth channel from masks and concatenate to image
        if self.depth_dir is not None:
            _, h, w = img.shape
            depth = masks.pop(-1)
            img = torch.cat([img, torch.unsqueeze(depth, 0)], dim=0)

        return img, target
