import cv2
import copy
import os.path
from torchvision.datasets import CocoDetection

class CocoSegmentationAlb(CocoDetection):
    def __init__(self, root, annFile, transform):
        super().__init__(root=root, annFile=annFile, transform=transform)

        categories = self.coco.dataset['categories']
        self.category_id_to_class_label = {}
        for label, category in enumerate(categories):
            self.category_id_to_class_label[category['id']] = label

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

        return img, target
