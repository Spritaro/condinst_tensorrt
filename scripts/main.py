import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning import LitCenterNet
from dataset import CocoSegmentationAlb


if __name__ == '__main__':
    # Transform
    transform = A.Compose([
        A.RandomScale(scale_limit=0.5, interpolation=1, p=0.5),
        A.transforms.PadIfNeeded(min_height=480, min_width=640, border_mode=cv2.BORDER_CONSTANT, value=0),
        # A.transforms.PadIfNeeded(min_height=480, min_width=640),
        A.RandomCrop(width=640, height=480),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Load data
    root_dir_train = os.path.expanduser('~/dataset/train2017')
    root_dir_val = os.path.expanduser('~/dataset/val2017')
    ann_path_train = os.path.expanduser('~/dataset/annotations_trainval2017/annotations/instances_train2017.json')
    ann_path_val = os.path.expanduser('~/dataset/annotations_trainval2017/annotations/instances_val2017.json')

    dataset_train = CocoSegmentationAlb(root=root_dir_train, annFile=ann_path_train, transform=transform)
    # dataset_train = CocoSegmentationAlb(root=root_dir_val, annFile=ann_path_val, transform=transform)
    dataset_val = CocoSegmentationAlb(root=root_dir_val, annFile=ann_path_val, transform=transform)

    coco_train = DataLoader(dataset_train, batch_size=4, collate_fn=lambda x: x)
    # coco_train = DataLoader(dataset_val, batch_size=4, collate_fn=lambda x: x)
    coco_val = DataLoader(dataset_val, batch_size=4, collate_fn=lambda x: x)

    # Check dataset
    index = 0
    batch = dataset_train.__getitem__(index)
    image, target = batch
    plt.imshow(image.permute(1,2,0))
    plt.savefig('images/train.png')
    mask = target[0]['segmentation']
    plt.imshow(mask)
    plt.savefig('images/masks.png')

    # Save checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='centernet-condinst-{epoch:02d}-{step:06d}-{val_loss:.2f}',
    )

    # Create model for training
    model = LitCenterNet(mode='training', num_classes=81)

    # model = LitCenterNet.load_from_checkpoint(
    #     'checkpoints/centernet-condinst-epoch=00-val_loss=0.00.ckpt',
    #     training=True, num_classes=81)

    # filepath = 'centernet.pt'
    # model.load_state_dict(torch.load(filepath)) # Load model

    # Train model
    trainer = pl.Trainer(
        # max_epochs=1,
        max_epochs=6,
        gpus=1,
        val_check_interval=10000,
        accumulate_grad_batches=30,
        callbacks=[checkpoint_callback],
        # resume_from_checkpoint='checkpoints/centernet-condinst-epoch=00-val_loss=0.00.ckpt'
    )
    trainer.fit(model, coco_train, coco_val)
    # trainer.fit(model, coco_train)

    # Save model
    filepath = 'centernet.pt'
    torch.save(model.state_dict(), filepath)


