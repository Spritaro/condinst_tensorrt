import albumentations as A
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader

import torchvision

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning import LitCenterNet
from dataset import CocoSegmentationAlb

parser = argparse.ArgumentParser(description="Parameters for training and inference")
parser.add_argument('command', type=str, help="train or test")

# Dataset options
parser.add_argument('--train_dir', type=str, default=None, help="path to train image dir (required for training)")
parser.add_argument('--train_ann', type=str, default=None, help="path to train annotation path (required for training)")
parser.add_argument('--val_dir', type=str, default=None, required=False, help="path to validation image dir (optional)")
parser.add_argument('--val_ann', type=str, default=None, required=False, help="path to validation dataset dir (optional)")
parser.add_argument('--num_classes', type=int, default=80, help="number of classes (default 80)")

# Train options
parser.add_argument('--pretrained_model', type=str, default=None, help="path to pretrained model (optional)")
parser.add_argument('--input_width', type=int, default=640, required=False, help="width of input image (default 640)")
parser.add_argument('--input_height', type=int, default=480, help="height of input image (default 480)")
parser.add_argument('--batch_size', type=int, default=8, help="batch size (default 8)")
parser.add_argument('--accumulate_grad_batches', type=int, default=16, help="number of gradients to accumulate (default 16)")
parser.add_argument('--num_workers', type=int, default=4, help="number of workers for data loader (default 4)")
parser.add_argument('--mixed_precision', type=bool, default=True, help="allow FP16 training (default True)")
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="path to checkpoint file (optional)")
parser.add_argument('--max_epochs', type=int, default=10, help="number of epochs (default 10")
parser.add_argument('--gpus', type=int, default=1, help="number of GPUs to train (0 for CPU, -1 for all GPUs) (default 1)")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="learning rate (default 1e-4)")

# Logging options
parser.add_argument('--tensorboard_log_dir', type=str, default='../runs', help="path to TensorBoard log dir (default '../runs')")
parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help="path to checkpoint directory (default '../checkpoints')")

# Output options
parser.add_argument('--save_model', type=str, default='../models/model.pt', help="path to save trained model (defalut '../models/model.py')")

# Test options
parser.add_argument('--topk', type=int, default=40, help="max number of object to detect during inference (default 40)")
parser.add_argument('--load_model', type=str, default='../models/model.pt', help="path to load trained model (default '../models/model.py')")
parser.add_argument('--export_onnx', type=str, default='../models/model.onnx', help="path to export as onnx model (default ../models/model.onnx')")
parser.add_argument('--test_image_dir', type=str, default='../test_image', help="path to test image dir (default '../test_image')")
parser.add_argument('--test_output_dir', type=str, default='../test_output', help="path to test output dir (default '../test_output')")

args = parser.parse_args()
assert args.command == 'train' or args.command == 'test'

if __name__ == '__main__':

    if args.command == 'train':
        assert args.train_dir is not None and args.train_ann is not None

        # Transform
        transform = A.Compose([
            A.RandomScale(scale_limit=0.5, interpolation=cv2.INTER_LINEAR, p=0.5),
            A.transforms.PadIfNeeded(min_width=args.input_width, min_height=args.input_height, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomCrop(width=args.input_width, height=args.input_height),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        # Load data
        root_dir_train = os.path.expanduser(args.train_dir)
        ann_path_train = os.path.expanduser(args.train_ann)
        dataset_train = CocoSegmentationAlb(root=root_dir_train, annFile=ann_path_train, transform=transform)
        coco_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=lambda x: x, num_workers=args.num_workers)

        if args.val_dir and args.val_ann:
            root_dir_val = os.path.expanduser(args.val_dir)
            ann_path_val = os.path.expanduser(args.val_ann)
            dataset_val = CocoSegmentationAlb(root=root_dir_val, annFile=ann_path_val, transform=transform)
            coco_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=lambda x: x, num_workers=args.num_workers)
        else:
            coco_val = None

        # Save checkpoints
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='{epoch:03d}-{step:06d}',
            save_last=True
        )

        # Create model for training
        model = LitCenterNet(mode='training', num_classes=args.num_classes, learning_rate=args.learning_rate)

        # Load pretrained weights
        if args.pretrained_model:
            pretrained_dict = torch.load(args.pretrained_model)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and v.shape == model_dict[k].shape)}
            model.load_state_dict(pretrained_dict, strict=False)

        # Mixed precision
        if args.mixed_precision:
            precision = 16
        else:
            precision = 32

        # Logger
        os.makedirs(args.tensorboard_log_dir, exist_ok=True)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.tensorboard_log_dir, default_hp_metric=False)

        # Train model
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            gpus=args.gpus,
            val_check_interval=1.0, # validate once per epoch
            accumulate_grad_batches=args.accumulate_grad_batches,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=args.resume_from_checkpoint,
            precision=precision,
            logger=tb_logger
        )
        trainer.fit(model, coco_train, coco_val)

        # Save model
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)

    else:

        # Load model for inference
        model = LitCenterNet(mode='inference', num_classes=args.num_classes, topk=args.topk)
        model.load_state_dict(torch.load(args.load_model))

        # Export to ONNX
        input_sample = torch.randn((1, 3, args.input_height, args.input_width))
        if args.mixed_precision:
            # Needs CUDA to support FP16
            model.half().to('cuda')
        model.to_onnx(args.export_onnx, input_sample, export_params=True, opset_version=11)

        # TODO: add test
