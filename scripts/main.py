import argparse
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import DataLoader

import torchvision

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from lightning import LitCondInst
from dataset import CocoSegmentationAlb

parser = argparse.ArgumentParser(description="Parameters for training and inference")
parser.add_argument('command', type=str, help="train or test or export")

# Dataset options
parser.add_argument('--train_dir', type=str, default=None, help="path to train image dir (required for training)")
parser.add_argument('--train_ann', type=str, default=None, help="path to train annotation path (required for training)")
parser.add_argument('--val_dir', type=str, default=None, required=False, help="path to validation image dir (optional for training, required for eval)")
parser.add_argument('--val_ann', type=str, default=None, required=False, help="path to validation dataset dir (optional for training, required for eval)")
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
parser.add_argument('--learning_rate', type=float, default=0.01, help="learning rate (default 0.01)")

# Logging options
parser.add_argument('--tensorboard_log_dir', type=str, default='../runs', help="path to TensorBoard log dir (default '../runs')")
parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help="path to checkpoint directory (default '../checkpoints')")

# Output options
parser.add_argument('--save_model', type=str, default='../models/model.pt', help="path to save trained model (defalut '../models/model.py')")

# Test options
parser.add_argument('--topk', type=int, default=40, help="max number of object to detect during inference (default 40)")
parser.add_argument('--score_threshold', type=float, default=0.3, help="score threshold for detection (default 0.3)")
parser.add_argument('--load_model', type=str, default='../models/model.pt', help="path to load trained model (default '../models/model.py')")
parser.add_argument('--test_image_dir', type=str, default='../test_image', help="path to test image dir (default '../test_image')")
parser.add_argument('--test_output_dir', type=str, default='../test_output', help="path to test output dir (default '../test_output')")

# Export options
parser.add_argument('--export_onnx', type=str, default='../models/model.onnx', help="path to export onnx model (default ../models/model.onnx)")

args = parser.parse_args()
assert args.command == 'train' or args.command == 'eval' or args.command == 'test' or args.command == 'export'

if __name__ == '__main__':

    if args.command == 'train':
        assert args.train_dir is not None and args.train_ann is not None

        # Transform for training
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
        model = LitCondInst(mode='training', num_classes=args.num_classes, learning_rate=args.learning_rate, score_threshold=args.score_threshold)

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

        # Load model for eval or test or export
        print("Loading model")
        model = LitCondInst(mode='inference', num_classes=args.num_classes, topk=args.topk)
        model.load_state_dict(torch.load(args.load_model))
        model.eval()
        if args.mixed_precision:
            # Needs CUDA to support FP16
            model.half().to('cuda')

        if args.command == 'eval':
            assert args.val_dir is not None and args.val_ann is not None

            # Transform for eval
            transform = A.Compose([
                A.LongestMaxSize(max_size=args.input_width, interpolation=cv2.INTER_LINEAR),
                A.transforms.PadIfNeeded(min_width=args.input_width, min_height=args.input_height, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.CenterCrop(width=args.input_width, height=args.input_height),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

            root_dir_val = os.path.expanduser(args.val_dir)
            ann_path_val = os.path.expanduser(args.val_ann)
            dataset_val = CocoSegmentationAlb(root=root_dir_val, annFile=ann_path_val, transform=transform)
            coco_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=lambda x: x, num_workers=args.num_workers)

            # Mixed precision
            if args.mixed_precision:
                precision = 16
            else:
                precision = 32

            # Evaluate model
            trainer = pl.Trainer(
                max_epochs=1,
                gpus=args.gpus,
                precision=precision
            )
            results = trainer.test(model, coco_val, ckpt_path=None)

        elif args.command == 'test':

            # Get list of image paths
            image_paths = glob.glob(os.path.join(args.test_image_dir, '*.jpg'))
            image_paths += glob.glob(os.path.join(args.test_image_dir, '*.jpeg'))
            image_paths += glob.glob(os.path.join(args.test_image_dir, '*.png'))

            for image_path in image_paths:
                print("Test {}".format(image_path))

                # Load test images
                image = cv2.imread(os.path.join(image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Preprocessing
                image = cv2.resize(image, dsize=(args.input_width, args.input_height))
                image_normalized = (image.astype(np.float32) - np.array([0.485, 0.456, 0.406]) * 255) / (np.array([0.229, 0.224, 0.225]) * 255)
                image_normalized = image_normalized.transpose(2, 0, 1) # HWC -> CHW
                image_normalized = image_normalized[None,:,:,:] # CHW -> NCHW
                image_normalized = torch.from_numpy(image_normalized).clone()

                # Use GPU if available
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                image_normalized = image_normalized.to(device)

                # Inference
                labels, probs, masks = model(image_normalized)

                # Conver to numpy array
                labels = labels.to('cpu').detach().numpy().copy()
                probs = probs.to('cpu').detach().numpy().copy()
                masks = masks.to('cpu').detach().numpy().copy()

                # Post processing
                num_objects, = probs[probs > args.score_threshold].shape
                print("{} obects detected".format(num_objects))

                if num_objects > 0:
                    masks = masks[0,:num_objects,:,:]
                else:
                    masks = np.zeros((1, args.input_height, args.input_width), dtype=np.float32)

                masks = masks.transpose(1, 2, 0)
                masks = masks.astype(np.float32)
                masks = cv2.resize(masks, dsize=(args.input_width, args.input_height), interpolation=cv2.INTER_LINEAR)
                masks = (masks > 0.5).astype(np.float32)

                # Add channel dimension if removed by cv2.resize()
                if len(masks.shape) == 2:
                    masks = masks[...,None]

                # Visualize masks
                mask_visualize = np.zeros((args.input_height, args.input_width, 3), dtype=np.float32)
                for i in range(masks.shape[2]):
                    mask_visualize[:,:,0] += masks[:,:,i] * (float(i+1)%8/7)
                    mask_visualize[:,:,1] += masks[:,:,i] * (float(i+1)%4/3)
                    mask_visualize[:,:,2] += masks[:,:,i] * (float(i+1)%2/1)
                mask_visualize = np.clip(mask_visualize, 0, 1)
                mask_visualize = mask_visualize * 255
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_visualize = image / 4 + mask_visualize * 3 / 4
                image_visualize = image_visualize.astype(np.int8)

                # Save results
                save_path = os.path.join(args.test_output_dir, os.path.basename(image_path))
                print("Saving to {}".format(save_path))
                os.makedirs(args.test_output_dir, exist_ok=True)
                cv2.imwrite(save_path, image_visualize)

        else:

            # Export to ONNX
            print("Exporting to ONNX")
            input_sample = torch.randn((1, 3, args.input_height, args.input_width))
            model.to_onnx(args.export_onnx, input_sample, export_params=True, opset_version=11)
            print("Done")
