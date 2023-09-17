import argparse
import cv2
import datetime
import glob
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
subparsers = parser.add_subparsers(dest="command")

# Model options
parser.add_argument('--input_width', type=int, default=640, required=False, help="width of input image (default 640)")
parser.add_argument('--input_height', type=int, default=480, help="height of input image (default 480)")
parser.add_argument('--input_channels', type=int, default=3, choices=[3, 4], help="number of input channels (default 3)")
parser.add_argument('--num_classes', type=int, default=80, help="number of classes (default 80)")
parser.add_argument('--topk', type=int, default=40, help="max number of object to detect during inference (default 40)")
parser.add_argument('--mixed_precision', type=bool, default=True, help="allow FP16 training (default True)")

# Pre-processing options
parser.add_argument('--mean', type=str, default='0.485,0.456,0.406', help="mean values of input data (default '0.485,0.456,0.406')")
parser.add_argument('--std', type=str, default='0.229,0.224,0.225', help="std values of input data (default '0.229,0.224,0.225')")

# Post-processing options
parser.add_argument('--score_threshold', type=float, default=0.3, help="score threshold for detection (default 0.3)")
parser.add_argument('--mask_threshold', type=float, default=0.5, help="threshold for mask binarization (default 0.5)")

# Create parser for "train" command
parser_train = subparsers.add_parser('train', help="train model")
# Datset options
parser_train.add_argument('--train_dir', type=str, required=True, help="path to train image dir (required)")
parser_train.add_argument('--train_ann', type=str, required=True, help="path to train annotation path (required)")
parser_train.add_argument('--train_depth', type=str, default=None, required=False, help="path to train depth image dir (required only when input_channels=4)")
parser_train.add_argument('--val_dir', type=str, default=None, required=False, help="path to validation image dir (optional)")
parser_train.add_argument('--val_ann', type=str, default=None, required=False, help="path to validation dataset dir (optional)")
parser_train.add_argument('--val_depth', type=str, default=None, required=False, help="path to validation depth image dir (required only when input_channels=4)")
# Training options
parser_train.add_argument('--pretrained_model', type=str, default=None, help="path to pretrained model (optional)")
parser_train.add_argument('--batch_size', type=int, default=8, help="batch size (default 8)")
parser_train.add_argument('--accumulate_grad_batches', type=int, default=16, help="number of gradients to accumulate (default 16)")
parser_train.add_argument('--num_workers', type=int, default=4, help="number of workers for data loader (default 4)")
parser_train.add_argument('--resume_from_checkpoint', type=str, default=None, help="path to checkpoint file (optional)")
parser_train.add_argument('--max_epochs', type=int, default=10, help="number of epochs (default 10")
parser_train.add_argument('--learning_rate', type=float, default=0.01, help="learning rate (default 0.01)")
# Logging options
parser_train.add_argument('--tensorboard_log_dir', type=str, default='../runs', help="path to TensorBoard log dir (default '../runs')")
# Output options
parser_train.add_argument('--checkpoint_dir', type=str, default='../checkpoints', help="path to checkpoint directory (default '../checkpoints')")
parser_train.add_argument('--save_model', type=str, default='../models/model.pt', help="path to save trained model (defalut '../models/model.py')")

# Create parser for "eval" command
parser_eval = subparsers.add_parser('eval', help="evaluate model")
# Datset options
parser_eval.add_argument('--val_dir', type=str, required=True, help="path to validation image dir (required)")
parser_eval.add_argument('--val_ann', type=str, required=True, help="path to validation dataset dir (required)")
parser_eval.add_argument('--val_depth', type=str, default=None, required=False, help="path to validation depth image dir (required only when input_channels=4)")
# Evaluation options
parser_eval.add_argument('--batch_size', type=int, default=8, help="batch size (default 8)")
parser_eval.add_argument('--num_workers', type=int, default=4, help="number of workers for data loader (default 4)")
parser_eval.add_argument('--load_model', type=str, default='../models/model.pt', help="path to trained model (default '../models/model.py')")
# Logging options
parser_eval.add_argument('--tensorboard_log_dir', type=str, default='../runs', help="path to TensorBoard log dir (default '../runs')")

# Create parser for "test" command
parser_test = subparsers.add_parser('test', help="test model")
# Test options
parser_test.add_argument('--test_image_dir', type=str, default='../test_image', help="path to test image dir (default '../test_image')")
parser_test.add_argument('--test_depth_dir', type=str, default=None, help="path to test image dir (required only when input_channels=4)")
parser_test.add_argument('--test_output_dir', type=str, default='../test_output', help="path to test output dir (default '../test_output')")
parser_test.add_argument('--load_model', type=str, default='../models/model.pt', help="path to trained model (default '../models/model.py')")
# Logging options
parser_test.add_argument('--tensorboard_log_dir', type=str, default='../runs', help="path to TensorBoard log dir (default '../runs')")

# Create parser for "export" command
parser_export = subparsers.add_parser('export', help="export model to ONNX format")
# Export options
parser_export.add_argument('--load_model', type=str, default='../models/model.pt', help="path to trained model (default '../models/model.py')")
parser_export.add_argument('--export_onnx', type=str, default='../models/model.onnx', help="path to export onnx model (default ../models/model.onnx)")

args = parser.parse_args()

if __name__ == '__main__':
    args.mean = [float(item) for item in args.mean.split(',')]
    args.std = [float(item) for item in args.std.split(',')]

    if args.command == 'train':

        # Transform for training
        transform_train = A.Compose([
            A.Affine(scale={'x': (0.7, 1.5), 'y': (0.7, 1.5)}, translate_percent={'x': (-0.5, 0.5), 'y': (-0.2, 0.2)}, rotate=(-90, 90), interpolation=cv2.INTER_LINEAR, mode=cv2.BORDER_CONSTANT, p=0.8),
            A.PadIfNeeded(min_width=args.input_width, min_height=args.input_height, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomCrop(width=args.input_width, height=args.input_height),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])

        # Load data
        root_dir_train = os.path.expanduser(args.train_dir)
        ann_path_train = os.path.expanduser(args.train_ann)
        depth_dir_train = os.path.expanduser(args.train_depth) if args.train_depth is not None else None
        dataset_train = CocoSegmentationAlb(root=root_dir_train, annFile=ann_path_train, depthDir=depth_dir_train, transform=transform_train)
        coco_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x: x)

        if args.val_dir and args.val_ann:
            transform_val = A.Compose([
                A.PadIfNeeded(min_width=args.input_width, min_height=args.input_height, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.CenterCrop(width=args.input_width, height=args.input_height),
                A.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ])

            root_dir_val = os.path.expanduser(args.val_dir)
            ann_path_val = os.path.expanduser(args.val_ann)
            depth_dir_val = os.path.expanduser(args.val_depth) if args.val_depth is not None else None
            dataset_val = CocoSegmentationAlb(root=root_dir_val, annFile=ann_path_val, depthDir=depth_dir_val, transform=transform_val)
            coco_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=lambda x: x, num_workers=args.num_workers)
        else:
            coco_val = None

        # Save checkpoints
        path = os.path.join(args.checkpoint_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(path, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            dirpath=path,
            filename='{epoch:03d}-{step:06d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
            save_last=True
        )

        # Create model for training
        model = LitCondInst(
            mode='training',
            input_channels=args.input_channels,
            num_classes=args.num_classes,
            topk=args.topk,
            learning_rate=args.learning_rate,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold)

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
            val_check_interval=1.0, # validate once per epoch
            accumulate_grad_batches=args.accumulate_grad_batches,
            callbacks=[checkpoint_callback],
            precision=precision,
            accelerator="auto",
            logger=tb_logger
        )
        trainer.fit(model, coco_train, coco_val, ckpt_path=args.resume_from_checkpoint)

        # Save model
        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        torch.save(model.state_dict(), args.save_model)

    else:

        # Load model for eval or test or export
        print("Loading model")
        model = LitCondInst(
            mode='inference',
            input_channels=args.input_channels,
            num_classes=args.num_classes,
            topk=args.topk,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold)
        model.load_state_dict(torch.load(args.load_model))
        model.eval()
        if args.mixed_precision:
            # Needs CUDA to support FP16
            model.half().to('cuda')

        if args.command == 'eval':

            # Transform for eval
            transform = A.Compose([
                A.PadIfNeeded(min_width=args.input_width, min_height=args.input_height, border_mode=cv2.BORDER_CONSTANT, value=0),
                A.CenterCrop(width=args.input_width, height=args.input_height),
                A.Normalize(mean=args.mean, std=args.std),
                ToTensorV2(),
            ])

            root_dir_val = os.path.expanduser(args.val_dir)
            ann_path_val = os.path.expanduser(args.val_ann)
            depth_dir_val = os.path.expanduser(args.val_depth) if args.val_depth is not None else None
            dataset_val = CocoSegmentationAlb(root=root_dir_val, annFile=ann_path_val, depthDir=depth_dir_val, transform=transform)
            coco_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=lambda x: x, num_workers=args.num_workers)

            # Mixed precision
            if args.mixed_precision:
                precision = 16
            else:
                precision = 32

            # Logger
            os.makedirs(args.tensorboard_log_dir, exist_ok=True)
            tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.tensorboard_log_dir, default_hp_metric=False)

            # Evaluate model
            trainer = pl.Trainer(
                max_epochs=1,
                precision=precision,
                accelerator="auto",
                logger=tb_logger
            )
            results = trainer.test(model, coco_val, ckpt_path=None)

        elif args.command == 'test':

            result_images = []

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
                image_normalized = (image.astype(np.float32) - np.array(args.mean) * 255) / (np.array(args.std) * 255)
                image_normalized = image_normalized.transpose(2, 0, 1) # HWC -> CHW
                image_normalized = image_normalized[None,:,:,:] # CHW -> NCHW
                image_normalized = torch.from_numpy(image_normalized).clone()

                # Read depth image and concatenate to image
                if args.test_depth_dir is not None:
                    basename = os.path.basename(image_path)
                    basename = os.path.splitext(basename)[0] + '.png'
                    path = os.path.join(args.test_depth_dir, basename)
                    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
                    depth = depth.astype(np.float32)

                    _, _, h, w = image_normalized.shape
                    rgbd = torch.zeros(size=(1, 4, h, w), dtype=torch.float32)
                    rgbd[0,:3,:,:] = image_normalized
                    rgbd[0,3,:,:] = torch.from_numpy(depth).clone() / 1000.
                    image_normalized = rgbd

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
                masks = (masks > args.mask_threshold).astype(np.float32)

                # Add channel dimension if removed by cv2.resize()
                if len(masks.shape) == 2:
                    masks = masks[...,None]

                # Visualize masks
                mask_visualize = np.zeros((args.input_height, args.input_width, 3), dtype=np.float32)
                for i in range(masks.shape[2]):
                    mask_visualize[:,:,0] += masks[:,:,i] * (float(i+1)%5/4)
                    mask_visualize[:,:,1] += masks[:,:,i] * (float(i+1)%4/3)
                    mask_visualize[:,:,2] += masks[:,:,i] * (float(i+1)%3/2)
                    contours, _ = cv2.findContours(masks[:,:,i].astype(np.uint8)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 0:
                        cnt = np.concatenate(contours, axis=0)
                        x, y, w, h = cv2.boundingRect(cnt)
                        image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
                mask_visualize = np.clip(mask_visualize, 0, 1)
                mask_visualize = mask_visualize * 255
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_visualize = image / 4 + mask_visualize * 3 / 4
                image_visualize = image_visualize.astype(np.uint8)

                # Save results
                save_path = os.path.join(args.test_output_dir, os.path.basename(image_path))
                print("Saving to {}".format(save_path))
                os.makedirs(args.test_output_dir, exist_ok=True)
                cv2.imwrite(save_path, image_visualize)

                result_images.append(image_visualize)

            # Show result images on TensorBoard
            log_dirs = glob.glob(os.path.join(args.tensorboard_log_dir, "default", "*"))
            if len(log_dirs) > 0:
                from torch.utils.tensorboard import SummaryWriter
                log_dirs.sort(key=os.path.getctime)
                writer = SummaryWriter(log_dirs[-1])

                x = np.stack(result_images, axis=0)
                x = np.transpose(x, axes=[0, 3, 1, 2])
                x = torch.from_numpy(x.astype(np.uint8)).clone()
                img_grid = torchvision.utils.make_grid(x)

                writer.add_image("test_results", img_grid)
                writer.close()

        else:

            # Export to ONNX
            print("Exporting to ONNX")
            input_names = ["input"]
            output_names = ["output_labels", "output_scores", "output_masks"]
            input_sample = torch.randn((1, args.input_channels, args.input_height, args.input_width))
            model.to_onnx(
                args.export_onnx, input_sample, export_params=True, opset_version=11,
                input_names=input_names, output_names=output_names)
            print("Done")
