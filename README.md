# Unofficial Implementation of CondInst for TensorRT

This repository is an unofficial implementation of [CondInst: Conditional Convolutions for Instance Segmentation](https://arxiv.org/abs/2003.05664).

This implementation utilizes heatmap-based object detection. Please see [CornerNet: Detecting Objects as Paired Keypoints](https://arxiv.org/abs/1808.01244) and [CenterNet: Objects as Points](https://arxiv.org/abs/1904.07850) for details.

## Benchmark

| Backbone | Dataset | Input image size | Output mask size | mAP | FP16 inference time<br>Jetson Nano 4GB | FP16 inference time<br>Jetson Xavier NX | Download link |
| ---      | ---     | ---              | ---              | --- | ---                  | ---                 | ---           |
| ResNet50-FPN | COCO2017 | 640x480 | 160x120 | 20.1 | 406 ms | 57.8 ms | [weight](https://drive.google.com/file/d/1oO6G_LlUyTwOjsI_sElxhLa7Yv82J-qo/view?usp=sharing), [onnx](https://drive.google.com/file/d/1WglUK8_Q9F8_jFf-dfy9VZaZo98Ue-3M/view?usp=sharing) |

## Setup

- Build docker image.
    ```sh
    $ cd docker
    $ docker build -t condinst_tensorrt -f Dockerfile.tensorrt .
    ```

- To build docker image without TensorRT, run the following command instead.

    ```sh
    $ cd docker
    $ docker build -t condinst_tensorrt .
    ```

- To build docker image on Jetson, run the following command.
    ```sh
    $ cd docker
    $ docker build -t condinst_tensorrt -f Dockerfile.jetson .
    ```

## Usage

- Run docker container.
    ```sh
    $ cd docker
    $ ./run.sh <path/to/this/repository> <path/to/dataset/directory>
    ```

- To run docker container on Jetson, run the following command instead.
    ```sh
    $ cd docker
    $ ./run_jetson.sh <path/to/this/repository> <path/to/dataset/directory>
    ```

### Train

- Train on COCO dataset.
    ```sh
    $ python3 main.py \
        --input_width 640 \
        --input_height 480 \
        --num_classes 80 \
        --topk 40 \
        --mixed_precision True \
        train \
        --train_dir <path/to/train/image/directory> \
        --train_ann <path/to/train/annotation/file> \
        --val_dir <path/to/validation/image/directory> \
        --val_ann <path/to/validation/annotation/file> \
        --pretrained_model <path/to/pretrained/model/if/available> \
        --batch_size 8 \
        --accumulate_grad_batches 16 \
        --num_workers 4 \
        --max_epochs 10 \
        --gpus 1 \
        --learning_rate 0.01 \
        --save_model <path/to/model.pt>
    ```
    - You might need to increase shared memory size for docker container in order to increase ```num_workers``` or ```input_width``` or ```input_height```.
    - If you experience NaN losses, try lowering ```learning_rate``` or disabling ```mixed_precision```

- Visualize loss on Tensorboard
    ```sh
    $ tensorboard --logdir=runs
    ```

### Evaluation

- Evaluate model. This feature is experimental and may not produce accurate results.
    ```sh
    $ python3 main.py \
        --input_width 640 \
        --input_height 480 \
        --num_classes 80 \
        --topk 40 \
        --mixed_precision True \
        eval \
        --val_dir <path/to/validation/image/directory> \
        --val_ann <path/to/validation/annotation/file> \
        --batch_size 8 \
        --num_workers 4 \
        --gpus 1 \
        --load_model <path/to/model.pt>
    ```

### Test

- Perform inference and visualize results.
    ```sh
    python3 main.py \
        --input_width 640 \
        --input_height 480 \
        --num_classes 80 \
        --topk 40 \ # Max number of detection
        --mixed_precision True \
        test \
        --test_image_dir <path/to/image/directory> \
        --test_output_dir <path/to/output/directory> \
        --load_model <path/to/model.pt>
    ```

### Export

- Export ONNX model.
    ```sh
    $ python3 main.py \
        --input_width 640 \
        --input_height 480 \
        --num_classes 80 \
        --topk 40 \ # Max number of detection
        --mixed_precision True \
        export \
        --load_model <path/to/model.pt> \
        --export_onnx <path/to/model.onnx>
    ```

- Build TensorRT engine on Jetson device or PC with TensorRT installed.
    ```sh
    $ export PATH=/usr/src/tensorrt/bin/:$PATH # Add path to TensorRT binary
    $ trtexec --fp16 --onnx=<path/to/model.onnx> --dumpProfile --saveEngine=<path/to/tensorrt.engine>
    ```

- Run TensorRT demo.
    ```sh
    $ python3 demo_tensorrt.py \
        --test_image_dir <path/to/image/directory> \
        --test_output_dir <path/to/output/directory> \
        --load_engine <path/to/tensorrt.engine>
    ```

### RGB-D images
See [Instructions for RGB-D images](docs/rgbd.md) for details.
