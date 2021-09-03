# Unofficial Implementation of CondInst for TensorRT

The architecture is based on [CondInst: Conditional Convolutions for Instance Segmentation](https://arxiv.org/abs/2003.05664).
In the original paper, FCOS is used for object detection, but in this implementation, heatmap (similar to the one in [CenterNet: Objects as Points](https://arxiv.org/abs/1904.07850)) is used to make it NMS-free.

## Setup

- Build docker image (without TensorRT).

    ```sh
    $ cd docker
    $ docker build -t condinst_tensorrt .
    ```

- To build docker image with TensorRT, run the following command instead.
    ```sh
    $ cd docker
    $ docker build -t condinst_tensorrt -f Dockerfile.trt .
    ```

## Usage

- Run docker container.
    ```sh
    $ cd docker
    $ ./run.sh <path/to/this/repository> <path/to/dataset/directory>
    ```

### Train

- Train on COCO dataset.
    ```sh
    $ python3 main.py train \
        --train_dir <path/to/train/image/directory> \
        --train_ann <path/to/train/annotation/file> \
        --val_dir <path/to/validation/image/directory> \
        --val_ann <path/to/validation/annotation/file> \
        --num_classes 80 \
        --pretrained_model <path/to/pretrained/model/if/available> \
        --input_width 640 \
        --input_height 480 \
        --batch_size 8 \
        --accumulate_grad_batches 16 \
        --num_workers 4 \
        --mixed_precision True \
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

- Evaluate model
    ```sh
    $ python3 main.py eval \
        --val_dir <path/to/validation/image/directory> \
        --val_ann <path/to/validation/annotation/file> \
        --num_classes 80 \
        --input_width 640 \
        --input_height 480 \
        --batch_size 8 \
        --num_workers 4 \
        --mixed_precision True \
        --gpus 1 \
        --load_model <path/to/model.pt>
    ```

### Test

- Perform inference and visualize results.
    ```sh
    python3 main.py test \
        --num_classes 80 \
        --topk 40 \ # Max number of detection
        --load_model <path/to/model.pt> \
        --test_image_dir <path/to/image/directory> \
        --test_output_dir <path/to/output/directory>
    ```

### Export

- Export ONNX model.
    ```sh
    $ python3 main.py export \
        --num_classes 80 \
        --input_width 640 \
        --input_height 480 \
        --topk 40 \ # Max number of detection
        --load_model <path/to/model.pt> \
        --onnx_model <path/to/model.onnx>
    ```

- Build TensorRT engine on Jetson device or PC with TensorRT installed.
    ```sh
    $ export PATH=/usr/src/tensorrt/bin/:$PATH # Add path to TensorRT binary
    $ trtexec --onnx=<path/to/model.onnx> --dumpProfile --saveEngine=<path/to/tensorrt.engine>
    ```

# TODO
- [ ] Add options for switching backbone
- [ ] Add options to change number of head layers
- [ ] Add RGB-D support
