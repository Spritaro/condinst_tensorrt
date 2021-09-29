# Instructions for RGB-D images

To use RGB-D images, follow the instructions below.

## RGB-D dataset format

The RGB-D dataset format is basicall COCO format with depth images. The depth images must be ```png``` and their filenames must be the same as corresponding color images except for the extension.

## Usage

### Train

- Train on RGB-D dataset.
    ```sh
    $ python3 main.py \
        --num_classes <number_of_classes> \
        --input_channels 4 \
        train \
        --train_dir <path/to/train/image/directory> \
        --train_ann <path/to/train/annotation/file> \
        --train_depth <path/to/train/depth/image/directory> \
        --val_dir <path/to/validation/image/directory> \
        --val_ann <path/to/validation/annotation/file> \
        --val_depth <path/to/validation/depth/image/directory> \
        --save_model <path/to/model.pt>
    ```

### Evaluation

- Evaluate model.
    ```sh
    $ python3 main.py \
        --num_classes <number_of_classes> \
        --input_channels 4 \
        eval \
        --val_dir <path/to/validation/image/directory> \
        --val_ann <path/to/validation/annotation/file> \
        --val_depth <path/to/validation/depth/image/directory> \
        --load_model <path/to/model.pt>
    ```

### Test

- Perform inference and visualize results.
    ```sh
    python3 main.py \
        --num_classes <number_of_classes> \
        --input_channels 4 \
        test \
        --test_image_dir <path/to/image/directory> \
        --test_depth_dir <path/to/depth/image/directory> \
        --test_output_dir <path/to/output/directory> \
        --load_model <path/to/model.pt>
    ```

### Export

- Export ONNX model.
    ```sh
    $ python3 main.py \
        --num_classes <number_of_classes> \
        --input_channels 4 \
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
        --test_depth_dir <path/to/depth/image/directory> \
        --test_output_dir <path/to/output/directory> \
        --load_engine <path/to/tensorrt.engine>
    ```
