#!/bin/bash

docker run -it --rm \
    --gpus all \
    --name sparseinst_tensorrt \
    --env="DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $1:/home/appuser/sparseinst_tensorrt:rw \
    -v $2:/home/appuser/dataset:ro \
    -w /home/appuser/sparseinst_tensorrt \
    -p 6006:6006 \
    tensorrt \
    bash
