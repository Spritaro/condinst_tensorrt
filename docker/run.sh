#!/bin/bash

docker run -it --rm \
    --gpus all \
    --name condinst_tensorrt \
    --env "DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $1:/home/appuser/condinst_tensorrt:rw \
    -v $2:/home/appuser/dataset:ro \
    -w /home/appuser/condinst_tensorrt \
    -p 6006:6006 \
    --shm-size 512mb \
    condinst_tensorrt \
    bash
