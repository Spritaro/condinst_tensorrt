#!/bin/bash

docker run -it --rm \
    --runtime nvidia \
    --network host \
    --group-add video \
    --name condinst_tensorrt \
    --env "DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $1:/home/appuser/condinst_tensorrt:rw \
    -v $2:/home/appuser/dataset:ro \
    -w /home/appuser/condinst_tensorrt \
    --shm-size 512mb \
    condinst_tensorrt \
    bash
