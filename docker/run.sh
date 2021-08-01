#!/bin/bash

docker run -it --rm \
    --gpus all \
    --name centernet_condinst_pl \
    --env "DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $1:/home/appuser/centernet_condinst_pl:rw \
    -v $2:/home/appuser/dataset:ro \
    -w /home/appuser/centernet_condinst_pl \
    -p 6006:6006 \
    --shm-size 512mb \
    centernet_condinst_pl \
    bash
