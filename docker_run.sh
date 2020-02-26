#!/usr/bin/env bash

## Fail if any command fails (use "|| true" if a command is ok to fail)
set -e
## Treat unset variables as error
set -u

## Setup X authority such that the container knows how to do graphical stuff
XSOCK="/tmp/.X11-unix";
XAUTH=`tempfile -s .docker.xauth`;
xauth nlist "${DISPLAY}"          \
  | sed -e 's/^..../ffff/'        \
  | xauth -f "${XAUTH}" nmerge -;

docker run                   \
  --rm                            \
  --volume "${XSOCK}:${XSOCK}:rw" \
  --volume "${XAUTH}:${XAUTH}:rw" \
  --env "XAUTHORITY=${XAUTH}"     \
  --env DISPLAY                   \
   --volume "${PWD}/data/:/host:rw"        \
  --hostname "${HOSTNAME}"        \
  --env QT_X11_NO_MITSHM=1 \
  -it docker-freicalib /bin/bash;

#python create_marker.py --tfam 36h11 --nx 10 --ny 4 --tsize 0.05 --double
#python vis_K.py tags/marker_32h11b2_4x10x_5cm.json data/run000_cam1.avi --calib_file_name K_cam1.json

