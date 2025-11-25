#!/bin/sh

# 1. Automate X11 Permissions (Allow all for bridge mode)
xhost +

# 2. CLEANUP: Remove old container
docker rm -f carla_client 2>/dev/null || true

# 3. Setup X11 Files
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# 4. Launch Docker
# - Removed --net=host (Fixes the Foxglove 9090 connection issue)
# - Added -p 9090:9090 (Actually exposes the port now)
docker run --platform linux/amd64 --privileged -it \
           --name carla_client \
           -p 9090:9090 \
           --volume=$XSOCK:$XSOCK:rw \
           --volume=$XAUTH:$XAUTH:rw \
           --volume=$HOME:$HOME \
           --shm-size=1gb \
           --env="XAUTHORITY=${XAUTH}" \
           --env="DISPLAY=host.docker.internal:0" \
           --env="SDL_VIDEODRIVER=x11" \
           --env="SDL_AUDIODRIVER=dummy" \
           --env="PYTHON_EGG_CACHE=/tmp/python-eggs" \
           --env="LIBGL_ALWAYS_SOFTWARE=1" \
           --env="TERM=xterm-256color" \
           --env="QT_X11_NO_MITSHM=1" \
           -u "melodic"  \
           carla:0.9.11 \
           bash -c "mkdir -p /tmp/python-eggs && bash"  