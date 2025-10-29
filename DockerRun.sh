#!/usr/bin/env bash
IMAGE_REPOSITORY=llms_from_scratch_env
IMAGE_TAG=latest
IMAGE_FULLNAME=${IMAGE_REPOSITORY}:${IMAGE_TAG}

# docker build
docker build -t ${IMAGE_FULLNAME} .

# run container
docker run \
--gpus all \
--interactive \
--tty \
--rm \
--mount=type=bind,src="$(pwd)",dst=/root/share \
--mount=type=bind,src=/etc/group,dst=/etc/group,readonly \
--mount=type=bind,src=/etc/passwd,dst=/etc/passwd,readonly \
$( if [ -e $HOME/.Xauthority ]; then echo "--mount=type=bind,src=$HOME/.Xauthority,dst=/root/.Xauthority"; fi ) \
--env=QT_X11_NO_MITSHM=1 \
--env=DISPLAY=${DISPLAY} \
--net=host \
--workdir /root/share \
--name=${IMAGE_REPOSITORY}_${IMAGE_TAG}_$(date "+%Y_%m%d_%H%M%S") \
${IMAGE_FULLNAME} \
bash
