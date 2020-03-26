# DOCKER IMAGES for ARMNN Compilation Environment

# To build a docker images
docker build --rm --build-arg proxy=$http_proxy --rm --tag armnn:v1 .

# To Run docker images
docker run -v /etc/localtime:/etc/localtime:ro --rm -it -e http_proxy -e https_proxy -e ftp_proxy -v `pwd`:/work armnn:v1 bash

# To run a docker with X11 support for GUI Application
docker run -v /etc/localtime:/etc/localtime:ro --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e http_proxy -e https_proxy -e ftp_proxy -v `pwd`:/work armnn:v1 bash

# To mount the camera and access camera from docker env
docker run -v /etc/localtime:/etc/localtime:ro --rm -it --device /dev/video0 -e http_proxy -e https_proxy -e ftp_proxy -v `pwd`:/work armnn:v1 bash
