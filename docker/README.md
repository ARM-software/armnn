# ARMNN Docker Files


## [Android NDK to build ArmNN](https://github.com/ARM-software/armnn/blob/branches/armnn_20_02/BuildGuideAndroidNDK.md):</br>

<b>armnn-android</b> folder has the docker file to build a Android NDK container to build ARMNN.

## [ArmNN on x86_64 for arm64](https://github.com/ARM-software/armnn/blob/branches/armnn_20_02/BuildGuideCrossCompilation.md)

<b>x86_64</b> folder has the docker file to build ArmNN under an x86_64 system to target an Arm64 system.

# To build a docker images
```bash
docker build --rm --build-arg proxy=$http_proxy --rm --tag armnn:v1 .
```

# To Run docker images
```bash
docker run -v /etc/localtime:/etc/localtime:ro --rm -it -e http_proxy -e https_proxy -e ftp_proxy -v `pwd`:/work armnn:v1 bash
```

# To run a docker with X11 support for GUI Application
```bash
docker run -v /etc/localtime:/etc/localtime:ro --rm -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e http_proxy -e https_proxy -e ftp_proxy -v `pwd`:/work armnn:v1 bash
```

# To mount the camera and access camera from docker env
```bash
docker run -v /etc/localtime:/etc/localtime:ro --rm -it --device /dev/video0 -e http_proxy -e https_proxy -e ftp_proxy -v `pwd`:/work armnn:v1 bash
```




