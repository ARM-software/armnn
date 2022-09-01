# How to install ArmNN via our APT repository on Ubuntu's Launchpad

* [Introduction](#introduction)
* [Add the Ubuntu Launchpad PPA to your system](#add-the-ubuntu-launchpad-ppa-to-your-system)
* [Outline of available packages](#outline-of-available-packages)
  + [x86_64](#x86_64)
  + [arm64](#arm64)
  + [armhf](#armhf)
* [Install desired combination of packages](#install-desired-combination-of-packages)
* [Installation of specific ABI versioned packages](#installation-of-specific-abi-versioned-packages)
* [Uninstall packages](#uninstall-packages)


## Introduction
These are the step by step instructions on how to install the Arm NN core, TensorflowLite Parser
as well as PyArmNN for x86_64, Arm64 and Armhf for Ubuntu 20.04.
The packages will also be added to Debian Bullseye, their progress can be tracked here:
https://tracker.debian.org/pkg/armnn.


## Add the Ubuntu Launchpad PPA to your system
* Add the PPA to your sources using a command contained in software-properties-common package:
    ```
    sudo apt install software-properties-common
    sudo add-apt-repository ppa:armnn/ppa
    sudo apt update
    ```
* More information about our PPA and the Ubuntu Launchpad service can be found at [launchpad.net](https://launchpad.net/~armnn/+archive/ubuntu/ppa)
## Outline of available packages

We provide a number of packages for each architecture; x86_64, aarch64 and armhf as outlined below.

ARMNN_MAJOR_VERSION: This is the ABI version of the Arm NN source that has been packaged based on
include/armnn/Version.hpp.

ARMNN_RELEASE_VERSION: This is the marketing release version based on the date source was released on github e.g. 20.11.

PACKAGE_VERSION: This is the version of the source package used to build the binaries packages from.

### x86_64
* Runtime Packages
```
libarmnn-cpuref-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
libarmnntfliteparser{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
libarmnn{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
python3-pyarmnn_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
```
* Development Packages
```
libarmnn-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
libarmnntfliteparser-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
```
* Dependency Packages (These are empty packages that provide a user-friendly name for other packages they will install)
```
armnn-latest-all_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
armnn-latest-ref_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
```
### arm64
* Runtime Packages
```
libarmnn-aclcommon{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn-cpuacc-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn-cpuref-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn-gpuacc-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnntfliteparser{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
python3-pyarmnn_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb

```
* Development Packages
```
libarmnn-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnntfliteparser-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb

```
* Dependency Packages (These are empty packages that provide a user-friendly name for other packages they will install)
```
armnn-latest-all_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
armnn-latest-cpu_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
armnn-latest-cpu-gpu_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
armnn-latest-cpu-gpu-ref_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
armnn-latest-gpu_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
armnn-latest-ref_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
```
### armhf
* Runtime Packages
```
libarmnn-aclcommon{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn-cpuacc-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn-cpuref-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn-gpuacc-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnntfliteparser{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
python3-pyarmnn_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb

```
* Development Packages
```
libarmnn-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnntfliteparser-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb

```
* Dependency Packages (These are empty packages that provide a user-friendly name for other packages they will install)
```
armnn-latest-all_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
armnn-latest-cpu_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
armnn-latest-cpu-gpu_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
armnn-latest-cpu-gpu-ref_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
armnn-latest-gpu_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
armnn-latest-ref_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
```

## Install desired combination of packages
The easiest way to install all of the available packages for your systems architecture is to run the command:

```
 sudo apt-get install -y python3-pyarmnn armnn-latest-all
 # Verify installation via python:
 python3 -c "import pyarmnn as ann;print(ann.GetVersion())"
 # Returns '{ARMNN_MAJOR_VERSION}.0.0' e.g. 32.0.0
```
This will install PyArmNN and the three backends for Neon (CpuAcc), OpenCL (GpuAcc) and our Reference Backend.
It will also install their dependencies including the arm-compute-library package along with the Tensorflow Lite Parser
and it's dependency Arm NN Core.
If the user does not wish to use PyArmNN they can go up a level of dependencies and instead just install the
armnn-latest-all package:
```
  # Install ArmNN Core, CpuAcc Backend, GpuAcc Backend and Reference Backend as well as the TensorFlow Lite Parser:
  # (This will only install CpuAcc and GpuAcc Backends on arm64 and armhf architectures)
  sudo apt-get install -y armnn-latest-all

  # Install ArmNN Core, CpuAcc Backend as well as the TensorFlow Lite Parser:
  sudo apt-get install -y armnn-latest-cpu

  # Install ArmNN Core, CpuAcc Backend, GpuAcc Backend as well as the TensorFlow Lite Parser:
  sudo apt-get install -y armnn-latest-cpu-gpu

  # Install ArmNN Core, GpuAcc Backend as well as the TensorFlow Lite Parser:
  sudo apt-get install -y armnn-latest-gpu

  # Install ArmNN Core, Reference Backend as well as the TensorFlow Lite Parser:
  sudo apt-get install -y armnn-latest-ref
```

## Installation of specific ABI versioned packages
Due to Debian Packaging requiring the pristine tarball from our Github release, the version on Launchpad may not align
with the released version on Github depending on the complexity of newly added features.
In order to check for the latest available Arm NN version use apt-cache search:
```
 apt-cache search libarmnn

 # This returns a list of matching packages including versions from previous releases
 libarmnn-cpuref-backend23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-cpuref-backend24 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-dev - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnntfliteparser-dev - Arm NN is an inference engine for CPUs, GPUs and NPUs # Note: removal of dash to suit debian naming conventions
 libarmnn-tfliteparser23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnntfliteparser24 - Arm NN is an inference engine for CPUs, GPUs and NPUs # Note: removal of dash to suit debian naming conventions
 libarmnntfliteparser24.5 - Arm NN is an inference engine for CPUs, GPUs and NPUs # Note: removal of dash to suit debian naming conventions
 libarmnn23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn24 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn25 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn30 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-aclcommon23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnnaclcommon24 - Arm NN is an inference engine for CPUs, GPUs and NPUs # Note: removal of dash to suit debian naming conventions
 libarmnn-cpuacc-backend23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-cpuacc-backend24 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-gpuacc-backend23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-gpuacc-backend24 - Arm NN is an inference engine for CPUs, GPUs and NPUs


 # Export the ARMNN_MAJOR_VERSION to the latest visible e.g. libarmnn30 to allow installation using the below examples
 export ARMNN_MAJOR_VERSION=30

  # As the Tensorflow Lite Parser is now ABI stable it will have a different version to ARMNN_MAJOR_VERSION please choose latest version:
  apt-cache search libarmnntfliteparser
  # Returns e.g. libarmnntfliteparser24.5 so we then export that version, for reference this comes from include/armnnTfLiteParser/Version.hpp:
  export TFLITE_PARSER_VERSION=24.5

  sudo apt-get install -y libarmnntfliteparser${TFLITE_PARSER_VERSION} libarmnn-cpuacc-backend${ARMNN_MAJOR_VERSION}
```

## Uninstall packages
The easiest way to uninstall all of the previously installed packages is to run the command:
```
 sudo apt-get purge -y armnn-latest-all
 sudo apt autoremove -y armnn-latest-all
```
