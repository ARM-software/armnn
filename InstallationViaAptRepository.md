# How to install ArmNN via our APT repository on Ubuntu's Launchpad

*  [Introduction](#introduction)
*  [Add the Ubuntu Launchpad PPA to your system](#addRepo)
*  [Outline of available packages](#availablePackages)
*  [Install desired combination of packages](#InstallPackages)
*  [Uninstall packages](#uninstallPackages)


#### <a name="introduction">Introduction</a>
These are the step by step instructions on how to install the ArmNN core, TensorflowLite Parser as well as PyArmnn for x86_64, Arm64 and Armhf for Ubuntu 20.04.
The packages will also be added to Debian Bullseye, their progress can be tracked here: https://tracker.debian.org/pkg/armnn


#### <a name="addRepo">Add the Ubuntu Launchpad PPA to your system</a>
* Add the PPA to your sources using a command contained in software-properties-common package:
    ```
    sudo apt install software-properties-common
    sudo add-apt-repository ppa:armnn/ppa
    sudo apt update
    ```
* More information about our PPA and the Ubuntu Launchpad service can be found at [launchpad.net](https://launchpad.net/~armnn/+archive/ubuntu/ppa)
#### <a name="availablePackages"> Outline of available packages</a>

We provide a number of packages for each architecture; x86_64, aarch64 and armhf:

##### x86_64
* Runtime Packages
```
libarmnn-cpuref-backend22_20.08-4_amd64.deb
libarmnn-tfliteparser22_20.08-4_amd64.deb
libarmnn22_20.08-4_amd64.deb
python3-pyarmnn_20.08-4_amd64.deb
```
* Development Packages
```
libarmnn-dev_20.08-4_amd64.deb
libarmnn-tfliteparser-dev_20.08-4_amd64.deb
```
##### arm64
* Runtime Packages
```
libarmnn-aclcommon22_20.08-4_arm64.deb
libarmnn-cpuacc-backend22_20.08-4_arm64.deb
libarmnn-cpuref-backend22_20.08-4_arm64.deb
libarmnn-gpuacc-backend22_20.08-4_arm64.deb
libarmnn-tfliteparser22_20.08-4_arm64.deb
libarmnn22_20.08-4_arm64.deb
python3-pyarmnn_20.08-4_arm64.deb

```
* Development Packages
```
libarmnn-dev_20.08-4_arm64.deb
libarmnn-tfliteparser-dev_20.08-4_arm64.deb

```
##### armhf
* Runtime Packages
```
libarmnn-aclcommon22_20.08-4_armhf.deb
libarmnn-cpuacc-backend22_20.08-4_armhf.deb
libarmnn-cpuref-backend22_20.08-4_armhf.deb
libarmnn-gpuacc-backend22_20.08-4_armhf.deb
libarmnn-tfliteparser22_20.08-4_armhf.deb
libarmnn22_20.08-4_armhf.deb
python3-pyarmnn_20.08-4_armhf.deb

```
* Development Packages
```
libarmnn-dev_20.08-4_armhf.deb
libarmnn-tfliteparser-dev_20.08-4_armhf.deb

```

#### <a name="InstallPackages"> Install desired combination of packages</a>
The easiest way to install all of the available packages for your systems architecture is to run the command:

(Please Note: libarmnn-cpuacc-backend has been built with NEON support, installing this backend on an armhf device not supporting NEON may cause a crash/undefined behaviour.)
```
 sudo apt-get install -y python3-pyarmnn libarmnn-cpuacc-backend22 libarmnn-gpuacc-backend22 libarmnn-cpuref-backend22
 # Verify installation via python:
 python3 -c "import pyarmnn as ann;print(ann.GetVersion())" 
 # Returns '22.0.0'
```
This will install PyArmnn and the three backends for Neon, Compute Library and our Reference Backend.
It will also install their dependencies including the arm-compute-library package along with the Tensorflow Lite Parser and it's dependency ArmNN Core.
If the user does not wish to use PyArmnn they can go up a level of dependencies and instead just install the Tensorflow Lite Parser:
```
  sudo apt-get install -y libarmnn-tfliteparser22 libarmnn-gpuacc-backend22
```

#### <a name="uninstallPackages"> Uninstall packages</a>
The easiest way to uninstall all of the previously installed packages is to run the command:
```
 sudo apt autoremove -y libarmnn22
```