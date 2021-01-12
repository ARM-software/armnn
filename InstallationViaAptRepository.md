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

We provide a number of packages for each architecture; x86_64, aarch64 and armhf as outlined below.

ARMNN_MAJOR_VERSION: This is the ABI version of the ArmNN source that has been packaged based on include/armnn/Version.hpp.

ARMNN_RELEASE_VERSION: This is the marketing release version based on the date source was released on github e.g. 20.11.

PACKAGE_VERSION: This is the version of the source package used to build the binaries packages from.

##### x86_64
* Runtime Packages
```
libarmnn-cpuref-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
libarmnn-tfliteparser{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
libarmnn{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
python3-pyarmnn_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
```
* Development Packages
```
libarmnn-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
libarmnn-tfliteparser-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_amd64.deb
```
##### arm64
* Runtime Packages
```
libarmnn-aclcommon{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn-cpuacc-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn-cpuref-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn-gpuacc-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn-tfliteparser{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
python3-pyarmnn_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb

```
* Development Packages
```
libarmnn-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb
libarmnn-tfliteparser-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_arm64.deb

```
##### armhf
* Runtime Packages
```
libarmnn-aclcommon{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn-cpuacc-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn-cpuref-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn-gpuacc-backend{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn-tfliteparser{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn{ARMNN_MAJOR_VERSION}_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
python3-pyarmnn_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb

```
* Development Packages
```
libarmnn-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb
libarmnn-tfliteparser-dev_{ARMNN_RELEASE_VERSION}-{PACKAGE_VERSION}_armhf.deb

```

#### <a name="VersionPackages"> Check latest version of packages</a>
Due to Debian Packaging requiring the pristine tarball from our Github release, the version on Launchpad may not align with the released version on Github depending on the complexity of newly added features.
In order to check for the latest available ArmNN version use apt-cache search:
```
 apt-cache search libarmnn

 # This returns a list of matching packages, the latest being libarmnn23 i.e. ARMNN_MAJOR_VERSION=23
 libarmnn-cpuref-backend22 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-cpuref-backend23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-dev - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-tfliteparser-dev - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-tfliteparser22 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-tfliteparser23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn22 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-aclcommon22 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-aclcommon23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-cpuacc-backend22 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-cpuacc-backend23 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-gpuacc-backend22 - Arm NN is an inference engine for CPUs, GPUs and NPUs
 libarmnn-gpuacc-backend23 - Arm NN is an inference engine for CPUs, GPUs and NPUs


 # Export the ARMNN_MAJOR_VERSION to allow installation using the below examples
 export ARMNN_MAJOR_VERSION=23
```


#### <a name="InstallPackages"> Install desired combination of packages</a>
The easiest way to install all of the available packages for your systems architecture is to run the command:

(Please Note: libarmnn-cpuacc-backend has been built with NEON support, installing this backend on an armhf device not supporting NEON may cause a crash/undefined behaviour.)
```
 sudo apt-get install -y python3-pyarmnn libarmnn-cpuacc-backend${ARMNN_MAJOR_VERSION} libarmnn-gpuacc-backend${ARMNN_MAJOR_VERSION} libarmnn-cpuref-backend${ARMNN_MAJOR_VERSION}
 # Verify installation via python:
 python3 -c "import pyarmnn as ann;print(ann.GetVersion())" 
 # Returns '{ARMNN_MAJOR_VERSION}.0.0' e.g. 23.0.0
```
This will install PyArmnn and the three backends for Neon, Compute Library and our Reference Backend.
It will also install their dependencies including the arm-compute-library package along with the Tensorflow Lite Parser and it's dependency ArmNN Core.
If the user does not wish to use PyArmnn they can go up a level of dependencies and instead just install the Tensorflow Lite Parser:
```
  sudo apt-get install -y libarmnn-tfliteparser${ARMNN_MAJOR_VERSION} libarmnn-gpuacc-backend${ARMNN_MAJOR_VERSION}
```

#### <a name="uninstallPackages"> Uninstall packages</a>
The easiest way to uninstall all of the previously installed packages is to run the command:
```
 sudo apt autoremove -y libarmnn${ARMNN_MAJOR_VERSION}
```