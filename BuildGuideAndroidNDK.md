# How to use the Android NDK to build Arm NN

- [Introduction](#introduction)
- [Download the Android NDK and make a standalone toolchain](#download-the-android-ndk-and-make-a-standalone-toolchain)
- [Build the Boost C++ libraries](#build-the-boost-c---libraries)
- [Build the Compute Library](#build-the-compute-library)
- [Build Google's Protobuf library](#build-google-s-protobuf-library)
- [Build Arm NN](#build-armnn)
- [Build Standalone Sample Dynamic Backend](#build-standalone-sample-dynamic-backend)
- [Run the Arm NN unit tests on an Android device](#run-the-armnn-unit-tests-on-an-android-device)


## Introduction
These are step by step instructions for using the Android NDK to build Arm NN.
They have been tested on a clean install of Ubuntu 16.04, and should also work with other OS versions.
The instructions show how to build the Arm NN core library.
Building protobuf is optional. We have given steps should the user wish to build it (i.e. as an Onnx dependency).
All downloaded or generated files will be saved inside the `~/armnn-devenv` directory.

#####Note: We are currently in the process of removing boost as a dependency to Arm NN. This process is finished for everything apart from our unit tests. This means you don't need boost to build and use Arm NN but you need it to execute our unit tests. Boost will soon be removed from Arm NN entirely. 

## Download the Android NDK and make a standalone toolchain

* Download the Android NDK from [the official website](https://developer.android.com/ndk/downloads/index.html):
 ```bash
 mkdir -p ~/armnn-devenv/toolchains
 cd ~/armnn-devenv/toolchains
 # For Mac OS, change the NDK download link accordingly.
 wget https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip
 unzip android-ndk-r20b-linux-x86_64.zip
 export NDK=~/armnn-devenv/android-ndk-r20b
 export NDK_TOOLCHAIN_ROOT=$NDK/toolchains/llvm/prebuilt/linux-x86_64
 export PATH=$NDK_TOOLCHAIN_ROOT/bin/:$PATH

 # You may want to append the above export variables commands to your `~/.bashrc` (or `~/.bash_profile` in Mac OS).
 ```

* With the android ndk-20b, you don't need to use the make_standalone_toolchain script to create a toolchain for a specific version of android. Android's current preference is for you to just specify the architecture and operating system while setting the compiler and just use the ndk directory.

## Build the Boost C++ libraries

* Download Boost version 1.64:
```bash
mkdir ~/armnn-devenv/boost
cd ~/armnn-devenv/boost
wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2
tar xvf boost_1_64_0.tar.bz2
```

* Build:

	(Requires clang if not previously installed: `sudo apt-get install clang`)
    
    Note: You can specify the 'Android_API' version you want. For example, if your ANDROID_API is 27 then the compiler will be aarch64-linux-android27-clang++.
```bash
echo "using clang : arm : aarch64-linux-android<Android_API>-clang++ ;" > $HOME/armnn-devenv/boost/user-config.jam
cd ~/armnn-devenv/boost/boost_1_64_0
./bootstrap.sh --prefix=$HOME/armnn-devenv/boost/install
./b2 install --user-config=$HOME/armnn-devenv/boost/user-config.jam \
 toolset=clang-arm link=static cxxflags=-fPIC \
 --with-test --with-log --with-program_options -j16
```

## Build the Compute Library
* Clone the Compute Library:

	(Requires Git if not previously installed: `sudo apt install git`)
``` bash
cd ~/armnn-devenv
git clone https://github.com/ARM-software/ComputeLibrary.git
```

* Checkout ComputeLibrary branch:
```bash
cd ComputeLibrary
git checkout <tag_name>
```
For example, if you want to checkout release tag of 21.02:
```bash
git checkout v21.02
```

* Build:

	(Requires SCons if not previously installed: `sudo apt install scons`)
```bash
scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" \
 benchmark_tests=0 validation_tests=0 os=android -j16
```

## Build Google's Protobuf library (Optional)

* Clone protobuf:
```bash
mkdir ~/armnn-devenv/google
cd ~/armnn-devenv/google
git clone https://github.com/google/protobuf.git
cd protobuf
git checkout -b v3.12.0 v3.12.0
```

* Build a native (x86) version of the protobuf libraries and compiler (protoc):

	(Requires cUrl, autoconf, llibtool, and other build dependencies if not previously installed: `sudo apt install curl autoconf libtool build-essential g++`)
```bash
./autogen.sh
mkdir x86_build
cd x86_build
../configure --prefix=$HOME/armnn-devenv/google/x86_pb_install
make install -j16
cd ..
```

* Build the arm64 version of the protobuf libraries:
```bash
mkdir arm64_build
cd arm64_build
CC=aarch64-linux-android<Android_API>-clang \
CXX=aarch64-linux-android<Android_API>-clang++ \
CFLAGS="-fPIE -fPIC" \
    LDFLAGS="-llog -lz -lc++_static" \
   ../configure --host=aarch64-linux-android \
   --prefix=$HOME/armnn-devenv/google/arm64_pb_install \
   --enable-cross-compile \
   --with-protoc=$HOME/armnn-devenv/google/x86_pb_install/bin/protoc
make install -j16
cd ..
```

## Build Arm NN

* Clone Arm NN source code:
```bash
cd ~/armnn-devenv/
git clone https://github.com/ARM-software/armnn.git
```

* Checkout Arm NN branch:
```bash
cd armnn
git checkout <branch_name>
git pull
```

For example, if you want to checkout release branch of 21.02:
```bash
git checkout branches/armnn_21_02
git pull
```

* Build Arm NN:

 	(Requires CMake if not previously installed: `sudo apt install cmake`)
```bash
mkdir ~/armnn-devenv/armnn/build
cd ~/armnn-devenv/armnn/build
CXX=aarch64-linux-android<Android_API>-clang++ \
CC=aarch64-linux-android<Android_API>-clang \
CXX_FLAGS="-fPIE -fPIC" \
cmake .. \
    -DCMAKE_ANDROID_NDK=$NDK \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=<Android_API> \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_EXE_LINKER_FLAGS="-pie -llog -lz" \
    -DARMCOMPUTE_ROOT=$HOME/armnn-devenv/ComputeLibrary/ \
    -DARMCOMPUTE_BUILD_DIR=$HOME/armnn-devenv/ComputeLibrary/build \
    -DBOOST_ROOT=$HOME/armnn-devenv/boost/install/ \
    -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DARMNNREF=1 \
    -DPROTOBUF_ROOT=$HOME/armnn-devenv/google/arm64_pb_install/
```

To include standalone sample dynamic backend tests, add the argument to enable the tests and the dynamic backend path to the CMake command:

```bash
-DSAMPLE_DYNAMIC_BACKEND=1 \
-DDYNAMIC_BACKEND_PATHS=$SAMPLE_DYNAMIC_BACKEND_PATH
# Where $SAMPLE_DYNAMIC_BACKEND_PATH is the path where libArm_SampleDynamic_backend.so library file is pushed
```

 * Run the build
```bash
make -j16
```

## Build Standalone Sample Dynamic Backend
* The sample dynamic backend is located in armnn/src/dynamic/sample
```bash
mkdir build
cd build
```

* Use CMake to configure the build environment, update the following script and run it from the armnn/src/dynamic/sample/build directory to set up the Arm NN build:
```bash
#!/bin/bash
CXX=aarch64-linux-android<Android_API>-clang++ \
CC=aarch64-linux-android<Android_API>-clang \
CXX_FLAGS="-fPIE -fPIC" \
cmake \
-DCMAKE_SYSTEM_NAME=Android \
-DCMAKE_CXX_FLAGS=--std=c++14 \
-DCMAKE_EXE_LINKER_FLAGS="-pie -llog" \
-DCMAKE_MODULE_LINKER_FLAGS="-llog" \
-DBOOST_ROOT=$HOME/armnn-devenv/boost/install \
-DBoost_SYSTEM_LIBRARY=$HOME/armnn-devenv/boost/install/lib/libboost_system.a \
-DARMNN_PATH=$HOME/armnn-devenv/armnn/build/libarmnn.so ..
```

* Run the build
```bash
make
```

## Run the Arm NN unit tests on an Android device


* Push the build results to an Android device and make symbolic links for shared libraries:
  Currently adb version we have used for testing is 1.0.41.
```bash
adb push libarmnn.so /data/local/tmp/
    adb push libtimelineDecoder.so /data/local/tmp/
adb push UnitTests /data/local/tmp/
adb push $NDK/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so /data/local/tmp/
adb push $HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so /data/local/tmp/libprotobuf.so.23.0.0
adb shell 'ln -s libprotobuf.so.23.0.0 /data/local/tmp/libprotobuf.so.23'
adb shell 'ln -s libprotobuf.so.23.0.0 /data/local/tmp/libprotobuf.so'
```

* Push the files needed for the unit tests (they are a mix of files, directories and symbolic links):
```bash
adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/testSharedObject
adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/testSharedObject/* /data/local/tmp/src/backends/backendsCommon/test/testSharedObject/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend
adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/testDynamicBackend/* /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1
adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath1/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2
adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
adb shell ln -s Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1
adb shell ln -s Arm_CpuAcc_backend.so.1 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2
adb shell ln -s Arm_CpuAcc_backend.so.1.2 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2.3
adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_GpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
adb shell ln -s nothing /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_no_backend.so

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath3

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5
adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath5/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6
adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath6/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath7

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9
adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath9/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9/

adb shell mkdir -p /data/local/tmp/src/backends/dynamic/reference
adb push -p ~/armnn-devenv/armnn/build/src/backends/dynamic/reference/Arm_CpuRef_backend.so /data/local/tmp/src/backends/dynamic/reference/

# If the standalone sample dynamic tests are enabled, also push libArm_SampleDynamic_backend.so library file to the folder specified as $SAMPLE_DYNAMIC_BACKEND_PATH when Arm NN is built.
# This is the example when $SAMPLE_DYNAMIC_BACKEND_PATH is specified as /data/local/tmp/dynamic/sample/:

adb shell mkdir -p /data/local/tmp/dynamic/sample/
adb push -p ${WORKING_DIR}/armnn/src/dynamic/sample/build/libArm_SampleDynamic_backend.so /data/local/tmp/dynamic/sample/
```

* Run Arm NN unit tests:
```bash
adb shell 'LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/vendor/lib64/egl /data/local/tmp/UnitTests'
```
If libarmnnUtils.a is present in `~/armnn-devenv/armnn/build/` and the unit tests run without failure then the build was successful.
