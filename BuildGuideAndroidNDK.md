# How to use the Android NDK to build Arm NN

- [Introduction](#introduction)
- [Initial Setup](#initial-setup)
- [Download the Android NDK and make a standalone toolchain](#download-the-android-ndk-and-make-a-standalone-toolchain)
- [Install Cmake](#install-cmake)
- [Build Flatbuffers](#build-flatbuffers)
- [Download Arm NN](#download-arm-nn)
- [Get And Build TFLite](#get-and-build-tflite)
- [Build Arm Compute Library](#build-arm-compute-library)
- [Build Arm NN](#build-arm-nn)
- [Build Standalone Sample Dynamic Backend](#build-standalone-sample-dynamic-backend)
- [Run the Arm NN unit tests on an Android device](#run-the-arm-nn-unit-tests-on-an-android-device)


## Introduction
These are step-by-step instructions for using the Android NDK to build Arm NN.
They have been tested on a clean installation of Ubuntu 18.04 and 20.04, and should also work with other OS versions.
The instructions show how to build the Arm NN core library and its dependencies.

For ease of use there is a shell script version of this guide located in the scripts directory called [build_android_ndk_guide.sh](scripts/build_android_ndk_guide.sh). 
Run the script with a -h flag to see the command line parameters.

## Initial Setup

First, we need to specify the Android version and the directories you want to build armnn in and to install some applications required to build Arm NN and its dependencies.

```bash
export ANDROID_API=30
export WORKING_DIR=$HOME/armnn-devenv
export NDK_DIR=$WORKING_DIR/android-ndk-r25
export NDK_TOOLCHAIN_ROOT=$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64
export PATH=$NDK_TOOLCHAIN_ROOT/bin/:$PATH
```
You may want to append the above export variables commands to your `~/.bashrc` (or `~/.bash_profile` in macOS).

The ANDROID_API variable should be set to the Android API version number you are using. For example, "30" for Android R. The WORKING_DIR can be any directory you have write permissions to.

### Required Applications
Git is required to obtain Arm NN. If this has not been already installed then install it using: 
```bash
sudo apt install git
```

Arm Compute Library requires SCons. If this has not been already installed then install it using:
```bash
sudo apt install scons
```

CMake is required to build Arm NN and its dependencies. If this has not been already installed then install it using:
```bash
sudo apt install cmake
```

## Download the Android NDK and make a standalone toolchain

Download the Android NDK from [the official website](https://developer.android.com/ndk/downloads/index.html):
```bash
mkdir -p $WORKING_DIR
cd $WORKING_DIR
# For Mac OS, change the NDK download link accordingly.
wget https://dl.google.com/android/repository/android-ndk-r25-linux.zip
unzip android-ndk-r25-linux.zip
```
With Android NDK-25, you no longer need to use the make_standalone_toolchain script to create a toolchain for a specific version of Android. Android's current preference is for you to just specify the architecture and operating system while setting the compiler and just use the ndk directory.

## Install Cmake
Cmake 3.19rc3 or later is required to build Arm NN. If you are using Ubuntu 20.04 the command given in [Initial Setup](#initial-setup) should install a usable version. If you're using Ubuntu 18.04 you may need to compile cmake yourself. 

```bash
cd $WORKING_DIR
sudo apt-get install libssl-dev
wget https://github.com/Kitware/CMake/releases/download/v3.19.0-rc3/cmake-3.19.0-rc3.tar.gz
tar -zxvf cmake-3.19.0-rc3.tar.gz
cd cmake-3.19.0-rc3
./bootstrap --prefix=$WORKING_DIR/cmake/install
make all install
cd..
```

## Build Flatbuffers

Download Flatbuffers:
```bash
cd $WORKING_DIR
wget https://github.com/google/flatbuffers/archive/v2.0.6.tar.gz
tar xf v2.0.6.tar.gz
```

Build Flatbuffers for x86:
```bash
cd $WORKING_DIR/flatbuffers-2.0.6

rm -f CMakeCache.txt

rm -rf build-x86
mkdir build-x86
cd build-x86

rm -rf $WORKING_DIR/flatbuffers-x86
mkdir $WORKING_DIR/flatbuffers-x86

CXXFLAGS="-fPIC" $CMAKE .. \
    -DFLATBUFFERS_BUILD_FLATC=1 \
    -DCMAKE_INSTALL_PREFIX:PATH=$WORKING_DIR/flatbuffers-x86

make all install -j16
```
Note: -fPIC is added to allow users to use the libraries in shared objects.

Build Flatbuffers for Android:
```bash
cd $WORKING_DIR/flatbuffers-2.0.6

rm -f CMakeCache.txt

rm -rf build-android
mkdir build-android
cd build-android

rm -rf $WORKING_DIR/flatbuffers-android
mkdir $WORKING_DIR/flatbuffers-android

CC=/usr/bin/aarch64-linux-gnu-gcc CXX=/usr/bin/aarch64-linux-gnu-g++ \
CXXFLAGS="-fPIC" \
cmake .. \
    -DCMAKE_ANDROID_NDK=$NDK_DIR \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=27 \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_CXX_FLAGS=--std=c++14 \
    -DFLATBUFFERS_BUILD_FLATC=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLATBUFFERS_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=$WORKING_DIR/flatbuffers-android

make all install -j16
```

## Download Arm NN
Clone Arm NN: 

```bash
cd $WORKING_DIR
git clone https://github.com/ARM-software/armnn.git
```

Checkout the Arm NN branch:
```bash
cd armnn
git checkout <branch_name>
git pull
```

For example, if you want to check out the 23.02 release branch:
```bash
cd armnn
git checkout branches/armnn_23_02
git pull
```

## Get And Build TFLite
This optional step is only required if you intend to build the TFLite delegate or parser for Arm NN. 

First clone tensorflow:
```bash
cd $WORKING_DIR
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git fetch && git checkout "tags/v2.10.0"
```
Arm NN provides a script that downloads the version of Tensorflow that Arm NN was tested with:
```bash
git fetch && git checkout $(../armnn/scripts/get_tensorflow.sh -p)
```
Next build Tensorflow Lite:
```bash
cd $WORKING_DIR
mkdir -p tflite-out/android
cd tflite-out/android

CMARGS="-DCMAKE_TOOLCHAIN_FILE=$NDK_DIR/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=$ANDROID_API"

cmake $CMARGS $WORKING_DIR/tensorflow/tensorflow/lite

cd $WORKING_DIR
cmake --build tflite-out/android -j 16
```
Now generate the Tensorflow Lite Schema for the TFLite parser:
```bash
cd $WORKING_DIR
mkdir -p $WORKING_DIR/tflite-out/tensorflow/tensorflow/lite/schema

SCHEMA_LOCATION=$WORKING_DIR/tensorflow/tensorflow/lite/schema/schema.fbs
cp $SCHEMA_LOCATION $WORKING_DIR/tflite-out/tensorflow/tensorflow/lite/schema

cd $WORKING_DIR/tflite-out/tensorflow/tensorflow/lite/schema
$WORKING_DIR/flatbuffers-x86/bin/flatc -c --gen-object-api --reflect-types --reflect-names schema.fbs
```

## Build Arm Compute Library
Clone Arm Compute Library:

```bash
cd $WORKING_DIR
git clone https://github.com/ARM-software/ComputeLibrary.git
```
Checkout Arm Compute Library release tag:
```bash
cd ComputeLibrary
git checkout <tag_name>
```
For example, if you want to check out the 23.02 release tag:
```bash
cd ComputeLibrary
git checkout v23.02
```

Arm NN and Arm Compute Library are developed closely together. To use a particular version of Arm NN you will need a compatible version of ACL. 
Arm NN provides a script that downloads the version of Arm Compute Library that Arm NN was tested with:
```bash
git checkout $(../armnn/scripts/get_compute_library.sh -p) 
```
Build the Arm Compute Library:
```bash
scons arch=arm64-v8a os=android toolchain_prefix=llvm- compiler_prefix=aarch64-linux-android$ANDROID_API- \
    neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" \
    benchmark_tests=0 validation_tests=0 -j16
```

## Build Arm NN

Build Arm NN:

```bash
mkdir $WORKING_DIR/armnn/build
cd $WORKING_DIR/armnn/build
CXX=aarch64-linux-android$ANDROID_API-clang++ \
CC=aarch64-linux-android$ANDROID_API-clang \
CXX_FLAGS="-fPIE -fPIC" \
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_ANDROID_NDK=$NDK_DIR \
    -DNDK_VERSION=r25 \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=$ANDROID_API \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_SYSROOT=$WORKING_DIR/android-ndk-r25/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
    -DARMCOMPUTE_ROOT=$WORKING_DIR/ComputeLibrary \
    -DARMCOMPUTE_BUILD_DIR=$WORKING_DIR/ComputeLibrary/build \
    -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DARMNNREF=1 \
    -DFLATBUFFERS_INCLUDE_PATH=$WORKING_DIR/flatbuffers-x86/include \
    -DFLATBUFFERS_ROOT=$WORKING_DIR/flatbuffers-android \
    -DFLATC_DIR=$WORKING_DIR/flatbuffers-x86 \
    -DBUILD_UNIT_TESTS=1 \
    -DBUILD_TESTS=1 \
    -fexception \
```

To include the Arm NN TFLite delegate add these arguments to the above list:

```bash
    -DBUILD_ARMNN_TFLITE_DELEGATE=1 \
    -DTENSORFLOW_ROOT=$WORKING_DIR/tensorflow \
    -DTFLITE_LIB_ROOT=$WORKING_DIR/tflite-out/android \
    -DTFLITE_ROOT_DIR=$WORKING_DIR/tensorflow/tensorflow/lite \
```

To include the Arm NN TFLite Parser add these arguments to the above list:

```bash
    -DBUILD_TF_LITE_PARSER=1 \
    -DTF_LITE_GENERATED_PATH=$WORKING_DIR/tflite-out/tensorflow/tensorflow/lite/schema \
    -DTENSORFLOW_ROOT=$WORKING_DIR/tensorflow \
    -DTFLITE_LIB_ROOT=$WORKING_DIR/tflite-out/android \
```

To include standalone sample dynamic backend tests, add these arguments to enable the tests and the dynamic backend path to the CMake command:

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
This step is optional. The sample dynamic backend is located in armnn/src/dynamic/sample
```bash
mkdir build
cd build
```

* Use CMake to configure the build environment, update the following script and run it from the armnn/src/dynamic/sample/build directory to set up the Arm NN build:
```bash
#!/bin/bash
CXX=aarch64-linux-android$ANDROID_API-clang++ \
CC=aarch64-linux-android$ANDROID_API-clang \
CXX_FLAGS="-fPIE -fPIC" \
cmake \
    -DCMAKE_C_COMPILER_WORKS=TRUE \
    -DCMAKE_CXX_COMPILER_WORKS=TRUE \
    -DCMAKE_ANDROID_NDK=$NDK_DIR \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=$ANDROID_API \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_SYSROOT=$WORKING_DIR/android-ndk-r25/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
    -DCMAKE_CXX_FLAGS=--std=c++14 \
    -DCMAKE_EXE_LINKER_FLAGS="-pie -llog" \
    -DCMAKE_MODULE_LINKER_FLAGS="-llog" \
    -DARMNN_PATH=$WORKING_DIR/armnn/build/libarmnn.so ..
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
adb push $NDK_DIR/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so /data/local/tmp/
```

* Push the files needed for the unit tests (they are a mix of files, directories and symbolic links):
```bash
adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/testSharedObject
adb push -p $WORKING_DIR/armnn/build/src/backends/backendsCommon/test/testSharedObject/* /data/local/tmp/src/backends/backendsCommon/test/testSharedObject/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend
adb push -p $WORKING_DIR/armnn/build/src/backends/backendsCommon/test/testDynamicBackend/* /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1
adb push -p $WORKING_DIR/armnn/build/src/backends/backendsCommon/test/backendsTestPath1/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2
adb push -p $WORKING_DIR/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
adb shell ln -s Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1
adb shell ln -s Arm_CpuAcc_backend.so.1 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2
adb shell ln -s Arm_CpuAcc_backend.so.1.2 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2.3
adb push -p $WORKING_DIR/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_GpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
adb shell ln -s nothing /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_no_backend.so

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath3

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5
adb push -p $WORKING_DIR/armnn/build/src/backends/backendsCommon/test/backendsTestPath5/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6
adb push -p $WORKING_DIR/armnn/build/src/backends/backendsCommon/test/backendsTestPath6/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath7

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9
adb push -p $WORKING_DIR/armnn/build/src/backends/backendsCommon/test/backendsTestPath9/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9/

adb shell mkdir -p /data/local/tmp/src/backends/dynamic/reference
adb push -p $WORKING_DIR/armnn/build/src/backends/dynamic/reference/Arm_CpuRef_backend.so /data/local/tmp/src/backends/dynamic/reference/
```

If the standalone sample dynamic tests are enabled, also push libArm_SampleDynamic_backend.so library file to the folder specified as $SAMPLE_DYNAMIC_BACKEND_PATH when Arm NN is built.
This is the example when $SAMPLE_DYNAMIC_BACKEND_PATH is specified as /data/local/tmp/dynamic/sample/:

```bash
adb shell mkdir -p /data/local/tmp/dynamic/sample/
adb push -p $WORKING_DIR/armnn/src/dynamic/sample/build/libArm_SampleDynamic_backend.so /data/local/tmp/dynamic/sample/
```
If the delegate was built, push the delegate unit tests too.
```bash
adb push $WORKING_DIR/armnn/build/delegate/DelegateUnitTests /data/local/tmp/
adb push $WORKING_DIR/armnn/build/delegate/libarmnnDelegate.so /data/local/tmp/
```
Run Arm NN unit tests:
```bash
adb shell 'LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/vendor/lib64/egl /data/local/tmp/UnitTests'
```
If the delegate was built run Arm Delegate NN unit tests:
```bash
adb shell 'LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/vendor/lib64/egl /data/local/tmp/DelegateUnitTests'
```
If libarmnnUtils.a is present in `$WORKING_DIR/armnn/build/` and the unit tests run without failure then the build was successful.
