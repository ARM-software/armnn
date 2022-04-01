# How to use the Android NDK to build Arm NN

- [Introduction](#introduction)
- [Download the Android NDK and make a standalone toolchain](#download-the-android-ndk-and-make-a-standalone-toolchain)
- [Build Google's Protobuf library](#build-google-s-protobuf-library)
- [Download Arm NN](#download-arm-nn)
- [Build Arm Compute Library](#build-arm-compute-library)
- [Build Arm NN](#build-arm-nn)
- [Build Standalone Sample Dynamic Backend](#build-standalone-sample-dynamic-backend)
- [Run the Arm NN unit tests on an Android device](#run-the-armnn-unit-tests-on-an-android-device)


## Introduction
These are step by step instructions for using the Android NDK to build Arm NN.
They have been tested on a clean install of Ubuntu 18.04 and 20.04, and should also work with other OS versions.
The instructions show how to build the Arm NN core library.
Building protobuf is optional. We have given steps should the user wish to build it (i.e. as an Onnx dependency).
All downloaded or generated files will be saved inside the `$HOME/armnn-devenv` directory.

## Download the Android NDK and make a standalone toolchain

* Download the Android NDK from [the official website](https://developer.android.com/ndk/downloads/index.html):
 ```bash
 mkdir -p $HOME/armnn-devenv/
 cd $HOME/armnn-devenv/
 # For Mac OS, change the NDK download link accordingly.
 wget https://dl.google.com/android/repository/android-ndk-r20b-linux-x86_64.zip
 unzip android-ndk-r20b-linux-x86_64.zip
 export NDK=$HOME/armnn-devenv/android-ndk-r20b
 export NDK_TOOLCHAIN_ROOT=$NDK/toolchains/llvm/prebuilt/linux-x86_64
 export PATH=$NDK_TOOLCHAIN_ROOT/bin/:$PATH

 # You may want to append the above export variables commands to your `~/.bashrc` (or `~/.bash_profile` in Mac OS).
 ```

* With the android ndk-20b, you don't need to use the make_standalone_toolchain script to create a toolchain for a specific version of android. Android's current preference is for you to just specify the architecture and operating system while setting the compiler and just use the ndk directory.

## Build Google's Protobuf library (Optional)

* Clone protobuf: 
  (Requires Git if not previously installed: `sudo apt install git`)
```bash
mkdir $HOME/armnn-devenv/google
cd $HOME/armnn-devenv/google
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

Note: The ANDROID_API variable should be set to the Android API version number you are using. E.g. "30" for Android R.

## Download Arm NN
* Clone Arm NN: 
  (Requires Git if not previously installed: `sudo apt install git`)

```bash
cd $HOME/armnn-devenv
git clone https://github.com/ARM-software/armnn.git
```

* Checkout Arm NN branch:
```bash
cd armnn
git checkout <branch_name>
git pull
```

For example, if you want to check out the 21.11 release branch:
```bash
git checkout branches/armnn_21_11
git pull
```

## Build Arm Compute Library
* Clone Arm Compute Library:

```bash
cd $HOME/armnn-devenv
git clone https://github.com/ARM-software/ComputeLibrary.git
```
* Checkout Arm Compute Library release tag:
```bash
cd ComputeLibrary
git checkout <tag_name>
```
For example, if you want to checkout the 21.11 release tag:
```bash
git checkout v21.11
```

Arm NN and Arm Compute Library are developed closely together. If you would like to use a particular release of Arm NN you will need the same release tag of ACL too.

Arm NN provides a script that downloads the version of Arm Compute Library that Arm NN was tested with:
```bash
git checkout $(../armnn/scripts/get_compute_library.sh -p) 
```
* the Arm Compute Library: 
  (Requires SCons if not previously installed: `sudo apt install scons`)
```bash
scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" \
 benchmark_tests=0 validation_tests=0 os=android -j16
```

## Build Arm NN

* Build Arm NN:
  (Requires CMake if not previously installed: `sudo apt install cmake`)
```bash
mkdir $HOME/armnn-devenv/armnn/build
cd $HOME/armnn-devenv/armnn/build
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
-DCMAKE_C_COMPILER_WORKS=TRUE \
-DCMAKE_CXX_COMPILER_WORKS=TRUE \
-DCMAKE_ANDROID_NDK=$NDK \
-DCMAKE_SYSTEM_NAME=Android \
-DCMAKE_SYSTEM_VERSION=$ANDROID_API \
-DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
-DCMAKE_SYSROOT=$HOME/armnn-devenv/android-ndk-r20b/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
-DCMAKE_CXX_FLAGS=--std=c++14 \
-DCMAKE_EXE_LINKER_FLAGS="-pie -llog" \
-DCMAKE_MODULE_LINKER_FLAGS="-llog" \
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
adb push -p $HOME/armnn-devenv/armnn/build/src/backends/backendsCommon/test/testSharedObject/* /data/local/tmp/src/backends/backendsCommon/test/testSharedObject/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend
adb push -p $HOME/armnn-devenv/armnn/build/src/backends/backendsCommon/test/testDynamicBackend/* /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1
adb push -p $HOME/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath1/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2
adb push -p $HOME/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
adb shell ln -s Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1
adb shell ln -s Arm_CpuAcc_backend.so.1 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2
adb shell ln -s Arm_CpuAcc_backend.so.1.2 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2.3
adb push -p $HOME/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_GpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
adb shell ln -s nothing /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_no_backend.so

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath3

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5
adb push -p $HOME/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath5/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6
adb push -p $HOME/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath6/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6/

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath7

adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9
adb push -p $HOME/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath9/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9/

adb shell mkdir -p /data/local/tmp/src/backends/dynamic/reference
adb push -p $HOME/armnn-devenv/armnn/build/src/backends/dynamic/reference/Arm_CpuRef_backend.so /data/local/tmp/src/backends/dynamic/reference/

# If the standalone sample dynamic tests are enabled, also push libArm_SampleDynamic_backend.so library file to the folder specified as $SAMPLE_DYNAMIC_BACKEND_PATH when Arm NN is built.
# This is the example when $SAMPLE_DYNAMIC_BACKEND_PATH is specified as /data/local/tmp/dynamic/sample/:

adb shell mkdir -p /data/local/tmp/dynamic/sample/
adb push -p $HOME/armnn-devenv/armnn/src/dynamic/sample/build/libArm_SampleDynamic_backend.so /data/local/tmp/dynamic/sample/
```

* Run Arm NN unit tests:
```bash
adb shell 'LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/vendor/lib64/egl /data/local/tmp/UnitTests'
```
If libarmnnUtils.a is present in `$HOME/armnn-devenv/armnn/build/` and the unit tests run without failure then the build was successful.
