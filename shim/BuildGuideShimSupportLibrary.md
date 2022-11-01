# How to use the Android NDK to build Arm NN

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Download Arm NN](#download-arm-nn)
- [Build Arm Compute Library](#build-arm-compute-library)
- [Build Arm NN](#build-arm-nn)
- [Build Arm NN Support Library](#build-arm-nn-support-library)
- [Build Arm NN Shim](#build-arm-nn-shim)


## Introduction
These are step by step instructions for building the Arm NN shim and support library for NNAPI.
This work is currently in an experimental phase.

## Prerequisites

The following are required to build the Arm NN support library
* Android NDK r25
  * Detailed setup can be found in [BuildGuideAndroidNDK.md](../BuildGuideAndroidNDK.md)
* Flatbuffer version 2.0.6
  * Detailed setup can be found in [BuildGuideCrossCompilation.md](../BuildGuideCrossCompilation.md)

The following is required to build the Arm NN shim
* AOSP Source (Android Open Source Project)
  * Download the source from the [official website](https://source.android.com/setup/build/downloading)
  * This guide will use release tag `android12-s1-release`


Set environment variables
```bash
export WORKING_DIR=<path to where the Arm NN source code, clframework and aosp repos will be cloned>
export AOSP_ROOT=<path to the root of Android tree where the shim will be built>
export AOSP_MODULES_ROOT=<path to where AOSP modules will be cloned i.e. $WORKING_DIR/aosp>
export ARMNN_BUILD_DIR=<path to the Arm NN build directory i.e. $WORKING_DIR/build>
export NDK=<path to>android-ndk-r25
export NDK_TOOLCHAIN_ROOT=$NDK/toolchains/llvm/prebuilt/linux-x86_64
export PATH=$NDK_TOOLCHAIN_ROOT/bin/:$PATH
export FLATBUFFERS_ANDROID_BUILD=<path to flatbuffers target android build>
export FLATBUFFERS_X86_BUILD=<path to flatbuffers host build-x86_64>
```

## Download Arm NN
If the user only wishes to build the Support Library with the NDK, the Arm NN repo can be cloned into any folder.
If the user also wishes to build the Arm NN Shim, for this the Arm NN repo will need to reside within
the Android tree in order for armnn/shim/Android.bp to be picked up by the Soong (Android) build system.
For example $AOSP_ROOT/vendor/arm/armnn


* Clone Arm NN:
  (Requires Git if not previously installed: `sudo apt install git`)

```bash
cd $WORKING_DIR
git clone https://github.com/ARM-software/armnn.git
```

## Build Arm Compute Library

Arm NN provides a script that downloads the version of Arm Compute Library that Arm NN was tested with:
```bash
${WORKING_DIR}/armnn/scripts/get_compute_library.sh
```
* Build the Arm Compute Library:
  (Requires SCons if not previously installed: `sudo apt install scons`)
```bash
cd ${WORKING_DIR}/clframework

scons arch=arm64-v8a \
toolchain_prefix=aarch64-linux-android- \
compiler_prefix=aarch64-linux-android29- \
neon=1 opencl=1 \
embed_kernels=1 \
build_dir=android-arm64v8a \
extra_cxx_flags="-Wno-parentheses-equality -Wno-missing-braces -fPIC" \
Werror=0 embed_kernels=1 examples=0 \
validation_tests=0 benchmark_tests=0 benchmark_examples=0 os=android -j16
```

## Build Arm NN and Serializer

* Build Arm NN:
  (Requires CMake if not previously installed: `sudo apt install cmake`)
```bash
cd $ARMNN_BUILD_DIR
CXX=aarch64-linux-android29-clang++ \
CC=aarch64-linux-android29-clang \
CXX_FLAGS="-fPIE -fPIC" cmake ${WORKING_DIR}/armnn \
-DCMAKE_ANDROID_NDK=$NDK \
-DCMAKE_SYSTEM_NAME=Android \
-DCMAKE_SYSTEM_VERSION=29 \
-DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
-DCMAKE_EXE_LINKER_FLAGS="-pie -llog -lz" \
-DARMCOMPUTE_ROOT=$WORKING_DIR/clframework/ \
-DARMCOMPUTE_BUILD_DIR=$WORKING_DIR/clframework/build/android-arm64v8a/ \
-DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DARMNNREF=1 \
-DFLATBUFFERS_ROOT=$FLATBUFFERS_ANDROID_BUILD \
-DFLATC_DIR=$FLATBUFFERS_X86_BUILD \
-DBUILD_ARMNN_SERIALIZER=1 -DBUILD_GATORD_MOCK=0 -DBUILD_BASE_PIPE_SERVER=0
```

 * Run the build
```bash
make -j16
```

## Build Arm NN Support Library

Building the support library requires building some AOSP libraries via the NDK. 
It should be possible to use $AOSP_ROOT instead of $AOSP_MODULES_ROOT.

However this example will instead clone the necessary AOSP repos outside of the Android tree and apply some minor patches
which were required to get it to build with the Android version used in this guide.

```bash
# Call a script which will clone the necessary AOSP repos (do not clone them into Android tree)
${WORKING_DIR}/armnn/shim/sl/scripts/clone_aosp_libs.sh $AOSP_MODULES_ROOT

# Modify the repos by applying patches
${WORKING_DIR}/armnn/shim/sl/scripts/modify_aosp_libs.sh $AOSP_MODULES_ROOT

# Build the Support Library
CMARGS="$CMARGS \
-DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
-DANDROID_ABI=arm64-v8a \
-DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
-DCMAKE_ANDROID_NDK=$NDK \
-DANDROID_PLATFORM=android-29 \
-DAOSP_MODULES_ROOT=$AOSP_MODULES_ROOT \
-DARMNN_SOURCE_DIR=$WORKING_DIR/armnn \
-DArmnn_DIR=$ARMNN_BUILD_DIR "

mkdir ${WORKING_DIR}/armnn/shim/sl/build
cd ${WORKING_DIR}/armnn/shim/sl/build

CXX=aarch64-linux-android29-clang++ \
CC=aarch64-linux-android29-clang \
cmake $CMARGS ../
make
```

## Build Arm NN Shim

By default the Arm NN shim Android.bp.off is not enabled.
It is enabled below by renaming it to Android.bp

```bash
cd ${WORKING_DIR}/armnn/shim
mv Android.bp.off Android.bp

cd $AOSP_ROOT
source build/envsetup.sh
lunch <device>-eng
cd vendor/arm/armnn/shim
export ARMNN_ANDROID_MK_ENABLE=0
mm
```

The built libraries and manifest file can be found here:
$AOSP_ROOT/out/target/product/<device>/vendor/lib64/libarmnn_support_library.so
$AOSP_ROOT/out/target/product/<device>/vendor/bin/hw/android.hardware.neuralnetworks-shim-service-armnn
$AOSP_ROOT/out/target/product/<device>/vendor/etc/vintf/manifest/android.hardware.neuralnetworks-shim-service-armnn.xml

Currently the Arm NN libraries are shared libraries and therefore will need to be pushed to the device:
$ARMNN_BUILD_DIR/libarmnn.so
