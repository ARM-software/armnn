#!/bin/bash
#
# Copyright © 2023-2025 Arm Ltd and Contributors. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
function Usage() {
  echo "This script builds Arm NN for Android using the Android NDK. The script builds"
  echo "the Arm NN core library and its dependencies."
  echo ""
  echo "Usage: $CMD [options]"
  echo "Options:"
  echo "    -l Use this copy of Arm NN and ComputeLibrary instead of cloning new copies"
  echo "       <1 or 0> defaults to 1"
  echo "    -a Override Arm NN branch (defaults to main branch). For example to use 23.02 release use branch name branches/armnn_23_02"
  echo "    -c Override ACL branch (defaults to main branch). For example to use 23.02 release use tag v23.02"
  echo "    -A Android API level defaults to 30"
  echo "    -n Neon (CpuAcc backend) enabled <1 or 0> defaults to 1"
  echo "    -g CL (GpuAcc backend) enabled <1 or 0> defaults to 1"
  echo "    -r Reference (CpuRef backend) enabled <1 or 0> defaults to 1"
  echo "    -u Build tests and test applications <1 or 0> defaults to 1"
  echo "    -d TfLite Delegate enabled <1 or 0> defaults to 1"
  echo "    -p TfLite Parser enabled <1 or 0> defaults to 1"
  echo "    -s Dynamic Sample enabled <1 or 0> defaults to 0"
  echo "    -i Installation directory defaults to ~/armnn-devenv"
  echo "    -t Push to board and run tests <1 or 0> defaults to 0"

  exit 1
}

function AssertZeroExitCode {
  EXITCODE=$?
  if [[ $EXITCODE -ne 0 ]]; then
    echo -e "Previous command exited with code $EXITCODE"
    exit 1
  fi
}

THIS_FILE=$(readlink -f "$0")
BASE_DIR=$(dirname "$THIS_FILE")

# Set variables and working directory
CREATE_LINKS=1
ARMNN_BRANCH=main
ACL_BRANCH=main
ACL_NEON=1
ACL_CL=1
REFERENCE=1
BUILD_TESTS=1
DELEGATE=1
TFLITE_PARSER=1
DYNAMIC_SAMPLE=0
CMAKE=$(which cmake)
WORKING_DIR=$HOME/armnn-devenv

ANDROID_API=30
PUSH_TO_BOARD=0

# Parse the command line arguments to get build type
while getopts "hl:a:c:A:n:g:r:u:d:p:s:i:t:" opt; do
  ((OPTION_COUNTER+=1))
  case "$opt" in
    h|\?) Usage;;
    l) CREATE_LINKS=$OPTARG;;
    a) ARMNN_BRANCH=$OPTARG;;
    c) ACL_BRANCH=$OPTARG;;
    A) ANDROID_API=$OPTARG;;
    n) ACL_NEON=$OPTARG;;
    g) ACL_CL=$OPTARG;;
    r) REFERENCE=$OPTARG;;
    u) BUILD_TESTS=$OPTARG;;
    d) DELEGATE=$OPTARG;;
    p) TFLITE_PARSER=$OPTARG;;
    s) DYNAMIC_SAMPLE=$OPTARG;;
    i) WORKING_DIR=$OPTARG;;
    t) PUSH_TO_BOARD=$OPTARG;;
  esac
done
shift $((OPTIND - 1))

export NDK_DIR=$WORKING_DIR/android-ndk-r26b

export NDK_TOOLCHAIN_ROOT=$NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64
export PATH=$NDK_TOOLCHAIN_ROOT/bin/:$PATH

pushd $WORKING_DIR

function GetAndroidNDK {
    cd $WORKING_DIR
    if [[ ! -d android-ndk-r26b ]]; then
        echo "+++ Getting Android NDK"
        wget -O android-ndk-r26b-linux.zip https://dl.google.com/android/repository/android-ndk-r26b-linux.zip
        unzip -o android-ndk-r26b-linux.zip
        rm -f android-ndk-r26b-linux.zip
    fi
}

function GetAndBuildCmake {
    echo "+++ Building Cmake 3.22.1"
    cd $WORKING_DIR
    sudo apt-get update -y && sudo apt-get install -y libssl-dev build-essential ca-certificates
    wget https://github.com/Kitware/CMake/releases/download/v3.22.1/cmake-3.22.1.tar.gz
    tar -zxf cmake-3.22.1.tar.gz
    pushd cmake-3.22.1
    ./bootstrap --prefix=$WORKING_DIR/cmake/install
    make all install -j16
    popd
}

function GetAndBuildFlatbuffers {
    cd $WORKING_DIR

    if [[ ! -d flatbuffers-24.3.25 ]]; then
        echo "+++ Getting Flatbuffers"
        wget https://github.com/google/flatbuffers/archive/v24.3.25.tar.gz
        tar xf v24.3.25.tar.gz
    fi
    #Build FlatBuffers
    echo "+++ Building x86 Flatbuffers library"
    cd $WORKING_DIR/flatbuffers-24.3.25

    rm -f CMakeCache.txt

    rm -rf build-x86
    mkdir build-x86
    cd build-x86

    rm -rf $WORKING_DIR/flatbuffers-x86
    mkdir $WORKING_DIR/flatbuffers-x86

    CXXFLAGS="-fPIC" $CMAKE .. \
          -DFLATBUFFERS_BUILD_FLATC=1 \
          -DTFLITE_HOST_TOOLS_DIR=$WORKING_DIR/flatbuffers-x86/bin \
          -DCMAKE_INSTALL_PREFIX:PATH=$WORKING_DIR/flatbuffers-x86

    make all install -j16

    echo "+++ Building Android Flatbuffers library"
    cd $WORKING_DIR/flatbuffers-24.3.25

    rm -f CMakeCache.txt

    rm -rf build-android
    mkdir -p build-android
    cd build-android

    rm -rf $WORKING_DIR/flatbuffers-android
    mkdir -p $WORKING_DIR/flatbuffers-android

    CC=/usr/bin/aarch64-linux-gnu-gcc CXX=/usr/bin/aarch64-linux-gnu-g++ \
    CXXFLAGS="-fPIC" $CMAKE .. \
          -DCMAKE_ANDROID_NDK=$NDK_DIR \
          -DCMAKE_SYSTEM_NAME=Android \
          -DCMAKE_SYSTEM_VERSION=$ANDROID_API \
          -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
          -DCMAKE_CXX_FLAGS=--std=c++14 \
          -DFLATBUFFERS_BUILD_FLATC=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -DFLATBUFFERS_BUILD_TESTS=OFF \
          -DCMAKE_INSTALL_PREFIX=$WORKING_DIR/flatbuffers-android

    make all install -j16
}

function GetArmNN {
    cd $WORKING_DIR

    if [[ ! -d armnn ]]; then
        if [[ $CREATE_LINKS = 1 ]]; then
            if [[ -d $BASE_DIR/../../armnn ]]; then
                echo "+++ Linking Arm NN"
                echo "$BASE_DIR"
               ln -s $BASE_DIR/../../armnn $WORKING_DIR/armnn
           else
               echo "Couldn't find Arm NN to link"
           fi
        else
            echo "+++ Cloning Arm NN"
            git clone https://github.com/ARM-software/armnn.git armnn
            cd armnn

            git checkout $ARMNN_BRANCH
            git log -1
       fi
    fi
}

function GetAndBuildComputeLibrary {
    cd $WORKING_DIR

    if [[ ! -d ComputeLibrary ]]; then
        if [[ $CREATE_LINKS = 1 ]]; then
            if [[ -d $BASE_DIR/../../ComputeLibrary ]]; then
                echo "+++ Linking ComputeLibrary"
                ln -s $BASE_DIR/../../ComputeLibrary $WORKING_DIR/ComputeLibrary
            else
                echo "+++ Cloning ComputeLibrary"
                git clone https://github.com/ARM-software/ComputeLibrary.git ComputeLibrary
                cd ComputeLibrary
                git checkout $($BASE_DIR/../../armnn/scripts/get_compute_library.sh -p)
                git log -1
            fi
        else
            echo "+++ Cloning ComputeLibrary"

            git clone https://github.com/ARM-software/ComputeLibrary.git ComputeLibrary
            cd ComputeLibrary
            git checkout $ACL_BRANCH
            git log -1
        fi
    fi
    cd $WORKING_DIR/ComputeLibrary

    echo "+++ Building Compute Library"
    scons toolchain_prefix=llvm- compiler_prefix=aarch64-linux-android$ANDROID_API- arch=arm64-v8a neon=$ACL_NEON opencl=$ACL_CL embed_kernels=$ACL_CL extra_cxx_flags="-fPIC" \
    benchmark_tests=0 validation_tests=0 os=android -j16
}

function GetAndBuildTFLite {
    TENSORFLOW_REVISION="tags/v2.19.0" # TF 2.19.0
    TFLITE_ROOT_DIR=${WORKING_DIR}/tensorflow/tensorflow/lite

    cd $WORKING_DIR

    if [[ ! -d tensorflow ]]; then
        if [[ -d $BASE_DIR/../../armnn ]]; then
            TENSORFLOW_REVISION=$($BASE_DIR/../../armnn/scripts/get_tensorflow.sh -p)
        fi
        echo "+++ Cloning TensorFlow"
        git clone https://github.com/tensorflow/tensorflow.git
        AssertZeroExitCode "Cloning TensorFlow failed"

        cd tensorflow

        echo "Checking out ${TENSORFLOW_REVISION}"
        git fetch && git checkout $TENSORFLOW_REVISION

        cd $WORKING_DIR
    fi

    # Ensure we use a known host protoc and headers to generate any protos
    HOST_PROTOC="$WORKING_DIR/protobuf-host/bin/protoc"
    PROTOBUF_HOST_INCLUDE="$WORKING_DIR/protobuf-host/include"
    if [[ ! -x "$HOST_PROTOC" ]]; then
        echo "Error: Host protoc not found at $HOST_PROTOC. Build Protobuf first." >&2
        exit 1
    fi

    CMARGS="-DTFLITE_ENABLE_XNNPACK=OFF \
            -DFLATBUFFERS_BUILD_FLATC=OFF \
            -DBUILD_SHARED_LIBS=OFF \
            -DBUILD_TESTING=OFF \
            -DTFLITE_BUILD_EXAMPLES=OFF \
            -DTFLITE_ENABLE_PROFILING=OFF \
            -DProtobuf_PROTOC_EXECUTABLE=$HOST_PROTOC \
            -DProtobuf_INCLUDE_DIRS=$PROTOBUF_HOST_INCLUDE \
            -DTFLITE_HOST_TOOLS_DIR=$WORKING_DIR/flatbuffers-x86/bin "

    # Two different naming conventions; one for build and the other for CC_OPT_FLAGS
    ANDROID_ARM_ARCH="arm64-v8a"

    # Clean previous build to avoid stale generated sources from older protoc
    rm -rf tflite-out/android
    mkdir -p tflite-out/android
    cd tflite-out/android

    echo "*** Configure and Cross-Compile TfLite for ${TARGET_MACHINE} with architecture ${ANDROID_ARM_ARCH}"
    echo "*** Outputting files to ${TFLITE_OUTPUT_DIR}/${TARGET_MACHINE}"

    CMARGS="$CMARGS -DCMAKE_TOOLCHAIN_FILE=$NDK_DIR/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=$ANDROID_ARM_ARCH \
        -DANDROID_PLATFORM=$ANDROID_API" \

    $CMAKE $CMARGS $TFLITE_ROOT_DIR
    AssertZeroExitCode "Failed to configure Tensorflow Lite from source"

    cd $WORKING_DIR

    $CMAKE --build tflite-out/android -j 16
    AssertZeroExitCode "Failed to build Tensorflow Lite from source"

    mkdir -p $WORKING_DIR/tflite-out/tensorflow/tensorflow/lite/schema

    # TensorFlow >= 2.14 moved schema to compiler/mlir/lite/schema
    SCHEMA_LOCATION_NEW=$WORKING_DIR/tensorflow/tensorflow/compiler/mlir/lite/schema/schema.fbs
    SCHEMA_LOCATION_OLD=$WORKING_DIR/tensorflow/tensorflow/lite/schema/schema.fbs
    if [[ -f "$SCHEMA_LOCATION_NEW" ]]; then
        SCHEMA_LOCATION="$SCHEMA_LOCATION_NEW"
    elif [[ -f "$SCHEMA_LOCATION_OLD" ]]; then
        SCHEMA_LOCATION="$SCHEMA_LOCATION_OLD"
    else
        echo "TensorFlow schema.fbs not found in expected locations." >&2
        echo "Tried: $SCHEMA_LOCATION_NEW and $SCHEMA_LOCATION_OLD" >&2
        exit 1
    fi

    cp "$SCHEMA_LOCATION" $WORKING_DIR/tflite-out/tensorflow/tensorflow/lite/schema

    cd $WORKING_DIR/tflite-out/tensorflow/tensorflow/lite/schema
    $WORKING_DIR/flatbuffers-x86/bin/flatc -c --gen-object-api --reflect-types --reflect-names schema.fbs
    AssertZeroExitCode "Failed to generate C++ schema from $SCHEMA_LOCATION"
}

function GetPrebuiltProtoc {
    # Download a prebuilt protoc binary + headers to speed up iteration
    # Set PROTOC_VERSION to override; defaults to 21.9 to align with build-tool flow
    local ver=${PROTOC_VERSION:-21.9}
    local url="https://github.com/protocolbuffers/protobuf/releases/download/v${ver}/protoc-${ver}-linux-x86_64.zip"
    local outzip="$WORKING_DIR/protoc-${ver}-linux-x86_64.zip"
    local dest="$WORKING_DIR/protobuf-host"

    echo "+++ Downloading prebuilt protoc ${ver}"
    mkdir -p "$dest"
    wget -O "$outzip" "$url"
    unzip -o "$outzip" -d "$dest"
    rm -f "$outzip"
    "$dest/bin/protoc" --version || true
}

function BuildArmNN {
    echo "+++ Building Arm NN"

    rm -rf $WORKING_DIR/armnn/build

    mkdir -p $WORKING_DIR/armnn/build
    cd $WORKING_DIR/armnn/build

    CMARGS="-DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_ANDROID_NDK=$NDK_DIR \
            -DNDK_VERSION=r26b \
            -DCMAKE_SYSTEM_NAME=Android \
            -DCMAKE_SYSTEM_VERSION=$ANDROID_API \
            -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
            -DCMAKE_SYSROOT=$WORKING_DIR/android-ndk-r26b/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
            -DARMCOMPUTE_ROOT=$WORKING_DIR/ComputeLibrary \
            -DARMCOMPUTE_BUILD_DIR=$WORKING_DIR/ComputeLibrary/build \
            -DARMCOMPUTENEON=$ACL_NEON -DARMCOMPUTECL=$ACL_CL -DARMNNREF=$REFERENCE \
            -DFLATBUFFERS_INCLUDE_PATH=$WORKING_DIR/flatbuffers-x86/include \
            -DFLATBUFFERS_ROOT=$WORKING_DIR/flatbuffers-android \
            -DFLATC_DIR=$WORKING_DIR/flatbuffers-x86 \
            -DBUILD_UNIT_TESTS=$BUILD_TESTS \
            -DBUILD_TESTS=$BUILD_TESTS "

    if [[ $TFLITE_PARSER == 1 ]]; then
        CMARGS="$CMARGS \
            -DBUILD_TF_LITE_PARSER=1 \
            -DTF_LITE_GENERATED_PATH=$WORKING_DIR/tflite-out/tensorflow/tensorflow/lite/schema \
            -DTENSORFLOW_ROOT=$WORKING_DIR/tensorflow \
            -DTFLITE_LIB_ROOT=$WORKING_DIR/tflite-out/android"
    fi

    if [[ $DELEGATE == 1 ]]; then
        CMARGS="$CMARGS \
            -DBUILD_ARMNN_TFLITE_DELEGATE=1 \
            -DTENSORFLOW_ROOT=$WORKING_DIR/tensorflow \
            -DTFLITE_LIB_ROOT=$WORKING_DIR/tflite-out/android \
            -DTFLITE_ROOT_DIR=$WORKING_DIR/tensorflow/tensorflow/lite \
            -DBUILD_DELEGATE_JNI_INTERFACE=0"
    fi

    if [[ $DYNAMIC_SAMPLE == 1 ]]; then
        DYNAMIC_SAMPLE_PATH="/data/local/tmp/dynamic/sample"
        CMARGS="$CMARGS \
                -DDYNAMIC_BACKEND_PATHS=$DYNAMIC_SAMPLE_PATH \
                -DSAMPLE_DYNAMIC_BACKEND=1"
    fi
    echo "args"
    echo $CMARGS
    CXX=aarch64-linux-android$ANDROID_API-clang++ \
    CC=aarch64-linux-android$ANDROID_API-clang \
    CXX_FLAGS="-fPIE -fPIC" \
    $CMAKE $CMARGS ..
    make -j16
}

function BuildStandaloneDynamicBackend {
    echo "+++ Building Standalone Dynamic Sample Backend"
    cd $WORKING_DIR/armnn/src/dynamic/sample
    BUILD_DIR=build
    rm -rf $BUILD_DIR
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    CXX=aarch64-linux-android$ANDROID_API-clang++ \
    CC=aarch64-linux-android$ANDROID_API-clang \
    CXX_FLAGS="-fPIE -fPIC" \
    $CMAKE \
    -DCMAKE_C_COMPILER_WORKS=TRUE \
    -DCMAKE_CXX_COMPILER_WORKS=TRUE \
    -DCMAKE_ANDROID_NDK=$NDK_DIR \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=$ANDROID_API \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_SYSROOT=$WORKING_DIR/android-ndk-r26b/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
    -DCMAKE_CXX_FLAGS=--std=c++14 \
    -DCMAKE_EXE_LINKER_FLAGS="-pie -llog" \
    -DCMAKE_MODULE_LINKER_FLAGS="-llog" \
    -DARMNN_PATH=$WORKING_DIR/armnn/build/libarmnn.so ..
    make
}

# push sources to board
function PushBuildSourcesToBoard {
    cd $WORKING_DIR/armnn/build
    adb start-server
    adb shell rm -rf /data/local/tmp/*
    echo "+++ Pushing sources to board"
    adb root
    adb remount
    sleep 10s
    adb version
    adb push libarmnn.so /data/local/tmp/
    adb push libtimelineDecoder.so /data/local/tmp/
    adb push libtimelineDecoderJson.so /data/local/tmp/
    adb push GatordMock /data/local/tmp/
    adb push libarmnnBasePipeServer.so /data/local/tmp/
    adb push libarmnnTestUtils.so /data/local/tmp/
    adb push libarmnnTfLiteParser.so /data/local/tmp/
    adb push UnitTests /data/local/tmp/
    if [[ $DELEGATE == 1 ]]; then
        adb push ${WORKING_DIR}/armnn/build/delegate/DelegateUnitTests /data/local/tmp/
        adb push ${WORKING_DIR}/armnn/build/delegate/libarmnnDelegate.so /data/local/tmp/
    fi
    #adb push $NDK_DIR/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so /data/local/tmp/
    adb push $NDK_DIR/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/x86_64-linux-android/libc++_shared.so /data/local/tmp/
    echo "+++ Pushing test files to board"
    adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/testSharedObject
    adb push -p ${WORKING_DIR}/armnn/build/src/backends/backendsCommon/test/testSharedObject/* /data/local/tmp/src/backends/backendsCommon/test/testSharedObject/
    adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend
    adb push -p ${WORKING_DIR}/armnn/build/src/backends/backendsCommon/test/testDynamicBackend/* /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend/
    adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1
    adb push -p ${WORKING_DIR}/armnn/build/src/backends/backendsCommon/test/backendsTestPath1/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1/
    adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2
    adb push -p ${WORKING_DIR}/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
    adb shell 'ln -s Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1'
    adb shell 'ln -s Arm_CpuAcc_backend.so.1 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2'
    adb shell 'ln -s Arm_CpuAcc_backend.so.1.2 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2.3'
    adb push -p ${WORKING_DIR}/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_GpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
    adb shell 'ln -s nothing /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_no_backend.so'
    adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath3
    adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5
    adb push -p ${WORKING_DIR}/armnn/build/src/backends/backendsCommon/test/backendsTestPath5/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5/
    adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6
    adb push -p ${WORKING_DIR}/armnn/build/src/backends/backendsCommon/test/backendsTestPath6/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6/
    adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath7
    adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9
    adb push -p ${WORKING_DIR}/armnn/build/src/backends/backendsCommon/test/backendsTestPath9/* /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9/
    adb shell mkdir -p /data/local/tmp/src/backends/dynamic/reference
    adb push -p ${WORKING_DIR}/armnn/build/src/backends/dynamic/reference/Arm_CpuRef_backend.so /data/local/tmp/src/backends/dynamic/reference/
    if [[ $DYNAMIC_SAMPLE == 1 ]]; then
        adb shell mkdir -p /data/local/tmp/dynamic/sample/
        adb push -p ${WORKING_DIR}/armnn/src/dynamic/sample/build/libArm_SampleDynamic_backend.so /data/local/tmp/dynamic/sample/
    fi
    echo "+++ Running UnitTests"
    adb shell LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/vendor/lib64/egl /data/local/tmp/UnitTests ; printf $?
    if [[ $DELEGATE == 1 ]]; then
        adb shell LD_LIBRARY_PATH=/data/local/tmp:/vendor/lib64:/vendor/lib64/egl /data/local/tmp/DelegateUnitTests ; printf $?
    fi
}

# Cleanup any previous runs, setup clean directories
echo "+++ Creating $WORKING_DIR directory"
mkdir -p $WORKING_DIR
AssertZeroExitCode "Creating $WORKING_DIR directory failed"

GetAndroidNDK
if [[ $? != 0 ]] ; then
    echo "Downloading Android NDK failed"
    exit 1
fi

GetAndBuildCmake

CMAKE=$WORKING_DIR/cmake/install/bin/cmake
GetAndBuildFlatbuffers
if [[ $? != 0 ]] ; then
    echo "Building Flatbuffers failed"
    exit 1
fi
GetArmNN
if [[ $? != 0 ]] ; then
    echo "Cloning Arm NN failed"
    exit 1
fi

# Build TFLite if the Delegate or Parser is required
if [[ "$DELEGATE" -eq 1 || "$TFLITE_PARSER" -eq 1 ]]; then
    # Faster path: use prebuilt protoc instead of rebuilding
    GetPrebuiltProtoc
    GetAndBuildTFLite
fi

if [[ $? != 0 ]] ; then
    echo "Building tflite failed"
    exit 1
fi
GetAndBuildComputeLibrary
if [[ $? != 0 ]] ; then
    echo "Building ComputeLibrary failed"
    exit 1
fi
BuildArmNN
if [[ $? != 0 ]] ; then
    echo "Building Arm NN failed"
    exit 1
fi
if [[ $DYNAMIC_SAMPLE == 1 ]]; then
  BuildStandaloneDynamicBackend
fi
if [[ $PUSH_TO_BOARD == 1 ]]; then
  PushBuildSourcesToBoard
fi
if [[ "$R_new" -eq 0 ]]; then
    echo "Success!"
else
    echo "Failed to run UnitTests"
    exit 1
fi
