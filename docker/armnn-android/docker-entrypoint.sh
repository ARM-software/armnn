#!/bin/bash

set -e

dArmNN=/work
dComputeLib=/home/armnn-devenv/ComputeLibrary
dTensorflow=/home/armnn-devenv/google/tensorflow
dFlatBuffer=/home/armnn-devenv/flatbuffers

#Function to build ARMNN
function buildArmNN()
{
    mkdir -p ${dArmNN}/armnn-devenv && cd ${dArmNN}/armnn-devenv
    git clone https://github.com/ARM-software/armnn.git && cd armnn/
    mkdir build && cd build
    CXX=aarch64-linux-android-clang++ \
    CC=aarch64-linux-android-clang \
    CXX_FLAGS="-fPIE -fPIC" \
    cmake .. \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_ANDROID_STANDALONE_TOOLCHAIN=/home/armnn-devenv/toolchains/aarch64-android-r17b/ \
    -DCMAKE_EXE_LINKER_FLAGS="-pie -llog" \
    -DARMCOMPUTE_ROOT=/home/armnn-devenv/ComputeLibrary/ \
    -DARMCOMPUTE_BUILD_DIR=/home/armnn-devenv/ComputeLibrary/build \
    -DBOOST_ROOT=/home/armnn-devenv/boost/install/ \
    -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DARMNNREF=1 \
    -DTF_GENERATED_SOURCES=/home/armnn-devenv/google/tf_pb/ -DBUILD_TF_PARSER=1 \
    -DPROTOBUF_ROOT=/home/armnn-devenv/google/arm64_pb_install/
    make -j8
}

# Function to update Compute Lib
function updateComputeLib()
{
    pushd ${dComputeLib}
    git pull 
    scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" \
    benchmark_tests=0 validation_tests=0 os=android -j8
    echo "Compute Lib updated"
    popd
}

# Function to update FlatBuffer Lib
function updateFlatBuffer()
{
    pushd ${dFlatBuffer}
    git pull 
    cd android && cd jni && \
    rm -rf Application.mk && \
    echo "APP_STL := c++_static" >> Application.mk && \
    echo "NDK_TOOLCHAIN_VERSION" := clang >> Application.mk && \
    echo "APP_CPPFLAGS :=-std=c++11" >> Application.mk && \
    echo "APP_ABI := arm64-v8a" >> Application.mk && \
    echo "APP_PLATFORM := android-23" >> Application.mk && \
    echo "NDK_PLATFORM=android-23" >> Application.mk && \
    cd ../ && ndk-build -B
    echo "Compute Lib updated"
    popd 
}

# Main
if [ ! -d "/work/armnn-devenv/armnn/" ];
then 
    buildArmNN
fi

# Check Compute Library changes from repo
cd ${dComputeLib}
if [ $(git rev-parse HEAD) = $(git ls-remote $(git rev-parse --abbrev-ref @{u} | 
sed 's/\// /g') | cut -f1) ]
then
    echo "Compute Lib Up-to-date"
else 
    echo "New changes are availble for Compute Library repo."
    echo "Do you wanna update (y/n)?"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        updateComputeLib
    fi
fi 

# Check Tensorflow changes from repo
cd ${dTensorflow}
if [ $(git rev-parse HEAD) = $(git ls-remote $(git rev-parse --abbrev-ref @{u} | 
sed 's/\// /g') | cut -f1) ]
then
    echo "Tensrflow Lib Up-to-date"
else 
    echo "Tensrflow Lib Not Up-to-date"
    echo "Skipping for now. Issue: #267"
    #echo "New changes are availble for Compute Library repo."
    #echo "Do you wanna update (y/n)?"
    #read answer
    #if [ "$answer" != "${answer#[Yy]}" ] ;then
    # 
    #fi
fi 

# Check FlatBuffer changes from repo
cd ${dFlatBuffer}
if [ $(git rev-parse HEAD) = $(git ls-remote $(git rev-parse --abbrev-ref @{u} | 
sed 's/\// /g') | cut -f1) ]
then
    echo "FlatBuffer Up-to-date"
else 
    echo "FlatBuffer Not Up-to-date"
    echo "New changes are availble for Compute Library repo."
    echo "Do you wanna update (y/n)?"
    read answer
    if [ "$answer" != "${answer#[Yy]}" ] ;then
        updateFlatBuffer
    fi
fi

exec "$@"