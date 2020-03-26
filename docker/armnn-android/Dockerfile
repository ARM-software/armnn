FROM ubuntu:18.04
ENV TERM linux
ENV DEBIAN_FRONTEND noninteractive

# Forward system proxy setting
# ARG proxy
# ENV http_proxy $proxy
# ENV https_proxy $proxy

# Basic apt update
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends locales ca-certificates &&  rm -rf /var/lib/apt/lists/*

# Set the locale to en_US.UTF-8, because the Yocto build fails without any locale set.
RUN locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Again, off the certificare
RUN echo "check_certificate = off" >> ~/.wgetrc
RUN echo "[global] \n\
trusted-host = pypi.python.org \n \
\t               pypi.org \n \
\t              files.pythonhosted.org" >> /etc/pip.conf

# Get basic packages
RUN apt-get update && apt-get install -y \
    apparmor \
    aufs-tools \
    automake \
    bash-completion \
    btrfs-tools \
    build-essential \
    cmake \
    createrepo \
    curl \
    dpkg-sig \
    g++ \
    gcc \
    git \
    iptables \
    jq \
    libapparmor-dev \
    libc6-dev \
    libcap-dev \
    libsystemd-dev \
    libyaml-dev \
    mercurial \
    net-tools \
    parallel \
    pkg-config \
    python-dev \
    python-mock \
    python-pip \
    python-setuptools \
    python-websocket \
    golang-go \
    iproute2 \
    iputils-ping \
    vim-common \
    vim \
    wget \
    libtool \
    unzip \
    scons \
    curl \
    autoconf \
    libtool \
    build-essential \
    g++ \ 
    cmake && rm -rf /var/lib/apt/lists/*

# Download the Android NDK and make a standalone toolchain
RUN mkdir -p /home/armnn-devenv/toolchains && \
    cd /home/armnn-devenv/toolchains && \
    wget https://dl.google.com/android/repository/android-ndk-r17b-linux-x86_64.zip && \
    unzip android-ndk-r17b-linux-x86_64.zip

ENV NDK /home/armnn-devenv/toolchains/android-ndk-r17b 

RUN $NDK/build/tools/make_standalone_toolchain.py \
    --arch arm64 \
    --api 26 \
    --stl=libc++ \
    --install-dir=/home/armnn-devenv/toolchains/aarch64-android-r17b

ENV PATH=/home/armnn-devenv/toolchains/aarch64-android-r17b/bin:$PATH

# Build the Boost C++ libraries
RUN mkdir /home/armnn-devenv/boost && \
    cd /home/armnn-devenv/boost && \
    wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2 && \
    tar xvf boost_1_64_0.tar.bz2

RUN echo "using gcc : arm : aarch64-linux-android-clang++ ;" > /home/armnn-devenv/boost/user-config.jam && \
    cd /home/armnn-devenv/boost/boost_1_64_0 && \
    ./bootstrap.sh --prefix=/home/armnn-devenv/boost/install && \
    ./b2 install --user-config=/home/armnn-devenv/boost/user-config.jam \
    toolset=gcc-arm link=static cxxflags=-fPIC --with-filesystem \
    --with-test --with-log --with-program_options -j8

# Build the Compute Library
RUN cd /home/armnn-devenv && \
    git clone https://github.com/ARM-software/ComputeLibrary.git && \
    cd ComputeLibrary && \
    scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" \
    benchmark_tests=0 validation_tests=0 os=android -j8

# RUN mkdir /home/armnn-devenv/google && \
RUN mkdir -p /home/armnn-devenv/google && \
    cd /home/armnn-devenv/google && \
    git clone https://github.com/google/protobuf.git && \
    cd protobuf && \
    git checkout -b v3.5.2 v3.5.2 && \
    ./autogen.sh && \
    mkdir x86_build && \
    cd x86_build && \
    ../configure --prefix=/home/armnn-devenv/google/x86_pb_install && \
    make install -j8
    
RUN cd /home/armnn-devenv/google/protobuf && \
    mkdir arm64_build && cd arm64_build && \
    CC=aarch64-linux-android-clang \
    CXX=aarch64-linux-android-clang++ \
    CFLAGS="-fPIE -fPIC" LDFLAGS="-pie -llog" \
    ../configure --host=aarch64-linux-android \
    --prefix=/home/armnn-devenv/google/arm64_pb_install \
    --with-protoc=/home/armnn-devenv/google/x86_pb_install/bin/protoc && \
    make install -j8

# clone Tensorflow
RUN cd /home/armnn-devenv/google/ && \
    git clone https://github.com/tensorflow/tensorflow.git 

# Clone ARMNN
RUN cd /home/armnn-devenv/ && \
    git clone https://github.com/ARM-software/armnn.git 

# Generate TensorFlow protobuf definitions
RUN cd /home/armnn-devenv/google/tensorflow && \
    git checkout a0043f9262dc1b0e7dc4bdf3a7f0ef0bebc4891e && \
    /home/armnn-devenv/armnn/scripts/generate_tensorflow_protobuf.sh \
    /home/armnn-devenv/google/tf_pb /home/armnn-devenv/google/x86_pb_install

ENV PATH=/home/armnn-devenv/toolchains/android-ndk-r17b:$PATH
# Build Google Flatbuffers for ARMNN TFLite Parser
RUN cd /home/armnn-devenv/ && \
    git clone https://github.com/google/flatbuffers.git && \
    cd flatbuffers && \
    cd android && cd jni && \
    rm -rf Application.mk && \
    echo "APP_STL := c++_static" >> Application.mk && \
    echo "NDK_TOOLCHAIN_VERSION" := clang >> Application.mk && \
    echo "APP_CPPFLAGS :=-std=c++11" >> Application.mk && \
    echo "APP_ABI := arm64-v8a" >> Application.mk && \
    echo "APP_PLATFORM := android-23" >> Application.mk && \
    echo "NDK_PLATFORM=android-23" >> Application.mk && \
    cd ../ && ndk-build -B

COPY ./docker-entrypoint.sh /usr/bin
RUN chmod +x /usr/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/bin/docker-entrypoint.sh"]

#To do:
# 1. Flatbuffers build Application.mk hardcode value need to fix.