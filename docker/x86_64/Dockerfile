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
    libssl-dev \
    g++ && rm -rf /var/lib/apt/lists/*

# Install Cross-compiling ToolChain
RUN apt-get update && apt-get install -y crossbuild-essential-arm64

# Install Cmake 3.19
RUN cd $HOME && \
    wget -O cmake-3.19.0.tar.gz https://cmake.org/files/v3.19/cmake-3.19.0.tar.gz && \
    tar -xzf cmake-3.19.0.tar.gz && \
    cd $HOME/cmake-3.19.0 && \
    ./bootstrap && \
    make && \
    make install

# Build and install Google's Protobuf library
# Download and Extract
RUN mkdir -p $HOME/google && \
    cd $HOME/google && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v3.12.0/protobuf-all-3.12.0.tar.gz && \
    tar -zxvf protobuf-all-3.12.0.tar.gz

# Build a native (x86_64) version
RUN cd $HOME/google/protobuf-3.12.0 && \
    mkdir x86_build && cd x86_build && \
    ../configure --prefix=$HOME/armnn-devenv/google/x86_64_pb_install && \
    make install -j$(nproc)

# Build the arm64 version of the protobuf libraries
RUN cd $HOME/google/protobuf-3.12.0 && \
    mkdir arm64_build && cd arm64_build && \
    export CC=aarch64-linux-gnu-gcc && \
    export CXX=aarch64-linux-gnu-g++ && \
    ../configure --host=aarch64-linux \
    --prefix=$HOME/armnn-devenv/google/arm64_pb_install \
    --with-protoc=$HOME/armnn-devenv/google/x86_64_pb_install/bin/protoc && \
    make install -j$(nproc)

# Dep Error - Bug ARMNN 
RUN apt-get update && apt-get install -y \
     python-numpy

# Setup Env
# ENV PATH=$HOME/armnn-devenv/google/x86_64_pb_install/bin/:$PATH
# ENV LD_LIBRARY_PATH=$HOME/armnn-devenv/google/x86_64_pb_install/lib/:$LD_LIBRARY_PATH

# Download ArmNN
RUN cd $HOME/armnn-devenv && git clone "https://review.mlplatform.org/ml/armnn" && \
    cd armnn && git checkout master

# Build Compute Library
RUN cd $HOME/armnn-devenv/ && git clone https://review.mlplatform.org/ml/ComputeLibrary && \
    cd ComputeLibrary && \
    git checkout $($HOME/armnn-devenv/armnn/scripts/get_compute_library.sh -p) && \
    scons Werror=0 arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" -j$(nproc)

# Download Tensorflow (Checkout latest tested version of TF using get_tensorflow.sh)
RUN cd $HOME/armnn-devenv && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout $($HOME/armnn-devenv/armnn/scripts/get_tensorflow.sh -p)

# Build TF Lite
RUN cd $HOME/armnn-devenv && \
    curl -LO https://storage.googleapis.com/mirror.tensorflow.org/developer.arm.com/media/Files/downloads/gnu-a/8.3-2019.03/binrel/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz && \
    mkdir tflite-toolchains && \
    tar xvf gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu.tar.xz -C tflite-toolchains && \
    mkdir -p tflite/build && \
    cd tflite/build && \
    ARMCC_PREFIX=$HOME/armnn-devenv/tflite-toolchains/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu- && \
    ARMCC_FLAGS="-funsafe-math-optimizations" && \
    cmake -DCMAKE_C_COMPILER=${ARMCC_PREFIX}gcc \
          -DCMAKE_CXX_COMPILER=${ARMCC_PREFIX}g++ \
          -DCMAKE_C_FLAGS="${ARMCC_FLAGS}" -DCMAKE_CXX_FLAGS="${ARMCC_FLAGS}" \
          -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON  -DCMAKE_SYSTEM_NAME=Linux \
          -DTFLITE_ENABLE_XNNPACK=OFF \
          -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
          $HOME/armnn-devenv/tensorflow/tensorflow/lite/ && \
    cmake --build .

# Download Flatbuffers
RUN cd $HOME/armnn-devenv && \
    wget -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz && \
    tar xf flatbuffers-1.12.0.tar.gz && \
    cd flatbuffers-1.12.0 && \
    rm -f CMakeCache.txt

# Build native x86_64 version of Flatbuffers
RUN cd $HOME/armnn-devenv && cd flatbuffers-1.12.0 && \
    mkdir build && \
    cd build && \
    cmake .. -DFLATBUFFERS_BUILD_FLATC=1 \
         -DCMAKE_INSTALL_PREFIX:PATH=$HOME/armnn-devenv/flatbuffers \
         -DFLATBUFFERS_BUILD_TESTS=0 && \
    make all install -j$(nproc)

# Build arm64 version of Flatbuffers
RUN cd $HOME/armnn-devenv&& cd flatbuffers-1.12.0  && \
    mkdir build-arm64 && \
    cd build-arm64 && \
    CXXFLAGS="-fPIC" cmake .. -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
         -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
         -DFLATBUFFERS_BUILD_FLATC=1 \
         -DCMAKE_INSTALL_PREFIX:PATH=$HOME/armnn-devenv/flatbuffers-arm64 \
         -DFLATBUFFERS_BUILD_TESTS=0 && \
    make all install -j$(nproc)

# Build onnx
RUN cd $HOME/armnn-devenv && git clone https://github.com/onnx/onnx.git && \
    cd onnx && \
    git fetch https://github.com/onnx/onnx.git 553df22c67bee5f0fe6599cff60f1afc6748c635 && git checkout FETCH_HEAD && \
    LD_LIBRARY_PATH=$HOME/armnn-devenv/google/x86_64_pb_install/lib:$LD_LIBRARY_PATH \
    $HOME/armnn-devenv/google/x86_64_pb_install/bin/protoc \
    onnx/onnx.proto --proto_path=. --proto_path=../google/x86_64_pb_install/include --cpp_out $HOME/armnn-devenv/onnx

# Generate TF Lite Schema
RUN cd $HOME/armnn-devenv && \
    cd tflite && \
    cp ../tensorflow/tensorflow/lite/schema/schema.fbs . && \
    ../flatbuffers-1.12.0/build/flatc -c --gen-object-api --reflect-types --reflect-names schema.fbs

# Build ArmNN
RUN cd $HOME/armnn-devenv && \
    cd armnn && mkdir build && cd build && \
    export CXX=aarch64-linux-gnu-g++ && \
    export CC=aarch64-linux-gnu-gcc && \
    cmake .. \
    -DCMAKE_CXX_FLAGS=-w \
    -DBUILD_TESTS=1 \
    -DARMCOMPUTE_ROOT=$HOME/armnn-devenv/ComputeLibrary \
    -DARMCOMPUTE_BUILD_DIR=$HOME/armnn-devenv/ComputeLibrary/build/ \
    -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DARMNNREF=1 \
    -DONNX_GENERATED_SOURCES=$HOME/armnn-devenv/onnx \
    -DBUILD_ONNX_PARSER=1 \
    -DBUILD_TF_LITE_PARSER=1 \
    -DBUILD_ARMNN_TFLITE_DELEGATE=1 \
    -DTENSORFLOW_ROOT=$HOME/armnn-devenv/tensorflow \
    -DTFLITE_LIB_ROOT=$HOME/armnn-devenv/tflite/build \
    -DTF_LITE_SCHEMA_INCLUDE_PATH=$HOME/armnn-devenv/tflite \
    -DFLATBUFFERS_ROOT=$HOME/armnn-devenv/flatbuffers-arm64 \
    -DFLATC_DIR=$HOME/armnn-devenv/flatbuffers-1.12.0/build \
    -DPROTOBUF_ROOT=$HOME/armnn-devenv/google/x86_64_pb_install \
    -DPROTOBUF_LIBRARY_DEBUG=$HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 \
    -DPROTOBUF_LIBRARY_RELEASE=$HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 && \
    make -j$(nproc)
