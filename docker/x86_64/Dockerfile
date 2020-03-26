FROM ubuntu:16.04
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

# Install Cross-compiling ToolChain
RUN apt-get update && apt-get install -y crossbuild-essential-arm64

# Build and install Google's Protobuf library
# Download and Extract
RUN mkdir -p $HOME/google && \
    cd $HOME/google && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.tar.gz && \
    tar -zxvf protobuf-all-3.5.1.tar.gz

# Build a native (x86_64) version
RUN cd $HOME/google/protobuf-3.5.1 && \
    mkdir x86_build && cd x86_build && \
    ../configure --prefix=$HOME/armnn-devenv/google/x86_64_pb_install && \
    make install -j16 

# Build the arm64 version of the protobuf libraries
RUN cd $HOME/google/protobuf-3.5.1 && \
    mkdir arm64_build && cd arm64_build && \
    export CC=aarch64-linux-gnu-gcc && \
    export CXX=aarch64-linux-gnu-g++ && \
    ../configure --host=aarch64-linux \
    --prefix=$HOME/armnn-devenv/google/arm64_pb_install \
    --with-protoc=$HOME/armnn-devenv/google/x86_64_pb_install/bin/protoc && \
    make install -j16

# Build Caffe for x86_64
# Dep Install
RUN apt-get update && apt-get install -y \
    libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev \
    --no-install-recommends libboost-all-dev \
    libgflags-dev libgoogle-glog-dev liblmdb-dev \
    libopenblas-dev \
    libatlas-base-dev 


# Download
RUN cd $HOME && git clone https://github.com/BVLC/caffe.git

# Makefile update
# To Do: Don't copy the Local Make file to docker
# RUN cd $HOME/caffe/ && rm Makefile.config.example
COPY Makefile.config /tmp
RUN mv /tmp/Makefile.config $HOME/caffe/

# Dep Error - Bug ARMNN 
RUN apt-get update && apt-get install -y \
     python-numpy

# Setup Env
# ENV PATH=$HOME/armnn-devenv/google/x86_64_pb_install/bin/:$PATH
# ENV LD_LIBRARY_PATH=$HOME/armnn-devenv/google/x86_64_pb_install/lib/:$LD_LIBRARY_PATH

# Compile CAFFE
RUN cd $HOME/caffe/ && mkdir build && cd build && \
    export PATH=$HOME/armnn-devenv/google/x86_64_pb_install/bin/:$PATH && \
    export LD_LIBRARY_PATH=$HOME/armnn-devenv/google/x86_64_pb_install/lib/:$LD_LIBRARY_PATH && \
    ldconfig && \
    cmake ../ && \
    make all -j8 && \
    make test -j8 && \
    make runtest -j8

# Build Boost library for arm64
RUN cd $HOME && wget http://downloads.sourceforge.net/project/boost/boost/1.64.0/boost_1_64_0.tar.gz && \
    tar xfz boost_1_64_0.tar.gz && \
    rm boost_1_64_0.tar.gz && \
    cd boost_1_64_0 && \
    echo "using gcc : arm : aarch64-linux-gnu-g++ ;" > user_config.jam && \
    ./bootstrap.sh --prefix=$HOME/armnn-devenv/boost_arm64_install && \
    ./b2 install toolset=gcc-arm link=static cxxflags=-fPIC --with-filesystem --with-test --with-log --with-program_options -j32 --user-config=user_config.jam 

# Build Compute Library
RUN cd $HOME/armnn-devenv/ && git clone https://github.com/ARM-software/ComputeLibrary.git && \
    cd ComputeLibrary && \
    scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" -j8 internal_only=0

# Build ArmNN
RUN cd $HOME && git clone https://github.com/ARM-software/armnn.git && \
    cd armnn && mkdir build && cd build && \
    export CXX=aarch64-linux-gnu-g++ && \
    export CC=aarch64-linux-gnu-gcc && \
    cmake .. \
    -DARMCOMPUTE_ROOT=$HOME/armnn-devenv/ComputeLibrary \
    -DARMCOMPUTE_BUILD_DIR=$HOME/armnn-devenv/ComputeLibrary/build/ \
    -DBOOST_ROOT=$HOME/armnn-devenv/boost_arm64_install/ \
    -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DARMNNREF=1 \
    -DCAFFE_GENERATED_SOURCES=$HOME/caffe/build/include \
    -DBUILD_CAFFE_PARSER=1 \
    -DPROTOBUF_ROOT=$HOME/armnn-devenv/google/x86_64_pb_install/ \
    -DPROTOBUF_LIBRARY_DEBUG=$HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.1 \
    -DPROTOBUF_LIBRARY_RELEASE=$HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.15.0.1 && \
    make -j8
