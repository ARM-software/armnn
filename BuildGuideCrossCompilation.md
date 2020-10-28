# How to Cross-Compile ArmNN on x86_64 for arm64

*  [Introduction](#introduction)
*  [Cross-compiling ToolChain](#installCCT)
*  [Build and install Google's Protobuf library](#buildProtobuf)
*  [Build Caffe for x86_64](#buildCaffe)
*  [Build Boost library for arm64](#installBaarch)
*  [Build Compute Library](#buildCL)
*  [Build ArmNN](#buildANN)
*  [Run Unit Tests](#unittests)
*  [Troubleshooting and Errors](#troubleshooting)


#### <a name="introduction">Introduction</a>
These are the step by step instructions on Cross-Compiling ArmNN under an x86_64 system to target an Arm64 system. This build flow has been tested with Ubuntu 16.04.
The instructions show how to build the ArmNN core library and the Boost, Protobuf, Caffe, Tensorflow, Tflite, Flatbuffer and Compute Libraries necessary for compilation.

#### <a name="installCCT">Cross-compiling ToolChain</a>
* Install the standard cross-compilation libraries for arm64:
    ```
    sudo apt install crossbuild-essential-arm64
    ```

#### <a name="buildProtobuf">Build and install Google's Protobuf library</a>

## We support protobuf version 3.12.0
* Get protobuf from here: https://github.com/protocolbuffers/protobuf : 
    ```bash
    git clone -b v3.12.0 https://github.com/google/protobuf.git protobuf
    cd protobuf
    git submodule update --init --recursive
    ./autogen
    ```
* Build a native (x86_64) version of the protobuf libraries and compiler (protoc):
  (Requires cUrl, autoconf, llibtool, and other build dependencies if not previously installed: sudo apt install curl autoconf libtool build-essential g++)
    ```
    mkdir x86_64_build
    cd x86_64_build
    ../configure --prefix=$HOME/armnn-devenv/google/x86_64_pb_install
    make install -j16
    cd ..
    ```
* Build the arm64 version of the protobuf libraries:
    ```
    mkdir arm64_build
    cd arm64_build
    export CC=aarch64-linux-gnu-gcc \
    export CXX=aarch64-linux-gnu-g++ \
    ../configure --host=aarch64-linux \
    --prefix=$HOME/armnn-devenv/google/arm64_pb_install \
    --with-protoc=$HOME/armnn-devenv/google/x86_64_pb_install/bin/protoc
    make install -j16
    cd ..
    ```

#### <a name="buildCaffe">Build Caffe for x86_64</a>
* Ubuntu 16.04 installation. These steps are taken from the full Caffe installation documentation at: http://caffe.berkeleyvision.org/install_apt.html
* Install dependencies:
    ```bash
    sudo apt-get install libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev
    sudo apt-get install --no-install-recommends libboost-all-dev
    sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
    sudo apt-get install libopenblas-dev
    sudo apt-get install libatlas-base-dev
    ```
* Download Caffe from: https://github.com/BVLC/caffe. We have tested using tag 1.0
    ```bash
    git clone https://github.com/BVLC/caffe.git
    cd caffe
    git checkout eeebdab16155d34ff8f5f42137da7df4d1c7eab0
    cp Makefile.config.example Makefile.config
    ```
* Adjust Makefile.config as necessary for your environment, for example:
    ```
    #CPU only version:
    CPU_ONLY := 1

    #Add hdf5 and protobuf include and library directories (Replace $HOME with explicit /home/username dir):
    INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/ $HOME/armnn-devenv/google/x86_64_pb_install/include/
    LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/ $HOME/armnn-devenv/google/x86_64_pb_install/lib/
    ```
* Setup environment:
    ```bash
    export PATH=$HOME/armnn-devenv/google/x86_64_pb_install/bin/:$PATH
    export LD_LIBRARY_PATH=$HOME/armnn-devenv/google/x86_64_pb_install/lib/:$LD_LIBRARY_PATH
    ```
* Compilation with Make:
    ```bash
    make all
    make test
    make runtest
    ```
    These should all run without errors
* caffe.pb.h and caffe.pb.cc will be needed when building ArmNN's Caffe Parser

#### <a name="installBaarch">Build Boost library for arm64</a>
* Build Boost library for arm64
    Download Boost version 1.64 from http://www.boost.org/doc/libs/1_64_0/more/getting_started/unix-variants.html
    Using any version of Boost greater than 1.64 will fail to build ArmNN, due to different dependency issues.
    ```bash
    tar -zxvf boost_1_64_0.tar.gz
    cd boost_1_64_0
    echo "using gcc : arm : aarch64-linux-gnu-g++ ;" > user_config.jam
    ./bootstrap.sh --prefix=$HOME/armnn-devenv/boost_arm64_install
    ./b2 install toolset=gcc-arm link=static cxxflags=-fPIC --with-test --with-log --with-program_options -j32 --user-config=user_config.jam
    ```

#### <a name="buildCL">Build Compute Library</a>
* Building the Arm Compute Library:
    ```bash
    git clone https://github.com/ARM-software/ComputeLibrary.git
    cd ComputeLibrary/
    git checkout <branch_name>
    git pull
    scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" -j8 internal_only=0
    ```

    For example, if you want to checkout release branch of 20.02:
        ```bash
        git checkout branches/arm_compute_20_02
        git pull
        ```

#### <a name="buildtf">Build Tensorflow</a>
* Building Tensorflow version 1.15:
    ```bash
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow/
    git checkout 590d6eef7e91a6a7392c8ffffb7b58f2e0c8bc6b
    ../armnn/scripts/generate_tensorflow_protobuf.sh ../tensorflow-protobuf ../google/x86_64_pb_install
    ```

#### <a name="buildflatbuffer">Build Flatbuffer</a>
* Building Flatbuffer version 1.12.0
    ```bash
    wget -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz
    tar xf flatbuffers-1.12.0.tar.gz
    cd flatbuffers-1.12.0
    rm -f CMakeCache.txt
    mkdir build
    cd build
    cmake .. -DFLATBUFFERS_BUILD_FLATC=1 \
	     -DCMAKE_INSTALL_PREFIX:PATH=$<DIRECTORY_PATH> \
	     -DFLATBUFFERS_BUILD_TESTS=0
    make all install
    ```

* Build arm64 version of flatbuffer
    ```bash
    mkdir build-arm64
    cd build-arm64
    # Add -fPIC to allow us to use the libraries in shared objects.
    CXXFLAGS="-fPIC" cmake .. -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
	     -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
	     -DFLATBUFFERS_BUILD_FLATC=1 \
	     -DCMAKE_INSTALL_PREFIX:PATH=$<DIRECTORY_PATH> \
	     -DFLATBUFFERS_BUILD_TESTS=0
    make all install
    ```

#### <a name="buildingONNX">Build Onnx</a>
* Building Onnx
    ```bash
    git clone https://github.com/onnx/onnx.git
    cd onnx
    git fetch https://github.com/onnx/onnx.git 553df22c67bee5f0fe6599cff60f1afc6748c635 && git checkout FETCH_HEAD
    export LD_LIBRARY_PATH=$<DIRECTORY_PATH>/protobuf-host/lib:$LD_LIBRARY_PATH
    ../google/x86_64_pb_install/bin/protoc onnx/onnx.proto --proto_path=. --proto_path=../google/x86_64_pb_install/include --cpp_out ../onnx
    ```

#### <a name="buildingtflite">Build TfLite</a>
* Building TfLite
    ```bash
    mkdir tflite
    cd tflite
    cp ../tensorflow/tensorflow/lite/schema/schema.fbs .
    ../flatbuffers-1.12.0/build/flatc -c --gen-object-api --reflect-types --reflect-names schema.fbs
    ```

#### <a name="buildANN">Build ArmNN</a>
* Compile ArmNN for arm64:
    ```bash
    git clone https://github.com/ARM-software/armnn.git
    cd armnn
    git checkout <branch_name>
    git pull
    mkdir build
    cd build
    ```

    For example, if you want to checkout release branch of 20.02:
        ```bash
        git checkout branches/armnn_20_02
        git pull
        ```

* Use CMake to configure your build environment, update the following script and run it from the armnn/build directory to set up the armNN build:
    ```bash
    #!/bin/bash
    export CXX=aarch64-linux-gnu-g++ \
    export CC=aarch64-linux-gnu-gcc \
    cmake .. \
    -DARMCOMPUTE_ROOT=$HOME/armnn-devenv/ComputeLibrary \
    -DARMCOMPUTE_BUILD_DIR=$HOME/armnn-devenv/ComputeLibrary/build/ \
    -DBOOST_ROOT=$HOME/armnn-devenv/boost_arm64_install/ \
    -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DARMNNREF=1 \
    -DCAFFE_GENERATED_SOURCES=$HOME/armnn-devenv/caffe/build/src \
    -DBUILD_CAFFE_PARSER=1 \
    -DONNX_GENERATED_SOURCES=$HOME/onnx \
    -DBUILD_ONNX_PARSER=1 \
    -DTF_GENERATED_SOURCES=$HOME/tensorflow-protobuf \
    -DBUILD_TF_PARSER=1 \
    -DBUILD_TF_LITE_PARSER=1 \
    -DTF_LITE_GENERATED_PATH=$HOME/tflite \
    -DFLATBUFFERS_ROOT=$HOME/flatbuffers-arm64 \
    -DFLATC_DIR=$HOME/flatbuffers-1.12.0/build \
    -DPROTOBUF_ROOT=$HOME/google/x86_64_pb_install \
    -DPROTOBUF_ROOT=$HOME/armnn-devenv/google/x86_64_pb_install/ \
    -DPROTOBUF_LIBRARY_DEBUG=$HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 \
    -DPROTOBUF_LIBRARY_RELEASE=$HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.23.0.0
    ```

* If you want to include standalone sample dynamic backend tests, add the argument to enable the tests and the dynamic backend path to the CMake command:
    ```bash
    -DSAMPLE_DYNAMIC_BACKEND=1 \
    -DDYNAMIC_BACKEND_PATHS=$SAMPLE_DYNAMIC_BACKEND_PATH
    ```
* Run the build
    ```bash
    make -j32
    ```

#### <a name="buildStandaloneBackend">Build Standalone Sample Dynamic Backend</a>
* The sample dynamic backend is located in armnn/src/dynamic/sample
    ```bash
    mkdir build
    cd build
    ```

* Use CMake to configure your build environment, update the following script and run it from the armnn/src/dynamic/sample/build directory to set up the armNN build:
    ```bash
    #!/bin/bash
    export CXX=aarch64-linux-gnu-g++ \
    export CC=aarch64-linux-gnu-gcc \
    cmake .. \
    -DCMAKE_CXX_FLAGS=--std=c++14 \
    -DBOOST_ROOT=$HOME/armnn-devenv/boost_arm64_install/ \
    -DBoost_SYSTEM_LIBRARY=$HOME/armnn-devenv/boost_arm64_install/lib/libboost_system.a \
    -DARMNN_PATH=$HOME/armnn-devenv/armnn/build/libarmnn.so
    ```

* Run the build
    ```bash
    make
    ```

#### <a name="unittests">Run Unit Tests</a>
* Copy the build folder to an arm64 linux machine
* Copy the libprotobuf.so.15.0.1 library file to the build folder
* If you enable the standalone sample dynamic tests, also copy libArm_SampleDynamic_backend.so library file to the folder specified as $SAMPLE_DYNAMIC_BACKEND_PATH when you build ArmNN 
* cd to the build folder on your arm64 machine and set your LD_LIBRARY_PATH to its current location:
    ```
    cd build/
    export LD_LIBRARY_PATH=`pwd`
    ```
* Create a symbolic link to libprotobuf.so.23.0.0:
    ```
    ln -s libprotobuf.so.23.0.0 ./libprotobuf.so.23
    ```
* Run the UnitTests:
    ```
    ./UnitTests
    Running 567 test cases...

    *** No errors detected
    ```
#### <a name="troubleshooting">Troubleshooting and Errors:</a>
#### Error adding symbols: File in wrong format
* When building armNN:
    ```
    /usr/local/lib/libboost_log.a: error adding symbols: File in wrong format
    collect2: error: ld returned 1 exit status
    CMakeFiles/armnn.dir/build.make:4028: recipe for target 'libarmnn.so' failed
    make[2]: *** [libarmnn.so] Error 1
    CMakeFiles/Makefile2:105: recipe for target 'CMakeFiles/armnn.dir/all' failed
    make[1]: *** [CMakeFiles/armnn.dir/all] Error 2
    Makefile:127: recipe for target 'all' failed
    make: *** [all] Error 2
    ```
* Boost libraries are not compiled for the correct architecture, try recompiling for arm64
##
#### Virtual memory exhausted
* When compiling the boost libraries:
    ```bash
    virtual memory exhausted: Cannot allocate memory
    ```
* Not enough memory available to compile. Increase the amount of RAM or swap space available.

##
#### Unrecognized command line option '-m64'
* When compiling the boost libraries:
    ```bash
    aarch64-linux-gnu-g++: error: unrecognized command line option ‘-m64’
    ```
* Clean the boost library directory before trying to build with a different architecture:
    ```bash
    sudo ./b2 clean
    ```
* It should show the following for arm64:
    ```bash
    - 32-bit                   : no
    - 64-bit                   : yes
    - arm                      : yes
    ```

##
#### Missing libz.so.1
* When compiling armNN:
    ```bash
    /usr/lib/gcc-cross/aarch64-linux-gnu/5/../../../../aarch64-linux-gnu/bin/ld: warning: libz.so.1, needed by /home/<username>/armNN/usr/lib64/libprotobuf.so.23.0.0, not found (try using -rpath or -rpath-link)
    ```

* Missing arm64 libraries for libz.so.1, these can be added by adding a second architecture to dpkg and explicitly installing them:
    ```bash
    sudo dpkg --add-architecture arm64
    sudo apt-get install zlib1g:arm64
    sudo apt-get update
    sudo ldconfig
    ```
* If apt-get update returns 404 errors for arm64 repos refer to section 5 below.
* Alternatively the missing arm64 version of libz.so.1 can be downloaded and installed from a .deb package here:
      https://launchpad.net/ubuntu/wily/arm64/zlib1g/1:1.2.8.dfsg-2ubuntu4
    ```bash
    sudo dpkg -i zlib1g_1.2.8.dfsg-2ubuntu4_arm64.deb
    ```
##
#### Unable to install arm64 packages after adding arm64 architecture
* Using sudo apt-get update should add all of the required repos for arm64 but if it does not or you are getting 404 errors the following instructions can be used to add the repos manually:
* From stackoverflow:
https://askubuntu.com/questions/430705/how-to-use-apt-get-to-download-multi-arch-library/430718
* Open /etc/apt/sources.list with your preferred text editor.

* Mark all the current (default) repos as \[arch=<current_os_arch>], e.g.
    ```bash
    deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ xenial main restricted
    ```
* Then add the following:
    ```bash
    deb [arch=arm64] http://ports.ubuntu.com/ xenial main restricted
    deb [arch=arm64] http://ports.ubuntu.com/ xenial-updates main restricted
    deb [arch=arm64] http://ports.ubuntu.com/ xenial universe
    deb [arch=arm64] http://ports.ubuntu.com/ xenial-updates universe
    deb [arch=arm64] http://ports.ubuntu.com/ xenial multiverse
    deb [arch=arm64] http://ports.ubuntu.com/ xenial-updates multiverse
    deb [arch=arm64] http://ports.ubuntu.com/ xenial-backports main restricted universe multiverse
    ```
* Update and install again:
    ```bash
    sudo apt-get install zlib1g:arm64
    sudo apt-get update
    sudo ldconfig
    ```
##
#### Undefined references to google::protobuf:: functions
* When compiling armNN there are multiple errors of the following type:
    ```
    libarmnnCaffeParser.so: undefined reference to `google::protobuf:*
    ```
* Missing or out of date protobuf compilation libraries.
    Use the command 'protoc --version' to check which version of protobuf is available (version 3.12.0 is required).
    Follow the instructions above to install protobuf 3.12.0
    Note this will require you to recompile Caffe for x86_64

##
#### Errors on strict-aliasing rules when compiling the Compute Library
* When compiling the Compute Library there are multiple errors on strict-aliasing rules:
     ```
    cc1plus: error: unrecognized command line option ‘-Wno-implicit-fallthrough’ [-Werror]
     ```
* Add Werror=0 to the scons command:
    ```
    scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" -j8 internal_only=0 Werror=0
    ```
