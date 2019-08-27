# How to use the Android NDK to build ArmNN

*  [Introduction](#introduction)
*  [Download the Android NDK and make a standalone toolchain](#downloadNDK)
*  [Build the Boost C++ libraries](#buildBoost)
*  [Build the Compute Library](#buildCL)
*  [Build Google's Protobuf library](#buildProtobuf)
*  [Download TensorFlow](#downloadTF)
*  [Build ArmNN](#buildArmNN)
*  [Run ArmNN UnitTests on an Android device](#runArmNNUnitTests)


#### <a name="introduction">Introduction</a>
These are step by step instructions for using the Android NDK to build ArmNN.
They have been tested on a clean install of Ubuntu 18.04, and should also work with other OS versions.
The instructions show how to build the ArmNN core library and the optional TensorFlow parser.
All downloaded or generated files will be saved inside the `~/armnn-devenv` directory.

#### <a name="downloadNDK">Download the Android NDK and make a standalone toolchain</a>

* Download the Android NDK from [the official website](https://developer.android.com/ndk/downloads/index.html):

     ```bash
     mkdir -p ~/armnn-devenv/toolchains
     cd ~/armnn-devenv/toolchains
     # For Mac OS, change the NDK download link accordingly.
     wget https://dl.google.com/android/repository/android-ndk-r17b-linux-x86_64.zip
     unzip android-ndk-r17b-linux-x86_64.zip
     export NDK=~/armnn-devenv/toolchains/android-ndk-r17b
     ```

	 You may want to append `export NDK=~/armnn-devenv/toolchains/android-ndk-r17b` to your `~/.bashrc` (or `~/.bash_profile` in Mac OS).

* Make a standalone toolchain:

	 (Requires python if not previously installed: `sudo apt install python`)

   ```bash
   # Create an arm64 API 26 libc++ toolchain.
   $NDK/build/tools/make_standalone_toolchain.py \
       --arch arm64 \
       --api 26 \
       --stl=libc++ \
       --install-dir=$HOME/armnn-devenv/toolchains/aarch64-android-r17b
   export PATH=$HOME/armnn-devenv/toolchains/aarch64-android-r17b/bin:$PATH
   ```

	 You may want to append `export PATH=$HOME/armnn-devenv/toolchains/aarch64-android-r17b/bin:$PATH` to your `~/.bashrc` (or `~/.bash_profile` in Mac OS).

#### <a name="buildBoost">Build the Boost C++ libraries</a>

* Download Boost version 1.64:

   ```bash
   mkdir ~/armnn-devenv/boost
   cd ~/armnn-devenv/boost
   wget https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.bz2
   tar xvf boost_1_64_0.tar.bz2
   ```

* Build:

	(Requires gcc if not previously installed: `sudo apt install gcc`)
	```bash
	echo "using gcc : arm : aarch64-linux-android-clang++ ;" > $HOME/armnn-devenv/boost/user-config.jam
	cd ~/armnn-devenv/boost/boost_1_64_0
	./bootstrap.sh --prefix=$HOME/armnn-devenv/boost/install
	./b2 install --user-config=$HOME/armnn-devenv/boost/user-config.jam \
     toolset=gcc-arm link=static cxxflags=-fPIC --with-filesystem \
	 --with-test --with-log --with-program_options -j16
    ```

#### <a name="buildCL">Build the Compute Library</a>
* Clone the Compute Library:

	(Requires Git if not previously installed: `sudo apt install git`)

	``` bash
	cd ~/armnn-devenv
	git clone https://github.com/ARM-software/ComputeLibrary.git
	```

* Build:

	(Requires SCons if not previously installed: `sudo apt install scons`)
	```bash
	cd ComputeLibrary
	scons arch=arm64-v8a neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" \
	 benchmark_tests=0 validation_tests=0 os=android -j16
	```

#### <a name="buildProtobuf">Build Google's Protobuf library</a>

* Clone protobuf:
	```bash
	mkdir ~/armnn-devenv/google
	cd ~/armnn-devenv/google
	git clone https://github.com/google/protobuf.git
	cd protobuf
	git checkout -b v3.5.2 v3.5.2
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
 	CC=aarch64-linux-android-clang \
 	  CXX=aarch64-linux-android-clang++ \
	  CFLAGS="-fPIE -fPIC" LDFLAGS="-pie -llog" \
       ../configure --host=aarch64-linux-android \
       --prefix=$HOME/armnn-devenv/google/arm64_pb_install \
       --with-protoc=$HOME/armnn-devenv/google/x86_pb_install/bin/protoc
 	make install -j16
	cd ..
	```

#### <a name="downloadTF">Download TensorFlow</a>
* Clone TensorFlow source code:

	```bash
	cd ~/armnn-devenv/google/
	git clone https://github.com/tensorflow/tensorflow.git
	```

#### <a name="buildArmNN">Build ArmNN</a>

* Clone ArmNN source code:

	```bash
	cd ~/armnn-devenv/
	git clone https://github.com/ARM-software/armnn.git
	```

* Generate TensorFlow protobuf definitions:

	```bash
	cd ~/armnn-devenv/google/tensorflow
	~/armnn-devenv/armnn/scripts/generate_tensorflow_protobuf.sh \
	 $HOME/armnn-devenv/google/tf_pb $HOME/armnn-devenv/google/x86_pb_install
	```

 * Build ArmNN:

 	(Requires CMake if not previously installed: `sudo apt install cmake`)

	```bash
	mkdir ~/armnn-devenv/armnn/build
	cd ~/armnn-devenv/armnn/build
	CXX=aarch64-linux-android-clang++ \
	 CC=aarch64-linux-android-clang \
	 CXX_FLAGS="-fPIE -fPIC" \
	 cmake .. \
      -DCMAKE_SYSTEM_NAME=Android \
      -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
      -DCMAKE_ANDROID_STANDALONE_TOOLCHAIN=$HOME/armnn-devenv/toolchains/aarch64-android-r17b/ \
      -DCMAKE_EXE_LINKER_FLAGS="-pie -llog" \
      -DARMCOMPUTE_ROOT=$HOME/armnn-devenv/ComputeLibrary/ \
      -DARMCOMPUTE_BUILD_DIR=$HOME/armnn-devenv/ComputeLibrary/build \
      -DBOOST_ROOT=$HOME/armnn-devenv/boost/install/ \
      -DARMCOMPUTENEON=1 -DARMCOMPUTECL=1 -DARMNNREF=1 \
      -DTF_GENERATED_SOURCES=$HOME/armnn-devenv/google/tf_pb/ -DBUILD_TF_PARSER=1 \
      -DPROTOBUF_ROOT=$HOME/armnn-devenv/google/arm64_pb_install/
	make -j16
	```

#### <a name="runArmNNUnitTests">Run the ArmNN unit tests on an Android device</a>


* Push the build results to an Android device and make symbolic links for shared libraries:

	```bash
	adb push libarmnnTfParser.so /data/local/tmp/
	adb push libarmnn.so /data/local/tmp/
	adb push UnitTests /data/local/tmp/
	adb push $NDK/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so /data/local/tmp/
	adb push $HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so /data/local/tmp/libprotobuf.so.15.0.1
	adb shell 'ln -s libprotobuf.so.15.0.1 /data/local/tmp/libprotobuf.so.15'
	adb shell 'ln -s libprotobuf.so.15.0.1 /data/local/tmp/libprotobuf.so'
	```

* Push the files needed for the unit tests (they are a mix of files, directories and symbolic links):

	```bash
	adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/testSharedObject
	adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/testSharedObject/ /data/local/tmp/src/backends/backendsCommon/test/testSharedObject/

	adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend
	adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/testDynamicBackend/ /data/local/tmp/src/backends/backendsCommon/test/testDynamicBackend/

	adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1
	adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath1/ /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath1/

	adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2
	adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
	adb shell ln -s Arm_CpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1
	adb shell ln -s Arm_CpuAcc_backend.so.1 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2
	adb shell ln -s Arm_CpuAcc_backend.so.1.2 /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_CpuAcc_backend.so.1.2.3
	adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath2/Arm_GpuAcc_backend.so /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/
	adb shell ln -s nothing /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath2/Arm_no_backend.so

	adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath3

	adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5
	adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath5/ /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath5/

	adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6
	adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath6/ /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath6/

	adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath7

	adb shell mkdir -p /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9
	adb push -p ~/armnn-devenv/armnn/build/src/backends/backendsCommon/test/backendsTestPath9/ /data/local/tmp/src/backends/backendsCommon/test/backendsTestPath9/

	adb shell mkdir -p /data/local/tmp/src/backends/dynamic/reference
	adb push -p ~/armnn-devenv/armnn/build/src/backends/dynamic/reference/Arm_CpuRef_backend.so /data/local/tmp/src/backends/dynamic/reference/
	```

* Run ArmNN unit tests:

	```bash
	adb shell 'LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/UnitTests'
	```

	If libarmnnUtils.a is present in `~/armnn-devenv/armnn/build/` and the unit tests run without failure then the build was successful.

