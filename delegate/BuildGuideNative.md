# Introduction

The ArmNN Delegate can be found within the ArmNN repository but it is a standalone piece of software. However,
it makes use of the ArmNN library. For this reason we have added two options to build the delegate. The first option
allows you to build the delegate together with the ArmNN library, the second option is a standalone build 
of the delegate.

This tutorial uses an Aarch64 machine with Ubuntu 18.04 installed that can build all components
natively (no cross-compilation required). This is to keep this guide simple.

1. [Dependencies](#Dependencies)
   * [Build Tensorflow for C++](#Build Tensorflow for C++)
   * [Build Flatbuffers](#Build Flatbuffers)
   * [Build the Arm Compute Library](#Build the Arm Compute Library)
   * [Build the ArmNN Library](#Build the ArmNN Library)
2. [Build the TfLite Delegate (Stand-Alone)](#Build the TfLite Delegate (Stand-Alone))
3. [Build the Delegate together with ArmNN](#Build the Delegate together with ArmNN)
4. [Integrate the ArmNN TfLite Delegate into your project](#Integrate the ArmNN TfLite Delegate into your project)

# Dependencies

Build Dependencies:
 * Tensorflow and Tensorflow Lite version 2.3.1
 * Flatbuffers 1.12.0
 * ArmNN 20.11 or higher

Required Tools:
 * Git
 * pip
 * wget
 * zip
 * unzip
 * cmake 3.7.0 or higher
 * scons
 * bazel 3.1.0

Our first step is to build all the build dependencies I have mentioned above. We will have to create quite a few
directories. To make navigation a bit easier define a base directory for the project. At this stage we can also
install all the tools that are required during the build.
```bash
export BASEDIR=/home
cd $BASEDIR
apt-get update && apt-get install git wget unzip zip python git cmake scons
```

## Build Tensorflow for C++
Tensorflow has a few dependencies on it's own. It requires the python packages pip3, numpy, wheel, keras_preprocessing
and also bazel which is used to compile Tensoflow. A description on how to build bazel can be 
found [here](https://docs.bazel.build/versions/master/install-compile-source.html). There are multiple ways. 
I decided to compile from source because that should work for any platform and therefore adds the most value 
to this guide. Depending on your operating system and architecture there might be an easier way.
```bash
# Install the python packages
pip3 install -U pip numpy wheel
pip3 install -U keras_preprocessing --no-deps

# Bazel has a dependency on JDK
apt-get install openjdk-11-jdk
# Build Bazel
wget -O bazel-3.1.0-dist.zip https://github.com/bazelbuild/bazel/releases/download/3.1.0/bazel-3.1.0-dist.zip
unzip -d bazel bazel-3.1.0-dist.zip
cd bazel
env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh 
# This creates an "output" directory where the bazel binary can be found
 
# Download Tensorflow
cd $BASEDIR
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow/
git checkout tags/v2.3.1 # Minimum version required for the delegate
```
Before tensorflow can be built, targets need to be defined in the `BUILD` file that can be 
found in the root directory of Tensorflow. Append the following two targets to the file:
```
cc_binary(
     name = "libtensorflow_all.so",
     linkshared = 1,
     deps = [
         "//tensorflow/core:framework",
         "//tensorflow/core:tensorflow",
         "//tensorflow/cc:cc_ops",
         "//tensorflow/cc:client_session",
         "//tensorflow/cc:scope",
         "//tensorflow/c:c_api",
     ],
)
cc_binary(
     name = "libtensorflow_lite_all.so",
     linkshared = 1,
     deps = [
         "//tensorflow/lite:framework",
         "//tensorflow/lite/kernels:builtin_ops",
     ],
)
```
Now the build process can be started. When calling "configure", as below, a dialog shows up that asks the 
user to specify additional options. If you don't have any particular needs to your build, decline all 
additional options and choose default values. Building `libtensorflow_all.so` requires quite some time. 
This might be a good time to get yourself another drink and take a break.
```bash
PATH="$BASEDIR/bazel/output:$PATH" ./configure
$BASEDIR/bazel/output/bazel build --define=grpc_no_ares=true --config=opt --config=monolithic --strip=always --config=noaws libtensorflow_all.so
$BASEDIR/bazel/output/bazel build --config=opt --config=monolithic --strip=always libtensorflow_lite_all.so
```

## Build Flatbuffers

Flatbuffers is a memory efficient cross-platform serialization library as 
described [here](https://google.github.io/flatbuffers/). It is used in tflite to store models and is also a dependency 
of the delegate. After downloading the right version it can be built and installed using cmake.
```bash
cd $BASEDIR
wget -O flatbuffers-1.12.0.zip https://github.com/google/flatbuffers/archive/v1.12.0.zip
unzip -d . flatbuffers-1.12.0.zip
cd flatbuffers-1.12.0 
mkdir install && mkdir build && cd build
# I'm using a different install directory but that is not required
cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$BASEDIR/flatbuffers-1.12.0/install 
make install
```

## Build the Arm Compute Library

The ArmNN library depends on the Arm Compute Library (ACL). It provides a set of functions that are optimized for 
both Arm CPUs and GPUs. The Arm Compute Library is used directly by ArmNN to run machine learning workloads on 
Arm CPUs and GPUs.

It is important to have the right version of ACL and ArmNN to make it work. Luckily, ArmNN and ACL are developed 
very closely and released together. If you would like to use the ArmNN version "20.11" you can use the same "20.11"
version for ACL too.

To build the Arm Compute Library on your platform, download the Arm Compute Library and check the branch 
out that contains the version you want to use and build it using `scons`.
```bash
cd $BASEDIR
git clone https://review.mlplatform.org/ml/ComputeLibrary 
cd ComputeLibrary/
git checkout <branch_name> # e.g. branches/arm_compute_20_11
# The machine used for this guide only has a Neon CPU which is why I only have "neon=1" but if 
# your machine has an arm Gpu you can enable that by adding `opencl=1 embed_kernels=1 to the command below
scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 
```

## Build the ArmNN Library

After building ACL we can now continue building ArmNN. To do so, download the repository and checkout the same 
version as you did for ACL. Create a build directory and use cmake to build it.
```bash
cd $BASEDIR
git clone "https://review.mlplatform.org/ml/armnn" 
cd armnn
git checkout <branch_name> # e.g. branches/armnn_20_11
mkdir build && cd build
# if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
cmake .. -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary -DARMCOMPUTENEON=1 -DBUILD_UNIT_TESTS=0 
make
```

# Build the TfLite Delegate (Stand-Alone)

The delegate as well as ArmNN is built using cmake. Create a build directory as usual and build the Delegate
with the additional cmake arguments shown below
```bash
cd $BASEDIR/armnn/delegate && mkdir build && cd build
cmake .. -DTENSORFLOW_LIB_DIR=$BASEDIR/tensorflow/bazel-bin \     # Directory where tensorflow libraries can be found
         -DTENSORFLOW_ROOT=$BASEDIR/tensorflow \                  # The top directory of the tensorflow repository
         -DTFLITE_LIB_ROOT=$BASEDIR/tensorflow/bazel-bin \        # In our case the same as TENSORFLOW_LIB_DIR 
         -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-1.12.0/install \ # The install directory 
         -DArmnn_DIR=$BASEDIR/armnn/build \                       # Directory where the ArmNN library can be found
         -DARMNN_SOURCE_DIR=$BASEDIR/armnn                        # The top directory of the ArmNN repository. 
                                                                  # Required are the includes for ArmNN
make
```

To ensure that the build was successful you can run the unit tests for the delegate that can be found in 
the build directory for the delegate. [Doctest](https://github.com/onqtam/doctest) was used to create those tests. Using test filters you can
filter out tests that your build is not configured for. In this case, because ArmNN was only built for Cpu 
acceleration (CpuAcc), we filter for all test suites that have `CpuAcc` in their name.
```bash
cd $BASEDIR/armnn/delegate/build
./DelegateUnitTests --test-suite=*CpuAcc* 
```
If you have built for Gpu acceleration as well you might want to change your test-suite filter:
```bash
./DelegateUnitTests --test-suite=*CpuAcc*,*GpuAcc*
```


# Build the Delegate together with ArmNN

In the introduction it was mentioned that there is a way to integrate the delegate build into ArmNN. This is
pretty straight forward. The cmake arguments that were previously used for the delegate have to be added
to the ArmNN cmake arguments. Also another argument `BUILD_ARMNN_TFLITE_DELEGATE` needs to be added to 
instruct ArmNN to build the delegate as well. The new commands to build ArmNN are as follows:
```bash
cd $BASEDIR
git clone "https://review.mlplatform.org/ml/armnn" 
cd armnn
git checkout <branch_name> # e.g. branches/armnn_20_11
mkdir build && cd build
# if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
cmake .. -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
         -DARMCOMPUTENEON=1 \
         -DBUILD_UNIT_TESTS=0 \
         -DBUILD_ARMNN_TFLITE_DELEGATE=1 \
         -DTENSORFLOW_LIB_DIR=$BASEDIR/tensorflow/bazel-bin \
         -DTENSORFLOW_ROOT=$BASEDIR/tensorflow \
         -DTFLITE_LIB_ROOT=$BASEDIR/tensorflow/bazel-bin \
         -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-1.12.0/install
make
```
The delegate library can then be found in `build/armnn/delegate`.


# Integrate the ArmNN TfLite Delegate into your project

The delegate can be integrated into your c++ project by creating a TfLite Interpreter and 
instructing it to use the ArmNN delegate for the graph execution. This should look similiar
to the following code snippet.
```objectivec
// Create TfLite Interpreter
std::unique_ptr<Interpreter> armnnDelegateInterpreter;
InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
                  (&armnnDelegateInterpreter)

// Create the ArmNN Delegate
armnnDelegate::DelegateOptions delegateOptions(backends);
std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                    theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                     armnnDelegate::TfLiteArmnnDelegateDelete);

// Instruct the Interpreter to use the armnnDelegate
armnnDelegateInterpreter->ModifyGraphWithDelegate(theArmnnDelegate.get());
```
For further information on using TfLite Delegates 
please visit the [tensorflow website](https://www.tensorflow.org/lite/guide)

