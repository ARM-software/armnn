# Delegate build guide introduction

The Arm NN Delegate can be found within the Arm NN repository but it is a standalone piece of software. However,
it makes use of the Arm NN library. For this reason we have added two options to build the delegate. The first option
allows you to build the delegate together with the Arm NN library, the second option is a standalone build 
of the delegate.

This tutorial uses an Aarch64 machine with Ubuntu 18.04 installed that can build all components
natively (no cross-compilation required). This is to keep this guide simple.

**Table of content:**
- [Delegate build guide introduction](#delegate-build-guide-introduction)
- [Dependencies](#dependencies)
   * [Download Arm NN](#download-arm-nn)
   * [Build Tensorflow Lite for C++](#build-tensorflow-lite-for-c--)
   * [Build Flatbuffers](#build-flatbuffers)
   * [Build the Arm Compute Library](#build-the-arm-compute-library)
   * [Build the Arm NN Library](#build-the-arm-nn-library)
- [Build the TfLite Delegate (Stand-Alone)](#build-the-tflite-delegate--stand-alone-)
- [Build the Delegate together with Arm NN](#build-the-delegate-together-with-arm-nn)
- [Integrate the Arm NN TfLite Delegate into your project](#integrate-the-arm-nn-tflite-delegate-into-your-project)


# Dependencies

Build Dependencies:
 * Tensorflow Lite: this guide uses version 2.5.0. Other versions may work.
 * Flatbuffers 1.12.0
 * Arm NN 21.11 or higher

Required Tools:
 * Git. This guide uses version 2.17.1. Other versions might work.
 * pip. This guide uses version 20.3.3. Other versions might work.
 * wget. This guide uses version 1.17.1. Other versions might work.
 * zip. This guide uses version 3.0. Other versions might work.
 * unzip. This guide uses version 6.00. Other versions might work.
 * cmake 3.16.0 or higher. This guide uses version 3.16.0
 * scons. This guide uses version 2.4.1. Other versions might work.

Our first step is to build all the build dependencies I have mentioned above. We will have to create quite a few
directories. To make navigation a bit easier define a base directory for the project. At this stage we can also
install all the tools that are required during the build. This guide assumes you are using a Bash shell.
```bash
export BASEDIR=~/ArmNNDelegate
mkdir $BASEDIR
cd $BASEDIR
apt-get update && apt-get install git wget unzip zip python git cmake scons
```

## Download Arm NN

First clone Arm NN using Git.

```bash
cd $BASEDIR
git clone "https://review.mlplatform.org/ml/armnn" 
cd armnn
git checkout <branch_name> # e.g. branches/armnn_21_11
```

## Build Tensorflow Lite for C++
Tensorflow has a few dependencies on it's own. It requires the python packages pip3, numpy,
and also Bazel or CMake which are used to compile Tensorflow. A description on how to build bazel can be
found [here](https://docs.bazel.build/versions/master/install-compile-source.html). But for this guide, we will
compile with CMake. Depending on your operating system and architecture there might be an easier way.
```bash
wget -O cmake-3.16.0.tar.gz https://cmake.org/files/v3.16/cmake-3.16.0.tar.gz
tar -xzf cmake-3.16.0.tar.gz -C $BASEDIR/

# If you have an older CMake, remove installed in order to upgrade
yes | sudo apt-get purge cmake
hash -r

cd $BASEDIR/cmake-3.16.0 
./bootstrap 
make 
sudo make install 
```

### Download and build Tensorflow Lite
Arm NN provides a script, armnn/scripts/get_tensorflow.sh, that can be used to download the version of TensorFlow that Arm NN was tested with:
```bash
cd $BASEDIR
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow/
git checkout $(../armnn/scripts/get_tensorflow.sh -p) # Minimum version required for the delegate is v2.3.1
```

Now the build process can be started. When calling "cmake", as below, you can specify a number of build
flags. But if you have no need to configure your tensorflow build, you can follow the exact commands below:
```bash
mkdir build # You are already inside $BASEDIR/tensorflow at this point
cd build
cmake $BASEDIR/tensorflow/tensorflow/lite -DTFLITE_ENABLE_XNNPACK=OFF
cmake --build . # This will be your DTFLITE_LIB_ROOT directory
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

The Arm NN library depends on the Arm Compute Library (ACL). It provides a set of functions that are optimized for 
both Arm CPUs and GPUs. The Arm Compute Library is used directly by Arm NN to run machine learning workloads on 
Arm CPUs and GPUs.

It is important to have the right version of ACL and Arm NN to make it work. Arm NN and ACL are developed very closely 
and released together. If you would like to use the Arm NN version "21.11" you should use the same "21.11" version for 
ACL too. Arm NN provides a script, armnn/scripts/get_compute_library.sh, that can be used to download the exact version 
of Arm Compute Library that Arm NN was tested with.

To build the Arm Compute Library on your platform, download the Arm Compute Library and checkout the tag that contains 
the version you want to use. Build it using `scons`.

```bash
cd $BASEDIR
git clone https://review.mlplatform.org/ml/ComputeLibrary 
cd ComputeLibrary/
git checkout $(../armnn/scripts/get_compute_library.sh -p) # e.g. v21.11
# The machine used for this guide only has a Neon CPU which is why I only have "neon=1" but if 
# your machine has an arm Gpu you can enable that by adding `opencl=1 embed_kernels=1 to the command below
scons arch=arm64-v8a neon=1 extra_cxx_flags="-fPIC" benchmark_tests=0 validation_tests=0 
```

## Build the Arm NN Library

With ACL built we can now continue to build Arm NN. Create a build directory and use `cmake` to build it.
```bash
cd $BASEDIR
cd armnn
mkdir build && cd build
# if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
cmake .. -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary -DARMCOMPUTENEON=1 -DBUILD_UNIT_TESTS=0 
make
```

# Build the TfLite Delegate (Stand-Alone)

The delegate as well as Arm NN is built using `cmake`. Create a build directory as usual and build the delegate
with the additional cmake arguments shown below
```bash
cd $BASEDIR/armnn/delegate && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=release                               # A release build rather than a debug build.
         -DTENSORFLOW_ROOT=$BASEDIR/tensorflow \                  # The root directory where tensorflow can be found.
         -DTFLITE_LIB_ROOT=$BASEDIR/tensorflow/build \               # Directory where tensorflow libraries can be found.
         -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-1.12.0/install \ # Flatbuffers install directory.
         -DArmnn_DIR=$BASEDIR/armnn/build \                       # Directory where the Arm NN library can be found
         -DARMNN_SOURCE_DIR=$BASEDIR/armnn                        # The top directory of the Arm NN repository. 
                                                                  # Required are the includes for Arm NN
make
```

To ensure that the build was successful you can run the unit tests for the delegate that can be found in 
the build directory for the delegate. [Doctest](https://github.com/onqtam/doctest) was used to create those tests. Using test filters you can
filter out tests that your build is not configured for. In this case, because Arm NN was only built for Cpu 
acceleration (CpuAcc), we filter for all test suites that have `CpuAcc` in their name.
```bash
cd $BASEDIR/armnn/delegate/build
./DelegateUnitTests --test-suite=*CpuAcc* 
```
If you have built for Gpu acceleration as well you might want to change your test-suite filter:
```bash
./DelegateUnitTests --test-suite=*CpuAcc*,*GpuAcc*
```

# Build the Delegate together with Arm NN

In the introduction it was mentioned that there is a way to integrate the delegate build into Arm NN. This is
pretty straight forward. The cmake arguments that were previously used for the delegate have to be added
to the Arm NN cmake arguments. Also another argument `BUILD_ARMNN_TFLITE_DELEGATE` needs to be added to 
instruct Arm NN to build the delegate as well. The new commands to build Arm NN are as follows:

Download Arm NN if you have not already done so:
```bash
cd $BASEDIR
git clone "https://review.mlplatform.org/ml/armnn" 
cd armnn
git checkout <branch_name> # e.g. branches/armnn_21_11
```
Build Arm NN with the delegate included
```bash
cd $BASEDIR
cd armnn
rm -rf build # Remove any previous cmake build.
mkdir build && cd build
# if you've got an arm Gpu add `-DARMCOMPUTECL=1` to the command below
cmake .. -DARMCOMPUTE_ROOT=$BASEDIR/ComputeLibrary \
         -DARMCOMPUTENEON=1 \
         -DBUILD_UNIT_TESTS=0 \
         -DBUILD_ARMNN_TFLITE_DELEGATE=1 \
         -DTENSORFLOW_ROOT=$BASEDIR/tensorflow \
         -DTFLITE_LIB_ROOT=$BASEDIR/tensorflow/build \
         -DFLATBUFFERS_ROOT=$BASEDIR/flatbuffers-1.12.0/install
make
```
The delegate library can then be found in `build/armnn/delegate`.

# Test the Arm NN delegate using the [TFLite Model Benchmark Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)

The TFLite Model Benchmark Tool has a useful command line interface to test delegates. We can use this to demonstrate the use of the Arm NN delegate and its options.

Some examples of this can be viewed in this [YouTube demonstration](https://www.youtube.com/watch?v=NResQ1kbm-M&t=920s).

## Download the TFLite Model Benchmark Tool

Binary builds of the benchmarking tool for various platforms are available [here](https://www.tensorflow.org/lite/performance/measurement#native_benchmark_binary). In this example I will target an aarch64 Linux environment. I will also download a sample uint8 tflite model from the [Arm ML Model Zoo](https://github.com/ARM-software/ML-zoo).

```bash
mkdir $BASEDIR/benchmarking
cd $BASEDIR/benchmarking
# Get the benchmarking binary.
wget https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/linux_aarch64_benchmark_model -O benchmark_model
# Make it executable.
chmod +x benchmark_model
# and a sample model from model zoo.
wget https://github.com/ARM-software/ML-zoo/blob/master/models/image_classification/mobilenet_v2_1.0_224/tflite_uint8/mobilenet_v2_1.0_224_quantized_1_default_1.tflite?raw=true -O mobilenet_v2_1.0_224_quantized_1_default_1.tflite
```

## Execute the benchmarking tool with the Arm NN delegate
You are already at $BASEDIR/benchmarking from the previous stage.
```bash
LD_LIBRARY_PATH=../armnn/build ./benchmark_model --graph=mobilenet_v2_1.0_224_quantized_1_default_1.tflite --external_delegate_path="../armnn/build/delegate/libarmnnDelegate.so" --external_delegate_options="backends:CpuAcc;logging-severity:info"
```
The "external_delegate_options" here are specific to the Arm NN delegate. They are used to specify a target Arm NN backend or to enable/disable various options in Arm NN. A full description can be found in the parameters of function tflite_plugin_create_delegate.

# Integrate the Arm NN TfLite Delegate into your project

The delegate can be integrated into your c++ project by creating a TfLite Interpreter and 
instructing it to use the Arm NN delegate for the graph execution. This should look similiar
to the following code snippet.
```objectivec
// Create TfLite Interpreter
std::unique_ptr<Interpreter> armnnDelegateInterpreter;
InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
                  (&armnnDelegateInterpreter)

// Create the Arm NN Delegate
armnnDelegate::DelegateOptions delegateOptions(backends);
std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                    theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                     armnnDelegate::TfLiteArmnnDelegateDelete);

// Instruct the Interpreter to use the armnnDelegate
armnnDelegateInterpreter->ModifyGraphWithDelegate(theArmnnDelegate.get());
```

For further information on using TfLite Delegates please visit the [tensorflow website](https://www.tensorflow.org/lite/guide)

For more details of the kind of options you can pass to the Arm NN delegate please check the parameters of function tflite_plugin_create_delegate.
