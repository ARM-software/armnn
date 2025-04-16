Frequently asked questions
==========================

Problems seen when trying to build armnn and ComputeLibrary obtained from GitHub
-----------------------------------------------------------------------------

Some users have encountered difficulties when attempting to build armnn and ComputeLibrary obtained from GitHub. The build generally fails reporting missing dependencies or fields in aclCommon, backendsCommon, cl or neon. These errors can look like this:

error: ‘HARD_SWISH’ is not a member of ‘AclActivationFunction {aka arm_compute::ActivationLayerInfo::ActivationFunction}’

The most common reason for these errors are a mismatch between armnn and clframework revisions. For any version of Arm NN the coresponding version of ComputeLibrary is detailed in scripts/get_compute_library.sh as DEFAULT_CLFRAMEWORKREVISION

On *nix like systems running this script will checkout ComputeLibrary, with the current default SHA, into ../../clframework/ relative to the location of the script.

Segmentation fault following a failed call to armnn::Optimize using CpuRef backend.
---------------------------------------------------------

In some error scenarios of calls to armnn::Optimize a null pointer may be returned. This contravenes the function documentation however, it can happen. Users are advised to check the value returned from the function as a precaution.

If you encounter this problem and are able to isolate it consider contributing a solution.

Adding or removing -Dxxx options when building Arm NN does not always work.
---------------------------------------------------------

Arm NN uses CMake for build configuration. CMake uses a cumulative cache of user options. That is, setting a value once on a cmake command line will be persisted until either you explicitly change the value or delete the cache. To delete the cache in Arm NN you must delete the build directory.

Many DynamicBackendTests fail with "Base path for shared objects does not exist".
---------------------------------------------------------
This problem most commonly occurs when the compile and runtime environments for the unit tests differ. These dynamic backend tests rely on a set of test files and directories at runtime. These files are created by default during the cmake build. At runtime the tests will look for these files in src/backends/backendsCommon/test/ relative to where the Unittests executable was built. The usual solution to to copy these files and directories into the new unit test execution environment. You can also specify a new root path for these files by adding a command line parameter to the Unittests executable: Unittests -- --dynamic-backend-build-dir "new path"


Tensorflow Lite benchmarking tool fails with segmentation fault when using the Arm NN delegate.
---------------------------------------------------------
There are occaisional problems using native build versions of the Tensorflow Lite benchmarking tool. It can be sensitive to errors in command line parameter usage. A simple misspelling of a delegate name will result in a bus error. Here is a sample command line usage that is known to work for the Arm NN delegate.

This example is for:

* Execution on Android using a native binary downloaded from [Tensorflow Lite performance measurment](https://www.tensorflow.org/lite/performance/measurement#native_benchmark_binary).
* Uses a TF Lite model that has been downloaded from the ML-zoo. In this case [MobileNet v2 1.0 224 UINT8](https://github.com/ARM-software/ML-zoo/tree/master/models/image_classification/mobilenet_v2_1.0_224/tflite_uint8).
* Arm NN and its dependent libraries are in the current directory, /data/local/tmp.

~~~
LD_LIBRARY_PATH=/vendor/lib64/egl:/vendor/lib/egl/:. ./android_aarch64_benchmark_model --num_threads=4 --graph=/data/local/tmp/mobilenet_v2_1.0_224_quantized_1_default_1.tflite --external_delegate_path="libarmnnDelegate.so" --external_delegate_options="backends:GpuAcc"
~~~

Arm NN fails to build intermittently on 18.04 Ubuntu
---------------------------------------------------------
Building Arm NN fails intermittently with error:

c++: internal compiler error: Killed (program cc1plus)

This errors appears to be related to the number of cmake jobs used to build Arm NN.Try limiting the jobs to 2 by modifying the make to:

make -j2

Arm NN delegate build fails with "undefined reference to `absl::lts_20220623::raw_logging_internal::RawLog"
----------------------------------------------------------
This build failure occurs because Tensorflow 2.10 has been built with GCC version older than 9.3.1. The solution is to rebuild with 9.3.1 or later.


How can I run a pretrained model using Arm NN?.
-----------------------------------------------
The easiest way is to use Arm NN's tool ExecuteNetwork, the source code for this tool is located in armnn/tests/ExecuteNetwork/. This tool is highly configurable allowing multiple options to specify different parameters and it has profiling capabilities which can be enabled by specifying the option --event-based-profiling.

An example of running a model with ExecuteNetwork:
~~~
LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH ./ExecuteNetwork -f tflite-binary -m ./my_test_model.tflite -c CpuAcc,CpuRef
~~~
In the command above we specify two backends, if the an operator required by the model is not supported in the first backend CpuAcc Arm NN will try to use the second backend CpuRef.

Arm NN Multithreading support.
------------------------------
Running multiple inferences in multiple threads concurrently is not supported, but concurrent inferences can be executed in multiple processes and a simple way to do this is by using Arm NN's ExecuteNetwork.
ArmNN supports multithreading at kernel level and this is implemented in Arm Compute Library (ACL) (https://github.com/ARM-software/ComputeLibrary/).
During inference, at the operator level, the main thread will create multiple threads and execute the same kernel on different parts of the data. At runtime ACL will detect the number of CPU cores in the system and use one thread per cpu core for each kernel.
Multithreading at operator level is not supported due to limitations in ACL, for more information please refer to https://arm-software.github.io/ComputeLibrary/latest/architecture.xhtml#architecture_thread_safety

On Android, Executables containing Arm NN delegate or Arm NN TfLite Parser occasionally SIGABORT during destruction of Flatbuffers.
------------------------------
Unloading some TfLite models occasionally throws a SIGABORT. The error looks similar to this:
~~~
#0  0x0000007ff22df5c4 in abort () from target:/apex/com.android.runtime/lib64/bionic/libc.so
#1  0x0000007ff22ca61c in scudo::die() () from target:/apex/com.android.runtime/lib64/bionic/libc.so
#2  0x0000007ff22cb244 in scudo::ScopedErrorReport::~ScopedErrorReport() () from target:/apex/com.android.runtime/lib64/bionic/libc.so
#3  0x0000007ff22cb768 in scudo::reportInvalidChunkState(scudo::AllocatorAction, void*) () from target:/apex/com.android.runtime/lib64/bionic/libc.so
#4  0x0000007ff22cd520 in scudo::Allocator<scudo::AndroidConfig, &scudo_malloc_postinit>::deallocate(void*, scudo::Chunk::Origin, unsigned long, unsigned long) () from target:/apex/com.android.runtime/lib64/bionic/libc.so
#5  0x0000007fee6f96f8 in flatbuffers::ClassicLocale::~ClassicLocale() () from target:/data/local/tmp/build.android.aarch64/armnn/libarmnnTfLiteParser.so
~~~
The solution to set the flag "-DFLATBUFFERS_LOCALE_INDEPENDENT=0" in the build. By default, this is already done for our internal executables, for example, ExecuteNetwork.
