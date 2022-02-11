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

Arm NN UnitTests fails intermittently with segmentation fault on aarch64.
----------------------------------------------------------
The DefaultAsyncExeuteWithThreads test seems to be throwing intermittent segmentation fault while running Arm NN Unittest in aarch64 architecture. This test will pass if you run the Unittest again.
