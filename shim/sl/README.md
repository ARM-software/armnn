# Arm NN Support Library Neural Networks driver

This directory contains the Arm NN Support Library for the Android Neural Networks API.

# Passing parameters to the support library runtime.

The support library inherits it's parameters from the Arm NN Android Neural Networks driver. Parameters are passed to it through an environment variable, ARMNN_SL_OPTIONS. A full list of parameters are available ./canonical/DriverOptions.cpp.

# Sample usage

## Running NeuralNetworksSupportLibraryTest

This test suite takes as it's first argument the path to a shared object implementation of the support library. Any library dependencies should be resolvable through the LD_LIBRARY_PATH mechanism. Setting ARMNN_SL_OPTIONS will pass parameters to the Arm NN Support Library Neural Networks driver.

Here we assume that Bash is the current shell and specify "-v" to enable verbose logging and "-c CpuAcc" to direct that the Neon(TM) accelerator be used.
~~~
ARMNN_SL_OPTIONS="-v -c CpuAcc" ./NeuralNetworksSupportLibraryTest ./libarmnn_support_library.so
~~~

## Running TfLite Benchmarking tool

This tools' parameters are described [here](https://www.tensorflow.org/lite/performance/measurement). The support library specific parts are to specify the path to the library and to ensure that ARMNN_SL_OPTIONS is set in the environment.

support for relaxed computation from Float32 to Float16"
~~~
ARMNN_SL_OPTIONS="-v -c GpuAcc -f" ./android_aarch64_benchmark_model --graph=./mymodel.tflite --num_threads=1 --use_nnapi=true --num_runs=1 --nnapi_support_library_path=./libarmnn_support_library.so --nnapi_accelerator_name=arm-armnn-sl
~~~

### License

The Arm NN Support Library Neural Networks driver is provided under the [MIT](https://spdx.org/licenses/MIT.html) license.
See [LICENSE](LICENSE) for more information. Contributions to this project are accepted under the same license.

Individual files contain the following tag instead of the full license text.

    SPDX-License-Identifier: MIT

This enables machine processing of license information based on the SPDX License Identifiers that are available here: http://spdx.org/licenses/
