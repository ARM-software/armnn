# Delegate Build Guide

This guide assumes that Arm NN has been built with the Arm NN TF Lite Delegate with the [Arm NN Build Tool](../build-tool/README.md).<br>
The Arm NN TF Lite Delegate can also be obtained from downloading the [Pre-Built Binaries on the GitHub homepage](../README.md).

**Table of Contents:**
- [Running DelegateUnitTests](#running-delegateunittests)
- [Run the TF Lite Benchmark Tool](#run-the-tflite-model-benchmark-tool)
  - [Download the TFLite Model Benchmark Tool](#download-the-tflite-model-benchmark-tool)
  - [Execute the benchmarking tool with the Arm NN TF Lite Delegate](#execute-the-benchmarking-tool-with-the-arm-nn-tf-lite-delegate)
- [Integrate the Arm NN TfLite Delegate into your project](#integrate-the-arm-nn-tflite-delegate-into-your-project)


## Running DelegateUnitTests

To ensure that the build was successful you can run the unit tests for the delegate that can be found in
the build directory for the delegate. [Doctest](https://github.com/onqtam/doctest) was used to create those tests. Using test filters you can
filter out tests that your build is not configured for. In this case, we run all test suites that have `CpuAcc` in their name.
```bash
cd <PATH_TO_ARMNN_BUILD_DIRECTORY>/delegate/build
./DelegateUnitTests --test-suite=*CpuAcc*
```
If you have built for Gpu acceleration as well you might want to change your test-suite filter:
```bash
./DelegateUnitTests --test-suite=*CpuAcc*,*GpuAcc*
```

## Run the TFLite Model Benchmark Tool

The [TFLite Model Benchmark](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) Tool has a useful command line interface to test the TF Lite Delegate.
We can use this to demonstrate the use of the Arm NN TF Lite Delegate and its options.

Some examples of this can be viewed in this [YouTube demonstration](https://www.youtube.com/watch?v=NResQ1kbm-M&t=920s).

### Download the TFLite Model Benchmark Tool

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

### Execute the benchmarking tool with the Arm NN TF Lite Delegate
You are already at $BASEDIR/benchmarking from the previous stage.
```bash
LD_LIBRARY_PATH=<PATH_TO_ARMNN_BUILD_DIRECTORY> ./benchmark_model --graph=mobilenet_v2_1.0_224_quantized_1_default_1.tflite --external_delegate_path="<PATH_TO_ARMNN_BUILD_DIRECTORY>/delegate/libarmnnDelegate.so" --external_delegate_options="backends:CpuAcc;logging-severity:info"
```
The "external_delegate_options" here are specific to the Arm NN delegate. They are used to specify a target Arm NN backend or to enable/disable various options in Arm NN. A full description can be found in the parameters of function tflite_plugin_create_delegate.

## Integrate the Arm NN TfLite Delegate into your project

The delegate can be integrated into your c++ project by creating a TfLite Interpreter and
instructing it to use the Arm NN delegate for the graph execution. This should look similar
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

For further information on using TfLite Delegates please visit the [TensorFlow website](https://www.tensorflow.org/lite/guide).

For more details of the kind of options you can pass to the Arm NN delegate please check the parameters of function tflite_plugin_create_delegate.
