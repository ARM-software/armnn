# Standalone dynamic backend developer guide

Arm NN allows adding new dynamic backends. Dynamic Backends can be compiled as standalone against Arm NN
and can be loaded by Arm NN dynamically at runtime.

To be properly loaded and used, the backend instances must comply to the standard interface for dynamic backends 
and to the versioning rules that enforce ABI compatibility.
The details of how to add dynamic backends can be found in [src/backends/README.md](../backends/README.md).

### Standalone dynamic backend build

The easiest way to build a standalone sample dynamic backend is to build using environment configured compiler
and specify the Arm NN path to the CMake command:

```shell
cd ${DYNAMIC_BACKEND_DIR}
mkdir build
cd build
cmake -DARMNN_PATH=${ARMNN_PATH}/libarmnn.so ..
```

Then run the build

```shell
make
```

The library will be created in ${DYNAMIC_BACKEND_DIR}/build.

## Dynamic backend loading paths

During the creation of the Runtime, Arm NN will scan a given set of paths searching for suitable dynamic backend objects to load.
A list of (absolute) paths can be specified at compile-time by setting a define named ```DYNAMIC_BACKEND_PATHS```
 in the form of a colon-separated list of strings.

```shell
-DDYNAMIC_BACKEND_PATHS="PATH_1:PATH_2...:PATH_N"
```

Example for setting the path to the sample standalone dynamic backend built from the previous step:

```shell
-DDYNAMIC_BACKEND_PATHS=${DYNAMIC_BACKEND_DIR}/build
```

The paths will be processed in the same order as they are indicated in the macro.

## Standalone dynamic backend example

The source code includes an example that is used to generate a simple dynamic backend and is provided at

[SampleDynamicBackend.hpp](./sample/SampleDynamicBackend.hpp)
[SampleDynamicBackend.cpp](./sample/SampleDynamicBackend.cpp)

The details of how to create backends can be found in [src/backends/README.md](../backends/README.md).

The makefile used for building the standalone reference dynamic backend is also provided:
[CMakeLists.txt](./sample/CMakeLists.txt)

### End-To-End steps to build and test the sample dynamic backend
To build and test the sample dynamic backend mentioned above, first Arm NN must be built with the
sample dynamic unit tests turned on (**-DSAMPLE_DYNAMIC_BACKEND**) and the path must be provided to the Arm NN build the
location of where the sample dynamic backend will be located at (**-DDYNAMIC_BACKEND_PATHS**) at runtime.
This path should reflect the location on the target device, if this is different that the machine on which Arm NN was built.

Arm NN can be built using the [Build Tool](../../build-tool/README.md) with the following additional comma-separated **--armnn-cmake-args** in the **BUILD_ARGS**:
```shell
--armnn-cmake-args='-DSAMPLE_DYNAMIC_BACKEND=1,-DDYNAMIC_BACKEND_PATHS=/tmp/armnn/sample_dynamic_backend'
```

Then the sample dynamic backend can be built standalone using the following commands:
```shell
cd armnn/src/dynamic/sample
mkdir build
cd build
cmake -DARMNN_PATH=${ARMNN_BUILD_PATH}/libarmnn.so ..
make
```

A shared library file named **libArm_SampleDynamic_backend.so** will now be located in the build directory. Copy this to the location
defined by -DDYNAMIC_BACKEND_PATHS at compile time:
```shell
cp libArm_SampleDynamic_backend.so /tmp/armnn/sample_dynamic_backend
```

Then run the Arm NN unit tests which will be located inside the build directory created by the Arm NN build-tool:
```shell
./UnitTests
```

To be confident that the standalone dynamic backend tests are running, run the unit tests with the following filter:
```shell
./UnitTests -tc=CreateSampleDynamicBackend,SampleDynamicBackendEndToEnd
[doctest] doctest version is "2.4.6"
[doctest] run with "--help" for options
===============================================================================
[doctest] test cases:  2 |  2 passed | 0 failed | 2796 skipped
[doctest] assertions: 11 | 11 passed | 0 failed |
[doctest] Status: SUCCESS!

```
