# Standalone dynamic backend developer guide

Arm NN allows adding new dynamic backends. Dynamic Backends can be compiled as standalone against Arm NN
and can be loaded by Arm NN dynamically at runtime.

To be properly loaded and used, the backend instances must comply to the standard interface for dynamic backends 
and to the versioning rules that enforce ABI compatibility.
The details of how to add dynamic backends can be found in [src/backends/README.md](../backends/README.md).


## Standalone dynamic backend example

The source code includes an example that is used to generate a dynamic implementation of the reference backend 
is provided at

[RefDynamicBackend.hpp](./reference/RefDynamicBackend.hpp)
[RefDynamicBackend.cpp](./reference/RefDynamicBackend.cpp)

The makefile used for building the standalone reference dynamic backend is also provided:
[CMakeLists.txt](./reference/CMakeLists.txt)


## Dynamic backend loading paths

During the creation of the Runtime, Arm NN will scan a given set of paths searching for suitable dynamic backend objects to load.
A list of (absolute) paths can be specified at compile-time by setting a define named ```DYNAMIC_BACKEND_PATHS```
 in the form of a colon-separated list of strings.

```shell
-DDYNAMIC_BACKEND_PATHS="PATH_1:PATH_2...:PATH_N"
```

The paths will be processed in the same order as they are indicated in the macro.
