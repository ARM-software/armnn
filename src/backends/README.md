# Backend developer guide

Arm NN allows adding new backends through the 'Pluggable Backend' mechanism.

## How to add a new backend

Backends reside under [src/backends](./), in separate subfolders. For Linux builds they must have a ```backend.cmake``` file,
which is read automatically by [src/backends/backends.cmake](backends.cmake). The ```backend.cmake``` file
under the backend-specific folder is then included by the main CMakeLists.txt file at the root of the
Arm NN source tree.

### The backend.cmake file

The ```backend.cmake``` has three main purposes:

1. It makes sure the artifact (a cmake OBJECT library) is linked into the Arm NN shared library by appending the name of the library to the ```armnnLibraries``` list.
2. It makes sure that the subdirectory where backend sources reside gets included into the build.
3. To include backend-specific unit tests, the object library for the unit tests needs to be added to the ```armnnUnitTestLibraries``` list.

Example ```backend.cmake``` file taken from [reference/backend.cmake](reference/backend.cmake):

```cmake
#
# Make sure the reference backend is included in the build.
# By adding the subdirectory, cmake requires the presence of CMakeLists.txt
# in the reference (backend) folder.
#
add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/reference)

#
# Add the cmake OBJECT libraries built by the reference backend to the
# list of libraries linked against the Arm NN shared library.
#
list(APPEND armnnLibraries armnnRefBackend armnnRefBackendWorkloads)

#
# Backend specific unit tests can be integrated through the
# armnnUnitTestLibraries variable. This makes sure that the
# UnitTests executable can run the backend-specific unit
# tests.
#
list(APPEND armnnUnitTestLibraries armnnRefBackendUnitTests)
```

### The CMakeLists.txt file

As described in the previous section, adding a new backend will require creating a ```CMakeLists.txt``` in
the backend folder. This follows the standard cmake conventions, and is required to build a static cmake OBJECT library
to be linked into the Arm NN shared library. As with any cmake build, the code can be structured into
subfolders and modules as the developer sees fit.

Example can be found under [reference/CMakeLists.txt](reference/CMakeLists.txt).

### The backend.mk file

Arm NN on Android uses the native Android build system. New backends are integrated by creating a
```backend.mk``` file, which has a single variable called ```BACKEND_SOURCES``` listing all cpp
files to be built by the Android build system for the Arm NN shared library.

Optionally, backend-specific unit tests can be added similarly, by
appending the list of cpp files to the ```BACKEND_TEST_SOURCES``` variable.

Example taken from [reference/backend.mk](reference/backend.mk):

```make
BACKEND_SOURCES := \
        RefLayerSupport.cpp \
        RefWorkloadFactory.cpp \
        workloads/Activation.cpp \
        workloads/ElementwiseFunction.cpp \
        workloads/Broadcast.cpp \
        ...

BACKEND_TEST_SOURCES := \
        test/RefCreateWorkloadTests.cpp \
        test/RefEndToEndTests.cpp \
        test/RefJsonPrinterTests.cpp \
        ...
```

## How to add common code across backends

For multiple backends that need common code, there is support for including them in the build
similarly to the backend code. This requires adding three files under a subfolder at the same level
as the backends folders. These are:

1. common.cmake
2. common.mk
3. CMakeLists.txt

They work the same way as the backend files. The only difference between them is that
common code is built first, so the backend code can depend on them.

[aclCommon](aclCommon) is an example for this concept and you can find the corresponding files:

1. [aclCommon/common.cmake](aclCommon/common.cmake)
2. [aclCommon/common.mk](aclCommon/common.mk)
3. [aclCommon/CMakeLists.txt](aclCommon/CMakeLists.txt)

## Identifying backends

Backends are identified by a string that must be unique across backends. This string is
wrapped in the [BackendId](../../include/armnn/BackendId.hpp) object for backward compatibility
with previous Arm NN versions.

## The IBackendInternal interface

All backends need to implement the [IBackendInternal](../../include/armnn/backends/IBackendInternal.hpp) interface.
The interface functions to be implemented are:

```c++
    virtual IMemoryManagerUniquePtr CreateMemoryManager() const = 0;
    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
            const IMemoryManagerSharedPtr& memoryManager = nullptr) const = 0;
    virtual IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const = 0;
    virtual IBackendProfilingContextPtr CreateBackendProfilingContext(const IRuntime::CreationOptions& creationOptions,
            armnn::profiling::IBackendProfiling& backendProfiling) const = 0;
    virtual ILayerSupportSharedPtr GetLayerSupport() const = 0;
    virtual Optimizations GetOptimizations() const = 0;
    virtual SubgraphUniquePtr OptimizeSubgraph(const SubgraphView& subgraph, bool& optimizationAttempted) const;
    virtual OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const;
```

Note that ```GetOptimizations()``` and ```SubgraphViewUniquePtr OptimizeSubgraphView(const SubgraphView& subgraph, bool& optimizationAttempted)```
have been deprecated.
The method ```OptimizationViews OptimizeSubgraph(const SubgraphView& subgraph)``` should be used instead to
apply specific optimizations to a given sub-graph.

The Arm NN framework then creates instances of the IBackendInternal interface with the help of the
[BackendRegistry](../../include/armnn/BackendRegistry.hpp) singleton.

**Important:** the ```IBackendInternal``` object is not guaranteed to have a longer lifetime than
the objects it creates. It is only intended to be a single entry point for the factory functions it has.
The best use of this is to be a lightweight, stateless object and make no assumptions between
its lifetime and the lifetime of the objects it creates.

For each backend one needs to register a factory function that can
be retrieved using a [BackendId](../../include/armnn/BackendId.hpp).
The Arm NN framework creates the backend interfaces dynamically when
it sees fit and it keeps these objects for a short period of time. Examples:

* During optimization Arm NN needs to decide which layers are supported by the backend.
  To do this, it creates a backends and calls the ```GetLayerSupport()``` function and creates
  an ```ILayerSupport``` object to help deciding this.
* During optimization Arm NN can run backend-specific optimizations. After splitting the graph into
  sub-graphs based on backends, it calls the ```OptimizeSubgraphView()``` function on each of them and, if possible,
  substitutes the corresponding sub-graph in the original graph with its optimized version.
* When the Runtime is initialized it creates an optional ```IBackendContext``` object and keeps this context alive
  for the Runtime's lifetime. It notifies this context object before and after a network is loaded or unloaded.
* When the LoadedNetwork creates the backend-specific workloads for the layers, it creates a backend
  specific workload factory and calls this to create the workloads.

## The BackendRegistry

As mentioned above, all backends need to be registered through the BackendRegistry so Arm NN knows
about them. Registration requires a unique backend ID string and a lambda function that
returns a unique pointer to the [IBackendInternal interface](../../include/armnn/backends/IBackendInternal.hpp).

For registering a backend only this lambda function needs to exist, not the actual backend. This
allows dynamically creating the backend objects when they are needed.

The BackendRegistry has a few convenience functions, like we can query the registered backends and
are able to tell if a given backend is registered or not.

Dynamic backends are registered during the runtime creation.

## The ILayerSupport interface

Arm NN uses the [ILayerSupport](../../include/armnn/backends/ILayerSupport.hpp) interface to decide if a layer
with a set of parameters (i.e. input and output tensors, descriptor, weights, filter, kernel if any) are
supported on a given backend. The backends need a way to communicate this information by implementing
the ```GetLayerSupport()``` function on the ```IBackendInternal``` interface.

Examples of this can be found in the [RefLayerSupport header](reference/RefLayerSupport.hpp)
and the [RefLayerSupport implementation](reference/RefLayerSupport.cpp).

## The IWorkloadFactory interface

The [IWorkloadFactory interface](backendsCommon/WorkloadFactory.hpp) is used for creating the backend
specific workloads. The factory function that creates the IWorkloadFactory object in the IBackendInterface
takes an IMemoryManager object.

To create a workload object the ```IWorkloadFactory``` takes a ```WorkloadInfo``` object that holds
the input and output tensor information and a workload specific queue descriptor.

## The IMemoryManager interface

Backends may choose to implement custom memory management. Arm NN supports this concept through the following
mechanism:

* the ```IBackendInternal``` interface has a ```CreateMemoryManager()``` function, which is called before
  creating the workload factory
* the memory manager is passed to the ```CreateWorkloadFactory(...)``` function so the workload factory can
  use it for creating the backend-specific workloads
* the LoadedNetwork calls ```Acquire()``` on the memory manager before it starts executing the network and
  it calls ```Release()``` in its destructor

## The Optimizations

The backends may choose to implement backend-specific optimizations.
This is supported through the method ```OptimizationViews OptimizeSubgraph(const SubgraphView& subgraph)``` of
the backend interface that allows the backends to apply their specific optimizations to a given sub-graph.

The ```OptimizeSubgraph(...)``` method returns an OptimizationViews object containing three lists:

* A list of the sub-graph substitutions: a "substitution" is a pair of sub-graphs, the first is the "substitutable" sub-graph,
  representing the part of the original graph that has been optimized by the backend, while the second is the "replacement" sub-graph,
  containing the actual optimized layers that will be replaced in the original graph correspondingly to the "substitutable" sub-graph
* A list of the failed sub-graphs: these are the parts of the original sub-graph that are not supported by the backend,
  thus have been rejected. Arm NN will try to re-allocate these parts on other backends if available.
* A list of the untouched sub-graphs: these are the parts of the original sub-graph that have not been optimized,
  but that can run (unoptimized) on the backend.

The previous way backends had to provide a list optimizations to the Optimizer (through the ```GetOptimizations()``` method)
is still in place for backward compatibility, but it's now considered deprecated and will be remove in a future release.

## The IBackendContext interface

Backends may need to be notified whenever a network is loaded or unloaded. To support that, one can implement the optional
[IBackendContext](../../include/armnn/backends/IBackendContext.hpp) interface. The framework calls the ```CreateBackendContext(...)```
method for each backend in the Runtime. If the backend returns a valid unique pointer to a backend context, then the
runtime will hold this for its entire lifetime. It then calls the following interface functions for each stored context:

* ```BeforeLoadNetwork(NetworkId networkId)```
* ```AfterLoadNetwork(NetworkId networkId)```
* ```BeforeUnloadNetwork(NetworkId networkId)```
* ```AfterUnloadNetwork(NetworkId networkId)```

## The UseCustomMemoryAllocator interface

Backends can also have an associated CustomMemoryAllocator registered with them that ArmNN will use to allocate
intra/inter-layer memory. This particular feature is required if you want a backend to use ProtectedContentAllocation.
To support this on your own backend you must implement the UseCustomMemoryAllocator interface.

This interface returns a boolean value which indicates if the provided allocator is supported by
the backend. This interface is also used by the lambda function returned by the Backend Registry to configure
the CustomMemoryAllocator. Within the backend itself there should be a wrapper class to convert the generic
CustomMemoryAllocator provided by the interface into something that is more suitable for your own backend.

Examples of how this can be done are in the [ClBackend header](cl/ClBackend.hpp) and the
[ClRegistryInitializer header](cl/ClRegistryInitializer.cpp)

## The GetCapabilities interface

This is a list of BackendCapabilities currently supported by the backend. It consists of a constant list of
Name/Value pairs, each containing a string name, and a boolean value to indicate support. For example to
indicate support for ProtectedContentAllocation you would return {"ProtectedContentAllocation", true}

An example can be found at the top of [ClBackend header](cl/ClBackend.hpp)

## Dynamic backends

Backends can also be loaded by Arm NN dynamically at runtime.
To be properly loaded and used, the backend instances must comply to the standard interface for dynamic backends and to the versioning
rules that enforce ABI compatibility.

## Dynamic backends base interface

The dynamic backend shared object must expose the following interface for Arm NN to handle it correctly:

```c++
extern "C"
{
const char* GetBackendId();
void GetVersion(uint32_t* outMajor, uint32_t* outMinor);
void* BackendFactory();
}
```

Interface details:

* ```extern "C"``` is needed to use avoid C++ name mangling, necessary to allow Arm NN to dynamically load the symbols.
* ```GetBackendId()```: must return the unique id of the dynamic backends.
  If at the time of the loading the id already exists in the internal Arm NN's backend registry, the backend will be skipped and
  not loaded in Arm NN
* ```GetVersion()```: must return the version of the dynamic backend.
  The version must indicate the version of the Backend API the dynamic backend has been built with.
  The current Backend API version can be found by inspecting the IBackendInternal interface.
  At the time of loading, the version of the backend will be checked against the version of the Backend API Arm NN is built with.
  If the backend version is not compatible with the current Backend API, the backend will not be loaded as it will be assumed that
  it is not ABI compatible with the current Arm NN build.
* ```BackendFactory()```: must return a valid instance of the backend.
  The backend instance is an object that must inherit from the version of the IBackendInternal interface declared by GetVersion().
  It is the backend developer's responsibility to ensure that the backend implementation correctly reflects the version declared by
  GetVersion(), and that the object returned by the BackendFactory() function is a valid and well-formed instance of the IBackendInternal
  interface.

## Dynamic backend versioning and ABI compatibility

Dynamic backend versioning policy:

Updates to Arm NN's Backend API follow these rules: changes to the Backend API (the IBackendInternal interface) that break
ABI compatibility with the previous API version will be indicated by a change of the API's major version, while changes
that guarantee ABI compatibility with the previous API version will be indicated by a change in API's the minor version.

For example:

* Dynamic backend version 2.4 (i.e. built with Backend API version 2.4) is compatible with Arm NN's Backend API version 2.4
  (same version, backend built against the same Backend API)
* Dynamic backend version 2.1 (i.e. built with Backend API version 2.1) is compatible with Arm NN's Backend API version 2.4
  (same major version, backend built against earlier compatible API)
* Dynamic backend version 2.5 (i.e. built with Backend API version 2.5) is not compatible with Arm NN's Backend API version 2.4
  (same major version, backend built against later incompatible API, backend might require update to the latest compatible backend API)
* Dynamic backend version 2.0 (i.e. built with Backend API version 2.0) is not compatible with Arm NN's Backend API version 1.0
  (backend requires a completely new API version)
* Dynamic backend version 2.0 (i.e. built with Backend API version 2.0) is not compatible with Arm NN's Backend API version 3.0
  (backward compatibility in the Backend API is broken)

## Dynamic backend loading paths

During the creation of the Runtime, Arm NN will scan a given set of paths searching for suitable dynamic backend objects to load.
A list of (absolute) paths can be specified at compile-time by setting a define named ```DYNAMIC_BACKEND_PATHS``` in the form of a colon-separated list of strings.

```shell
-DDYNAMIC_BACKEND_PATHS="PATH_1:PATH_2...:PATH_N"
```

The paths will be processed in the same order as they are indicated in the macro.

It is also possible to override those paths at runtime when creating the Runtime object by setting the value of the ```m_DynamicBackendsPath``` member in the CreationOptions class.
Only one path is allowed for the override via the CreationOptions class.
By setting the value of the ```m_DynamicBackendsPath``` to a path in the filesystem, Arm NN will entirely ignore the list of paths passed via the
```DYNAMIC_BACKEND_PATHS``` compiler directive.

All the specified paths are validated before processing (they must exist, must be directories, and must be absolute paths),
in case of error a warning message will be added to the log, but Arm NN's execution will not be stopped.
If all paths are not valid, then no dynamic backends will be loaded in the Arm NN's runtime.

Passing an empty list of paths at compile-time and providing no path override at runtime will effectively disable the
dynamic backend loading feature, and no dynamic backends will be loaded into Arm NN's runtime.

## Dynamic backend file naming convention

During the creation of a Runtime object, Arm NN will scan the paths specified for dynamic backend loading searching for suitable backend objects.
Arm NN will try to load only the files that match the following accepted naming scheme:

```shell
<vendor>_<name>_backend.so[<version>] (e.g. "Arm_GpuAcc_backend.so" or "Arm_GpuAcc_backend.so.1.2.3")
```

Only alphanumeric characters are allowed for both the `<vendor>` and the `<name>` fields, namely lowercase and/or uppercase characters,
and/or numerical digits (see the table below for examples).
Only dots and numbers are allowed for the optional `<version>` field.

Symlinks to other files are allowed to support the standard linux shared object versioning:

```shell
Arm_GpuAcc_backend.so -> Arm_GpuAcc_backend.so.1.2.3
Arm_GpuAcc_backend.so.1 -> Arm_GpuAcc_backend.so.1.2.3
Arm_GpuAcc_backend.so.1.2 -> Arm_GpuAcc_backend.so.1.2.3
Arm_GpuAcc_backend.so.1.2.3
```

Files are identified by their full canonical path, so it is allowed to have files with the same name in different directories.
However, if those are actually the same dynamic backend, only the first in order of parsing will be loaded.

Examples:

| Filename                                                 | Description                                       |
| -------------------------------------------------------- | ------------------------------------------------- |
| Arm_GpuAcc_backend.so                                    | valid: basic backend name                         |
| Arm_GpuAcc_backend.so.1                                  | valid: single field version number                |
| Arm_GpuAcc_backend.so.1.2                                | valid: multiple field version number              |
| Arm_GpuAcc_backend.so.1.2.3                              | valid: multiple field version number              |
| Arm_GpuAcc_backend.so.10.1.27                            | valid: Multiple digit version                     |
| Arm_GpuAcc_backend.so.10.1.33.                           | not valid: dot not followed by version number     |
| Arm_GpuAcc_backend.so.3.4..5                             | not valid: dot not followed by version number     |
| Arm_GpuAcc_backend.so.1,1.1                              | not valid: comma instead of dot in the version    |
| Arm123_GpuAcc_backend.so                                 | valid: digits in vendor name are allowed          |
| Arm_GpuAcc456_backend.so                                 | valid: digits in backend id are allowed           |
| Arm%Co_GpuAcc_backend.so                                 | not valid: invalid character in vendor name       |
| Arm_Gpu.Acc_backend.so                                   | not valid: invalid character in backend id        |
| GpuAcc_backend.so                                        | not valid: missing vendor name                    |
| _GpuAcc_backend.so                                       | not valid: missing vendor name                    |
| Arm__backend.so                                          | not valid: missing backend id                     |
| Arm_GpuAcc.so                                            | not valid: missing "backend" at the end           |
| __backend.so                                             | not valid: missing vendor name and backend id     |
| __.so                                                    | not valid: missing all fields                     |
| Arm_GpuAcc_backend                                       | not valid: missing at least ".so" at the end      |
| Arm_GpuAcc_backend_v1.2.so                               | not valid: extra version info at the end          |
| Arm_CpuAcc_backend.so                                    | valid: basic backend name                         |
| Arm_CpuAcc_backend.so.1 -> Arm_CpuAcc_backend.so         | valid: symlink to valid backend file              |
| Arm_CpuAcc_backend.so.1.2 -> Arm_CpuAcc_backend.so.1     | valid: symlink to valid symlink                   |
| Arm_CpuAcc_backend.so.1.2.3 -> Arm_CpuAcc_backend.so.1.2 | valid: symlink to valid symlink                   |
| Arm_no_backend.so -> nothing                             | not valid: symlink resolves to non-existent file  |
| pathA/Arm_GpuAcc_backend.so                              | valid: basic backend name                         |
| pathB/Arm_GpuAcc_backend.so                              | valid: but duplicated from pathA/                 |

Arm NN will try to load the dynamic backends in the same order as they are parsed from the filesystem.

## Dynamic backend examples

The source code includes an example that is used to generate some mock dynamic backends for testing purposes. The source files are:

[TestDynamicBackend.hpp](backendsCommon/test/TestDynamicBackend.hpp)
[TestDynamicBackend.cpp](backendsCommon/test/TestDynamicBackend.cpp)

This example is useful for going through all the use cases that constitute an invalid dynamic backend object, such as
an invalid/malformed implementation of the shared object interface, or an invalid value returned by any of the interface methods
that would prevent Arm NN from making use of the dynamic backend.

A dynamic implementation of the reference backend is also provided. The source files are:

[RefDynamicBackend.hpp](dynamic/reference/RefDynamicBackend.hpp)
[RefDynamicBackend.cpp](dynamic/reference/RefDynamicBackend.cpp)

The implementation itself is quite simple and straightforward. Since an implementation of this particular backend was already available,
the dynamic version is just a wrapper around the original code that simply returns the backend id, version and an instance of the
backend itself via the factory function.
For the sake of the example, the source code of the reference backend is used to build the dynamic version (as you would for any new
dynamic backend), while all the other symbols needed are provided by linking the dynamic backend against Arm NN.

The makefile used for building the reference dynamic backend is also provided: [CMakeLists.txt](dynamic/reference/CMakeLists.txt)

A unit test that loads the reference backend dynamically and that exercises it is also included in the file
[DynamicBackendTests.cpp](dynamic/backendsCommon/test/DynamicBackendTests.cpp), by the test case ```CreateReferenceDynamicBackend```.
In the test, a path on the filesystem is scanned for valid dynamic backend files (using the override option in ```CreationOptions```)
where only the reference dynamic backend is.
In this example the file is named ```Arm_CpuRef_backend.so```, which is compliant with the expected naming scheme for dynamic backends.
A ```DynamicBackend``` is created in the runtime to represent the newly loaded backend, then the backend is registered in the Backend
Registry with the id "CpuRef" (returned by ```GetBackendId()```).
The unit test makes sure that the backend is actually registered in Arm NN, before trying to create an instance of the backend by
calling the factory function provided through the shared object interface (```BackendFactory()```).
The backend instance is used to verify that everything is in order, testing basic 2D convolution support by making use of the
Layer Support API and the Workload Factory.
At the end of test, the runtime object goes out of scope and the dynamic backend instance is automatically destroyed, and the handle to
the shared object is closed.
