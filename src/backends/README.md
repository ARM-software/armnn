# Backend developer guide

ArmNN allows adding new backends through the 'Pluggable Backend' mechanism.

## How to add a new backend

Backends reside under [src/backends](./), in separate subfolders. For Linux builds they must have a ```backend.cmake``` file
which is read automatically by [src/backends/backends.cmake](backends.cmake). The ```backend.cmake``` file
under the backend-specific folder is then included by the main CMakeLists.txt file at the root of the
ArmNN source tree.

### The backend.cmake file

The ```backend.cmake``` has three main purposes:

1. It makes sure the artifact (a cmake OBJECT library) is linked into the ArmNN shared library by appending the name of the library to the ```armnnLibraries``` list.
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
# list of libraries linked against the ArmNN shared library.
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
to be linked into the ArmNN shared library. As with any cmake build, the code can be structured into
subfolders and modules as the developer sees fit.

Example can be found under [reference/CMakeLists.txt](reference/CMakeLists.txt).

### The backend.mk file

ArmNN on Android uses the native Android build system. New backends are integrated by creating a
```backend.mk``` file which has a single variable called ```BACKEND_SOURCES``` listing all cpp
files to be built by the Android build system for the ArmNN shared library.

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
with previous ArmNN versions.

## The IBackendInternal interface

All backends need to implement the [IBackendInternal](backendsCommon/IBackendInternal.hpp) interface.
The interface functions to be implemented are:

```c++
    virtual IMemoryManagerUniquePtr CreateMemoryManager() const = 0;
    virtual IWorkloadFactoryPtr CreateWorkloadFactory(
            const IMemoryManagerSharedPtr& memoryManager = nullptr) const = 0;
    virtual IBackendContextPtr CreateBackendContext(const IRuntime::CreationOptions&) const = 0;
    virtual ILayerSupportSharedPtr GetLayerSupport() const = 0;
    virtual Optimizations GetOptimizations() const = 0;
    virtual SubgraphUniquePtr OptimizeSubgraph(const SubgraphView& subgraph, bool& optimizationAttempted) const;
    virtual OptimizationViews OptimizeSubgraphView(const SubgraphView& subgraph) const;
```

Note that ```GetOptimizations()``` and ```SubgraphViewUniquePtr OptimizeSubgraphView(const SubgraphView& subgraph, bool& optimizationAttempted)```
have been deprecated.
The method ```OptimizationViews OptimizeSubgraph(const SubgraphView& subgraph)``` should be used instead to
apply specific optimizations to a given sub-graph.

The ArmNN framework then creates instances of the IBackendInternal interface with the help of the
[BackendRegistry](backendsCommon/BackendRegistry.hpp) singleton.

**Important:** the ```IBackendInternal``` object is not guaranteed to have a longer lifetime than
the objects it creates. It is only intended to be a single entry point for the factory functions it has.
The best use of this is to be a lightweight, stateless object and make no assumptions between
its lifetime and the lifetime of the objects it creates.

For each backend one needs to register a factory function that can
be retrieved using a [BackendId](../../include/armnn/BackendId.hpp).
The ArmNN framework creates the backend interfaces dynamically when
it sees fit and it keeps these objects for a short period of time. Examples:

* During optimization ArmNN needs to decide which layers are supported by the backend.
  To do this, it creates a backends and calls the ```GetLayerSupport()``` function and creates
  an ```ILayerSupport``` object to help deciding this.
* During optimization ArmNN can run backend-specific optimizations. After splitting the graph into
  sub-graphs based on backends, it calls the ```OptimizeSubgraphView()``` function on each of them and, if possible,
  substitutes the corresponding sub-graph in the original graph with its optimized version.
* When the Runtime is initialized it creates an optional ```IBackendContext``` object and keeps this context alive
  for the Runtime's lifetime. It notifies this context object before and after a network is loaded or unloaded.
* When the LoadedNetwork creates the backend-specific workloads for the layers, it creates a backend
  specific workload factory and calls this to create the workloads.

## The BackendRegistry

As mentioned above, all backends need to be registered through the BackendRegistry so ArmNN knows
about them. Registration requires a unique backend ID string and a lambda function that
returns a unique pointer to the [IBackendInternal interface](backendsCommon/IBackendInternal.hpp).

For registering a backend only this lambda function needs to exist, not the actual backend. This
allows dynamically creating the backend objects when they are needed.

The BackendRegistry has a few convenience functions, like we can query the registered backends and
 are able to tell if a given backend is registered or not.

## The ILayerSupport interface

ArmNN uses the [ILayerSupport](../../include/armnn/ILayerSupport.hpp) interface to decide if a layer
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

Backends may choose to implement custom memory management. ArmNN supports this concept through the following
mechanism:

* the ```IBackendInternal``` interface has a ```CreateMemoryManager()``` function which is called before
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
  thus have been rejected. ArmNN will try to re-allocate these parts on other backends if available.
* A list of the untouched sub-graphs: these are the parts of the original sub-graph that have not been optimized,
  but that can run (unoptimized) on the backend.

The previous way backends had to provide a list optimizations to the Optimizer (through the ```GetOptimizations()``` method)
is still in place for backward compatibility, but it's now considered deprecated and will be remove in a future release.

## The IBackendContext interface

Backends may need to be notified whenever a network is loaded or unloaded. To support that, one can implement the optional
[IBackendContext](backendsCommon/IBackendContext.hpp) interface. The framework calls the ```CreateBackendContext(...)```
method for each backend in the Runtime. If the backend returns a valid unique pointer to a backend context, then the
runtime will hold this for its entire lifetime. It then calls the following interface functions for each stored context:

* ```BeforeLoadNetwork(NetworkId networkId)```
* ```AfterLoadNetwork(NetworkId networkId)```
* ```BeforeUnloadNetwork(NetworkId networkId)```
* ```AfterUnloadNetwork(NetworkId networkId)```
