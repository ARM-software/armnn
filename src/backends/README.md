# Backend developer guide

ArmNN allows adding new backends through the 'Pluggable Backend' mechanism.

## How to add a new backend

Backends reside under [src/backends](./) in separate subfolders. For Linux builds they must have a ```backend.cmake``` file
which is read automatically by [src/backends/backends.cmake](backends.cmake). The ```backend.cmake``` file
under the backend specific folder is then included by the main CMakeLists.txt file at the root of the
ArmNN source tree.

### The backend.cmake file

The ```backend.cmake``` has three main purposes:

1. It makes sure the artifact (a cmake OBJECT library) is linked into the ArmNN shared library by appending the name of the library to the ```armnnLibraries``` list.
2. It makes sure that the subdirectory where backend sources reside gets included in the build.
3. To include backend specific unit tests, the object library for the unit tests needs to be added to the ```armnnUnitTestLibraries``` list.


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
# UnitTests executable can run the backend specific unit
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

Optionally, backend specific unit tests can be added similarly, by
appending the list of cpp files to the ```BACKEND_TEST_SOURCES``` variable.

Example taken from [reference/backend.mk](reference/backend.mk):

```make
BACKEND_SOURCES := \
        RefLayerSupport.cpp \
        RefWorkloadFactory.cpp \
        workloads/Activation.cpp \
        workloads/ArithmeticFunction.cpp \
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

## Registry interfaces

To integrate a new backend, it needs to be registered through the following singleton classes:

* [BackendRegistry](backendsCommon/BackendRegistry.hpp)
* [LayerSupportRegistry](backendsCommon/LayerSupportRegistry.hpp)

These Registries can register a factory function together with a [BackendId](../../include/armnn/BackendId.hpp)
key, that allows to create a registered object at a later time when needed.

There is support for statically registering the factory functions in
the [RegistryCommon.hpp](backendsCommon/RegistryCommon.hpp) header.

### The BackendRegistry and the IBackendInternal interface

The ```BackendRegistry``` registers a function that returns a unique pointer to an object that implements the [IBackendInternal interface](backendsCommon/IBackendInternal.hpp).

During optimization we assign backends to the layers in the network.
When we pass the resulting optimized network to the ```LoadedNetwork```,
it calls the factory function for all backends that are required by the
given network. The ```LoadedNetwork``` holds a reference to these
backend objects for its lifetime.

The ```LoadedNetwork``` also calls the ```CreateWorkloadFactory()```
function for each created backend once and uses the returned backend
specific workload factory object to create the workloads for the
layers. The ```LoadedNetwork``` holds the workload factory objects for
its lifetime.

Examples for this concept can be found in the [RefBackend header](reference/RefBackend.hpp) and the [RefBackend implementation](reference/RefBackend.cpp).

### The LayerSupportRegistry and the ILayerSupport object

ArmNN uses the [ILayerSupport](../../include/armnn/ILayerSupport.hpp)
interface to decide if a layer with a set of parameters (ie. input and
output tensors, descriptor, weights, filter, kernel if any) are
supported on a given backend. The backends need a way to communicate
this information.

In order to achieve this, the backends need to register a factory function
that can create an object that implements the ILayerSupport interface.
When ArmNN needs to decide if a layer is supported on a backend, it
looks up the factory function through the registry. Then it creates an
ILayerSupport object for the given backend and calls the corresponding
API function to check if the layer is supported.

Examples of this can be found in the [RefLayerSupport header](reference/RefLayerSupport.hpp)
and the [RefLayerSupport implementation](reference/RefLayerSupport.cpp).