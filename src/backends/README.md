# Backend developer guide

ArmNN allows adding new backends through the 'Pluggable Backend' mechanism.

## How to add a new backend

Backends reside under src/backends in separate subfolders. They must have a ```backend.cmake``` file
which is read automatically by [src/backends/backends.cmake](backends.cmake). The ```backend.cmake``` file
under the backend specific folder is then included by the main CMakeLists.txt file at the root of the
ArmNN source tree.

### The backend.cmake file

The ```backend.cmake``` has two main purposes:

1. It makes sure the artifact (a cmake OBJECT library) is linked into the ArmNN shared library.
2. It makes sure that the subdirectory where backend sources reside gets included in the build.

To achieve this there are two requirements for the ```backend.cmake``` file
(example taken from [reference/backend.cmake](reference/backend.cmake))

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
```

### The CMakeLists.txt file

As described in the previous section, adding a new backend will require creating a CMakeLists.txt in
the backend folder. This follows the standard cmake conventions, and is required to build an artifact
to be linked into the ArmNN shared library. As with any cmake build, the code can be structured into
subfolders and modules as the developer sees fit.

Example can be found under [reference/CMakeLists.txt](reference/CMakeLists.txt).

### The backend.mk file

ArmNN on Android uses the native Android build system. New backends are integrated by creating a
```backend.mk``` file which has a single variable called ```BACKEND_SOURCES``` listing all cpp
files to be built by the Android build system.

Example taken from [reference/backend.mk](reference/backend.mk):

```make
BACKEND_SOURCES := \
        RefLayerSupport.cpp \
        RefWorkloadFactory.cpp \
        workloads/Activation.cpp \
        workloads/ArithmeticFunction.cpp \
        workloads/Broadcast.cpp \
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
