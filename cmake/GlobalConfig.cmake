#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
#

option(BUILD_ONNX_PARSER "Build Onnx parser" OFF)
option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_TESTS "Build test applications" OFF)
option(BUILD_FOR_COVERAGE "Use no optimization and output .gcno and .gcda files" OFF)
option(ARMCOMPUTENEON "Build with ARM Compute NEON support" OFF)
option(ARMCOMPUTECL "Build with ARM Compute OpenCL support" OFF)
option(ARMNNREF "Build with ArmNN reference support" ON)
option(ARMNNTOSAREF "Build with TOSA reference support" OFF)
option(PROFILING_BACKEND_STREAMLINE "Forward the armNN profiling events to DS-5/Streamline as annotations" OFF)
# options used for heap profiling and leak checking
option(HEAP_PROFILING "Build with heap profiling enabled" OFF)
option(LEAK_CHECKING "Build with leak checking enabled" OFF)
option(GPERFTOOLS_ROOT "Location where the gperftools 'include' and 'lib' folders to be found" Off)
# options used for tensorflow lite support
option(BUILD_TF_LITE_PARSER "Build Tensorflow Lite parser" OFF)
option(BUILD_ARMNN_SERIALIZER "Build Armnn Serializer" OFF)
option(BUILD_ACCURACY_TOOL "Build Accuracy Tool" OFF)
option(FLATC_DIR "Path to Flatbuffers compiler" OFF)
option(TF_LITE_GENERATED_PATH "Tensorflow lite generated C++ schema location" OFF)
option(FLATBUFFERS_ROOT "Location where the flatbuffers 'include' and 'lib' folders to be found" Off)
option(TOSA_SERIALIZATION_LIB_ROOT "Location where the TOSA Serialization Library 'include' and 'lib' folders can be found" OFF)
option(TOSA_REFERENCE_MODEL_ROOT "Location where the TOSA Reference Model 'include' and 'lib' folders can be found" OFF)
option(TOSA_REFERENCE_MODEL_OUTPUT "TOSA Reference Model output is printed during layer support checks" ON)
option(DYNAMIC_BACKEND_PATHS "Colon seperated list of paths where to load the dynamic backends from" "")
option(SAMPLE_DYNAMIC_BACKEND "Include the sample dynamic backend and its tests in the build" OFF)
option(BUILD_GATORD_MOCK "Build the Gatord simulator for external profiling testing." ON)
option(BUILD_TIMELINE_DECODER "Build the Timeline Decoder for external profiling." ON)
option(BUILD_BASE_PIPE_SERVER "Build the server to handle external profiling pipe traffic" ON)
option(BUILD_PYTHON_WHL "Build Python wheel package" OFF)
option(BUILD_PYTHON_SRC "Build Python source package" OFF)
option(BUILD_STATIC_PIPE_LIBS "Build Static PIPE libraries" OFF)
option(BUILD_PIPE_ONLY "Build the PIPE libraries only" OFF)
option(BUILD_CLASSIC_DELEGATE "Build the Arm NN TfLite delegate" OFF)
option(BUILD_OPAQUE_DELEGATE "Build the Arm NN TfLite Opaque delegate" OFF)
option(BUILD_MEMORY_STRATEGY_BENCHMARK "Build the MemoryBenchmark" OFF)
option(BUILD_BARE_METAL "Disable features requiring operating system support" OFF)
option(BUILD_SHARED_LIBS "Determines if Armnn will be built statically or dynamically.
                          This is an experimental feature and not fully supported.
                          Only the ArmNN core and the Delegate can be built statically." ON)
option(EXECUTE_NETWORK_STATIC " This is a limited experimental build that is entirely static.
                                It currently only supports being set by changing the current CMake default options like so:
                                BUILD_TF_LITE_PARSER=1/0
                                BUILD_ARMNN_SERIALIZER=1/0
                                ARMCOMPUTENEON=1/0
                                ARMNNREF=1/0
                                ARMCOMPUTECL=0
                                BUILD_ONNX_PARSER=0
                                BUILD_CLASSIC_DELEGATE=0
                                BUILD_OPAQUE_DELEGATE=0
                                BUILD_TIMELINE_DECODER=0
                                BUILD_BASE_PIPE_SERVER=0
                                BUILD_UNIT_TESTS=0
                                BUILD_GATORD_MOCK=0" OFF)

if(BUILD_ARMNN_TFLITE_DELEGATE)
    message(BUILD_ARMNN_TFLITE_DELEGATE option is deprecated, it will be removed in 24.02, please use BUILD_CLASSIC_DELEGATE instead)
    set(BUILD_CLASSIC_DELEGATE 1)
endif()

include(SelectLibraryConfigurations)

set(COMPILER_IS_GNU_LIKE 0)
if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU OR
   ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang OR
   ${CMAKE_CXX_COMPILER_ID} STREQUAL AppleClang)
    set(COMPILER_IS_GNU_LIKE 1)
endif()

# Enable CCache if available and not disabled
option(USE_CCACHE "USE_CCACHE" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND USE_CCACHE)
    get_property(rule_launch_compile DIRECTORY PROPERTY RULE_LAUNCH_COMPILE)
    set_property(DIRECTORY PROPERTY RULE_LAUNCH_COMPILE "CCACHE_CPP2=yes ${rule_launch_compile} ccache")
endif()

# Enable distcc if available and not disabled
option(USE_DISTCC "USE_DISTCC" OFF)
find_program(DISTCC_FOUND distcc)
if(DISTCC_FOUND AND USE_DISTCC)
    get_property(rule_launch_compile DIRECTORY PROPERTY RULE_LAUNCH_COMPILE)
    set_property(DIRECTORY PROPERTY RULE_LAUNCH_COMPILE "${rule_launch_compile} distcc")
endif()

# Set to release configuration by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Compiler flags that are always set
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(COMPILER_IS_GNU_LIKE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Werror -Wold-style-cast -Wno-missing-braces -Wconversion -Wsign-conversion")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wno-psabi")
    endif()
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    # Disable C4996 (use of deprecated identifier) due to
    # https://developercommunity.visualstudio.com/content/problem/252574/deprecated-compilation-warning-for-virtual-overrid.html
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /wd4996")
    add_definitions(-DNO_STRICT=1)
endif()
if("${CMAKE_SYSTEM_NAME}" STREQUAL Android)
    # -lz is necessary for when building with ACL set with compressed kernels
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -llog -lz")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -llog -lz")
endif()

# Compiler flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")
if(COMPILER_IS_GNU_LIKE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /O2")
endif()

# Compiler flags for Debug builds
if(COMPILER_IS_GNU_LIKE)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd /ZI /Od")
    # Disable SAFESEH which is necessary for Edit and Continue to work
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG  "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG  "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /SAFESEH:NO")
endif()

# Modify RelWithDebInfo so that NDEBUG isn't defined.
# This enables asserts.
if (COMPILER_IS_GNU_LIKE)
    string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

# Compiler flags for code coverage measurements
if(BUILD_FOR_COVERAGE)
    if(NOT CMAKE_BUILD_TYPE EQUAL "Debug")
        message(WARNING "BUILD_FOR_COVERAGE set so forcing to Debug build")
        set(CMAKE_BUILD_TYPE "Debug")
    endif()

    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif()

if(BUILD_FOR_COVERAGE AND NOT BUILD_UNIT_TESTS)
    message(WARNING "BUILD_FOR_COVERAGE set but not BUILD_UNIT_TESTS, so code coverage will not be able to run")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules ${CMAKE_MODULE_PATH})

include(CMakeFindDependencyMacro)


if(EXECUTE_NETWORK_STATIC)
    add_definitions(-DARMNN_DISABLE_SOCKETS
                    -DBUILD_SHARED_LIBS=0
                    -DARMNN_EXECUTE_NETWORK_STATIC)
endif()

if(BUILD_BARE_METAL)
    add_definitions(-DARMNN_BUILD_BARE_METAL
            -DARMNN_DISABLE_FILESYSTEM
            -DARMNN_DISABLE_PROCESSES
            -DARMNN_DISABLE_THREADS
            -DARMNN_DISABLE_SOCKETS)
endif()

if (NOT BUILD_PIPE_ONLY)
  # cxxopts (Alternative to boost::program_options)
  find_path(CXXOPTS_INCLUDE cxxopts/cxxopts.hpp PATHS third-party NO_CMAKE_FIND_ROOT_PATH)
  include_directories(SYSTEM "${CXXOPTS_INCLUDE}")
endif()

if (NOT BUILD_PIPE_ONLY)
  # ghc (Alternative to boost::filesystem)
  find_path(GHC_INCLUDE ghc/filesystem.hpp PATHS third-party NO_CMAKE_FIND_ROOT_PATH)
  include_directories(SYSTEM "${GHC_INCLUDE}")
endif()

# JNI_BUILD has DBUILD_SHARED_LIBS set to 0 and not finding libs while building
# hence added NOT BUILD_CLASSIC_DELEGATE/BUILD_OPAQUE_DELEGATE condition
if(NOT BUILD_SHARED_LIBS AND NOT BUILD_CLASSIC_DELEGATE AND NOT BUILD_OPAQUE_DELEGATE)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a .lib)
endif()

# pthread
if (NOT BUILD_BARE_METAL)
find_package(Threads)
endif()

if (EXECUTE_NETWORK_STATIC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++ -static-libgcc -static -pthread")
endif()

# Favour the protobuf passed on command line
if(BUILD_ONNX_PARSER)
    find_library(PROTOBUF_LIBRARY_DEBUG NAMES "protobufd"
        PATHS ${PROTOBUF_ROOT}/lib
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(PROTOBUF_LIBRARY_DEBUG NAMES "protobufd")

    find_library(PROTOBUF_LIBRARY_RELEASE NAMES "protobuf"
        PATHS ${PROTOBUF_ROOT}/lib
        NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(PROTOBUF_LIBRARY_RELEASE NAMES "protobuf")

    select_library_configurations(PROTOBUF)

    find_path(PROTOBUF_INCLUDE_DIRS "google/protobuf/message.h"
              PATHS ${PROTOBUF_ROOT}/include
              NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_path(PROTOBUF_INCLUDE_DIRS "google/protobuf/message.h")

    include_directories(SYSTEM "${PROTOBUF_INCLUDE_DIRS}")
    add_definitions(-DPROTOBUF_USE_DLLS)

    add_definitions(-DARMNN_ONNX_PARSER)

    find_path(ONNX_GENERATED_SOURCES "onnx/onnx.pb.cc")

    # C++ headers generated for onnx protobufs
    include_directories(SYSTEM "${ONNX_GENERATED_SOURCES}")
endif()

if(BUILD_CLASSIC_DELEGATE)
    add_definitions(-DARMNN_TFLITE_DELEGATE)
endif()

if(BUILD_OPAQUE_DELEGATE)
    add_definitions(-DARMNN_TFLITE_OPAQUE_DELEGATE)
endif()

# Flatbuffers support for TF Lite, Armnn Serializer or the TOSA backend.
if(BUILD_TF_LITE_PARSER OR BUILD_ARMNN_SERIALIZER OR ARMNNTOSAREF)
    # verify we have a valid flatbuffers include path
    find_path(FLATBUFFERS_INCLUDE_PATH flatbuffers/flatbuffers.h
              HINTS ${FLATBUFFERS_ROOT}/include /usr/local/include /usr/include)

    message(STATUS "Flatbuffers headers are located at: ${FLATBUFFERS_INCLUDE_PATH}")

    find_library(FLATBUFFERS_LIBRARY
                 NAMES libflatbuffers.a flatbuffers
                 HINTS ${FLATBUFFERS_ROOT}/lib /usr/local/lib /usr/lib)

    message(STATUS "Flatbuffers library located at: ${FLATBUFFERS_LIBRARY}")
endif()

# Flatbuffers schema support for TF Lite
if(BUILD_TF_LITE_PARSER)
    find_path(TF_LITE_SCHEMA_INCLUDE_PATH
              schema_generated.h
              HINTS ${TF_LITE_GENERATED_PATH})

    message(STATUS "Tf Lite generated header found at: ${TF_LITE_SCHEMA_INCLUDE_PATH}")

    add_definitions(-DARMNN_TF_LITE_PARSER)
endif()

if(BUILD_ARMNN_SERIALIZER)
    add_definitions(-DARMNN_SERIALIZER)
    add_definitions(-DARMNN_SERIALIZER_SCHEMA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/src/armnnSerializer/ArmnnSchema.fbs")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/profiling)

# ARM Compute
# Note that ARM Compute has a different folder layout depending on the branch but also on
# whether it comes from a prepackaged archive (this is why we add several hints below)
if(ARMCOMPUTENEON OR ARMCOMPUTECL)
    find_path(ARMCOMPUTE_INCLUDE arm_compute/core/CL/OpenCL.h
              PATHS ${ARMCOMPUTE_ROOT}/include
              PATHS ${ARMCOMPUTE_ROOT}/applications/arm_compute
              PATHS ${ARMCOMPUTE_ROOT}
              NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_path(ARMCOMPUTE_INCLUDE arm_compute/core/CL/OpenCL.h)
    include_directories(SYSTEM "${ARMCOMPUTE_INCLUDE}")

    # Find the Arm Compute libraries if not already specified (the user may have already defined this in advance,
    # e.g. if building clframework as a dependent cmake project)
    if (NOT DEFINED ARMCOMPUTE_LIBRARIES)
        # We link to the static variant so that customers don't need to find and build a compatible version of clframework.
        # First try the folders specified ARMCOMPUTE_BUILD_DIR (with PATH_SUFFIXES for
        # Windows builds)
        if ((NOT DEFINED ARMCOMPUTE_BUILD_DIR) AND (DEFINED ARMCOMPUTE_ROOT))
            # Default build directory for ComputeLibrary is under the root
            set(ARMCOMPUTE_BUILD_DIR ${ARMCOMPUTE_ROOT}/build)
        endif()

        find_library(ARMCOMPUTE_LIBRARY_DEBUG NAMES arm_compute-static
                     PATHS ${ARMCOMPUTE_BUILD_DIR}
                     PATH_SUFFIXES "Debug"
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        find_library(ARMCOMPUTE_LIBRARY_RELEASE NAMES arm_compute-static
                     PATHS ${ARMCOMPUTE_BUILD_DIR}
                     PATH_SUFFIXES "Release"
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        find_library(ARMCOMPUTE_CORE_LIBRARY_DEBUG NAMES arm_compute_core-static
                     PATHS ${ARMCOMPUTE_BUILD_DIR}
                     PATH_SUFFIXES "Debug"
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        find_library(ARMCOMPUTE_CORE_LIBRARY_RELEASE NAMES arm_compute_core-static
                     PATHS ${ARMCOMPUTE_BUILD_DIR}
                     PATH_SUFFIXES "Release"
                     NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

        # In case it wasn't there, try a default search (will work in cases where
        # the library has been installed into a standard location)
        find_library(ARMCOMPUTE_LIBRARY_DEBUG NAMES arm_compute-static)
        find_library(ARMCOMPUTE_LIBRARY_RELEASE NAMES arm_compute-static)
        find_library(ARMCOMPUTE_CORE_LIBRARY_DEBUG NAMES arm_compute_core-static)
        find_library(ARMCOMPUTE_CORE_LIBRARY_RELEASE NAMES arm_compute_core-static)

        # In case it wasn't there, try the dynamic libraries
        # This case will get used in a linux setup where the Compute Library
        # has been installed in a standard system library path as a dynamic library
        find_library(ARMCOMPUTE_LIBRARY_DEBUG NAMES arm_compute)
        find_library(ARMCOMPUTE_LIBRARY_RELEASE NAMES arm_compute)
        find_library(ARMCOMPUTE_CORE_LIBRARY_DEBUG NAMES arm_compute_core)
        find_library(ARMCOMPUTE_CORE_LIBRARY_RELEASE NAMES arm_compute_core)

        set(ARMCOMPUTE_LIBRARIES
            debug ${ARMCOMPUTE_LIBRARY_DEBUG} ${ARMCOMPUTE_CORE_LIBRARY_DEBUG}
            optimized ${ARMCOMPUTE_LIBRARY_RELEASE} ${ARMCOMPUTE_CORE_LIBRARY_RELEASE} )
    endif()
endif()

# ARM Compute NEON backend
if(ARMCOMPUTENEON)
    # Add preprocessor definition for ARM Compute NEON
    add_definitions(-DARMCOMPUTENEON_ENABLED)
    # The ARM Compute headers contain some NEON intrinsics, so we need to build armnn with NEON support on armv7
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES armv7 AND COMPILER_IS_GNU_LIKE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    endif()
endif()

# ARM Compute OpenCL backend
if(ARMCOMPUTECL)
    # verify we have a valid flatbuffers include path
    find_path(FLATBUFFERS_INCLUDE_PATH flatbuffers/flatbuffers.h
              HINTS ${FLATBUFFERS_ROOT}/include /usr/local/include /usr/include)

    message(STATUS "Flatbuffers headers are located at: ${FLATBUFFERS_INCLUDE_PATH}")

    find_library(FLATBUFFERS_LIBRARY
                 NAMES libflatbuffers.a flatbuffers
                 HINTS ${FLATBUFFERS_ROOT}/lib /usr/local/lib /usr/lib)

    message(STATUS "Flatbuffers library located at: ${FLATBUFFERS_LIBRARY}")

    # Always use Arm compute library OpenCL headers
    find_path(OPENCL_INCLUDE CL/opencl.hpp
              PATHS ${ARMCOMPUTE_ROOT}/include
              NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    # Link against libOpenCL in opencl-1.2-stubs, but don't search there at runtime
    link_libraries(-L${ARMCOMPUTE_BUILD_DIR}/opencl-1.2-stubs)
    set(OPENCL_LIBRARIES OpenCL)

    include_directories(SYSTEM ${OPENCL_INCLUDE})

    # Add preprocessor definition for ARM Compute OpenCL
    add_definitions(-DARMCOMPUTECL_ENABLED)

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DARM_COMPUTE_DEBUG_ENABLED")
endif()

# Used by both Arm Compute backends, but should be added
# to the search path after the system directories if necessary
if(ARMCOMPUTENEON OR ARMCOMPUTECL)
    find_path(HALF_INCLUDE half/half.hpp)
    find_path(HALF_INCLUDE half/half.hpp
              PATHS ${ARMCOMPUTE_ROOT}/include
              NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    include_directories(SYSTEM ${HALF_INCLUDE})
endif()

# ArmNN reference backend
if(ARMNNREF)
    add_definitions(-DARMNNREF_ENABLED)
endif()

# If a backend requires TOSA common, add it here.
if(ARMNNTOSAREF)
    set(ARMNNTOSACOMMON ON)
endif()

if(ARMNNTOSACOMMON)
    # Locate the includes for the TOSA serialization library as it is needed for TOSA common and TOSA backends.
    message(STATUS "TOSA serialization library root set to ${TOSA_SERIALIZATION_LIB_ROOT}")

    find_path(TOSA_SERIALIZATION_LIB_INCLUDE tosa_serialization_handler.h
              HINTS ${TOSA_SERIALIZATION_LIB_ROOT}/include)
    message(STATUS "TOSA serialization library include directory located at: ${TOSA_SERIALIZATION_LIB_INCLUDE}")

    find_library(TOSA_SERIALIZATION_LIB
                 NAMES tosa_serialization_lib.a tosa_serialization_lib
                 HINTS ${TOSA_SERIALIZATION_LIB_ROOT}/lib /usr/local/lib /usr/lib)
    message(STATUS "TOSA serialization library set to ${TOSA_SERIALIZATION_LIB}")

    # Include required headers for TOSA Serialization Library
    include_directories(SYSTEM ${FLATBUFFERS_INCLUDE_PATH})
    include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third-party/half)
    include_directories(SYSTEM ${TOSA_SERIALIZATION_LIB_INCLUDE})
endif()

# ArmNN TOSA reference backend
if(ARMNNTOSAREF)
    # Locate the includes for the TOSA Reference Model, which is specific to the TOSA Reference Backend.
    message(STATUS "TOSA Reference Model root set to ${TOSA_REFERENCE_MODEL_ROOT}")

    find_path(TOSA_REFERENCE_MODEL_INCLUDE model_runner.h
              HINTS ${TOSA_REFERENCE_MODEL_ROOT}/include)
    message(STATUS "TOSA Reference Model include directory located at: ${TOSA_REFERENCE_MODEL_INCLUDE}")

    include_directories(SYSTEM ${TOSA_REFERENCE_MODEL_INCLUDE})

    find_library(TOSA_REFERENCE_MODEL_LIB
                 NAMES tosa_reference_model_lib.a tosa_reference_model_lib
                 HINTS ${TOSA_REFERENCE_MODEL_ROOT}/lib /usr/local/lib /usr/lib)
    message(STATUS "TOSA Reference Model set to ${TOSA_REFERENCE_MODEL_LIB}")

    if(TOSA_REFERENCE_MODEL_OUTPUT)
        add_definitions("-DTOSA_REFERENCE_MODEL_OUTPUT=1")
    endif()
endif()

# This is the root for the dynamic backend tests to search for dynamic
# backends. By default it will be the project build directory.
add_definitions(-DDYNAMIC_BACKEND_BUILD_DIR="${PROJECT_BINARY_DIR}")

# ArmNN dynamic backend
if(DYNAMIC_BACKEND_PATHS)
    add_definitions(-DARMNN_DYNAMIC_BACKEND_ENABLED)
endif()

if(SAMPLE_DYNAMIC_BACKEND)
    add_definitions(-DSAMPLE_DYNAMIC_BACKEND_ENABLED)
endif()

# Streamline annotate
if(PROFILING_BACKEND_STREAMLINE)
    include_directories("${GATOR_ROOT}/annotate")
    add_definitions(-DARMNN_STREAMLINE_ENABLED)
endif()

if(NOT BUILD_BARE_METAL AND NOT EXECUTE_NETWORK_STATIC)
if(HEAP_PROFILING OR LEAK_CHECKING)
    find_path(HEAP_PROFILER_INCLUDE gperftools/heap-profiler.h
            PATHS ${GPERFTOOLS_ROOT}/include
            NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    include_directories(SYSTEM "${HEAP_PROFILER_INCLUDE}")
    find_library(GPERF_TOOLS_LIBRARY
                NAMES tcmalloc_debug
                HINTS ${GPERFTOOLS_ROOT}/lib)
    link_directories(${GPERFTOOLS_ROOT}/lib)

    link_libraries(${GPERF_TOOLS_LIBRARY})
    if (HEAP_PROFILING)
        add_definitions("-DARMNN_HEAP_PROFILING_ENABLED=1")
    endif()
    if (LEAK_CHECKING)
        add_definitions("-DARMNN_LEAK_CHECKING_ENABLED=1")
    endif()
else()
    # Valgrind only works with gperftools version number <= 2.4
    CHECK_INCLUDE_FILE(valgrind/memcheck.h VALGRIND_FOUND)
endif()
endif()

if(NOT BUILD_TF_LITE_PARSER)
    message(STATUS "Tensorflow Lite parser support is disabled")
endif()

if(NOT BUILD_ARMNN_SERIALIZER)
    message(STATUS "Armnn Serializer support is disabled")
endif()

if(NOT BUILD_PYTHON_WHL)
    message(STATUS "PyArmNN wheel package is disabled")
endif()

if(NOT BUILD_PYTHON_SRC)
    message(STATUS "PyArmNN source package is disabled")
endif()

if(BUILD_PYTHON_WHL OR BUILD_PYTHON_SRC)
    find_package(PythonInterp 3 REQUIRED)
    if(NOT ${PYTHONINTERP_FOUND})
        message(FATAL_ERROR "Python 3.x required to build PyArmNN, but not found")
    endif()

    find_package(PythonLibs 3 REQUIRED)
    if(NOT ${PYTHONLIBS_FOUND})
        message(FATAL_ERROR "Python 3.x development package required to build PyArmNN, but not found")
    endif()

    find_package(SWIG 4 REQUIRED)
    if(NOT ${SWIG_FOUND})
        message(FATAL_ERROR "SWIG 4.x requried to build PyArmNN, but not found")
    endif()
endif()

# ArmNN source files required for all build options
include_directories(SYSTEM third-party)
