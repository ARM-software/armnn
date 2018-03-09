option(BUILD_CAFFE_PARSER "Build Caffe parser" OFF)
option(BUILD_TF_PARSER "Build Tensorflow parser" OFF)
option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_TESTS "Build test applications" OFF)
option(BUILD_FOR_COVERAGE "Use no optimization and output .gcno and .gcda files" OFF)
option(ARMCOMPUTENEON "Build with ARM Compute NEON support" OFF)
option(ARMCOMPUTECL "Build with ARM Compute OpenCL support" OFF)
option(PROFILING "Build with ArmNN built-in profiling support" OFF)
option(PROFILING_BACKEND_STREAMLINE "Forward the armNN profiling events to DS-5/Streamline as annotations" OFF)

include(SelectLibraryConfigurations)

set(COMPILER_IS_GNU_LIKE 0)
if(${CMAKE_CXX_COMPILER_ID} STREQUAL GNU OR ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Werror -Wold-style-cast -Wno-missing-braces -Wconversion -Wsign-conversion")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP")
    add_definitions(-DNOMINMAX=1 -DNO_STRICT=1)
endif()
if("${CMAKE_SYSTEM_NAME}" STREQUAL Android)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -llog")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -llog")
endif()

# Compiler flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")
if(COMPILER_IS_GNU_LIKE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD /O2")
endif()

# Compiler flags for Debug builds
if(COMPILER_IS_GNU_LIKE)
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd /ZI /Od")
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

# Boost
add_definitions("-DBOOST_ALL_NO_LIB") # Turn off auto-linking as we specify the libs manually
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.59 REQUIRED COMPONENTS unit_test_framework system filesystem log program_options)
include_directories(SYSTEM "${Boost_INCLUDE_DIR}")
link_directories(${Boost_LIBRARY_DIR})

# pthread
find_package (Threads)

# Favour the protobuf passed on command line
if(BUILD_TF_PARSER OR BUILD_CAFFE_PARSER)
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
endif()

# Caffe and its dependencies
if(BUILD_CAFFE_PARSER)
    add_definitions(-DARMNN_CAFFE_PARSER)

    find_path(CAFFE_GENERATED_SOURCES "caffe/proto/caffe.pb.h"
        HINTS ${CAFFE_BUILD_ROOT}/include)
    include_directories(SYSTEM "${CAFFE_GENERATED_SOURCES}")
endif()

if(BUILD_TF_PARSER)
    add_definitions(-DARMNN_TF_PARSER)

    find_path(TF_GENERATED_SOURCES "tensorflow/core/protobuf/saved_model.pb.cc")

    # C++ sources generated for tf protobufs
    file(GLOB_RECURSE TF_PROTOBUFS "${TF_GENERATED_SOURCES}/*.pb.cc")

    # C++ headers generated for tf protobufs
    include_directories(SYSTEM "${TF_GENERATED_SOURCES}")
endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ARM Compute
# Note that ARM Compute has a different folder layout depending on the branch but also on
# whether it comes from a prepackaged archive (this is why we add several hints below)
if(ARMCOMPUTENEON OR ARMCOMPUTECL)
    find_path(ARMCOMPUTE_INCLUDE arm_compute/core/CL/ICLKernel.h
              PATHS ${ARMCOMPUTE_ROOT}/include
              PATHS ${ARMCOMPUTE_ROOT}/applications/arm_compute
              PATHS ${ARMCOMPUTE_ROOT}
              NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_path(ARMCOMPUTE_INCLUDE arm_compute/core/CL/ICLKernel.h)
    include_directories(SYSTEM "${ARMCOMPUTE_INCLUDE}")

    # Find the Arm Compute libraries if not already specified (the user may have already defined this in advance,
    # e.g. if building clframework as a dependent cmake project)
    if (NOT DEFINED ARMCOMPUTE_LIBRARIES)
        # We link to the static variant so that customers don't need to find and build a compatible version of clframework.
        # First try the folders specified ARMCOMPUTE_BUILD_DIR (with PATH_SUFFIXES for
        # Windows builds)
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
    # Always use Arm compute library OpenCL headers
    find_path(OPENCL_INCLUDE CL/cl2.hpp
              PATHS ${ARMCOMPUTE_ROOT}/include
              NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    find_library(OPENCL_LIBRARIES OpenCL)
    if (NOT OPENCL_LIBRARIES)
        # Link against libOpenCL in opencl-1.2-stubs, but don't search there at runtime
        link_libraries(-L${ARMCOMPUTE_BUILD_DIR}/opencl-1.2-stubs)
        set(OPENCL_LIBRARIES OpenCL)
    endif()

    include_directories(${OPENCL_INCLUDE})

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
    include_directories(${HALF_INCLUDE})
endif()

# Built-in profiler
if(PROFILING)
    add_definitions(-DARMNN_PROFILING_ENABLED)
endif()

# Streamline annotate
if(PROFILING_BACKEND_STREAMLINE)
    include_directories("${GATOR_ROOT}/annotate")
    add_definitions(-DARMNN_STREAMLINE_ENABLED)
endif()

