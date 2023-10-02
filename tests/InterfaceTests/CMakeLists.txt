#
# Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Usage: cmake -DARMNN_ROOT=<path to armnn library and includes>

cmake_minimum_required (VERSION 3.7.0)
project(InterfaceTests)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Reusing the TfLite cmake module from Arm NN. This module should have no
# external dependencies on other parts of Arm NN. Its only required
# parameter is TFLITE_LIB_ROOT
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../../delegate/cmake/Modules/")

# This is the base path for the Arm NN binary build. The root directory
# contains the library files and also an include directory.
option(ARMNN_ROOT "Location where the Arm NN libraries and 'include' directory can be found" Off)

# TfLite needs flatbuffers.
option(FLATBUFFERS_ROOT "Location where Flatbuffers include files can be found." Off)

# To test both delegates we need TfLite.
option(TFLITE_INCLUDE_ROOT "Location where the TfLite includes can be found." Off)
option(TFLITE_LIB_ROOT "Location where the built TfLite libraries can be found." Off)


# Locate the Arm NN libraries and includes.
message(STATUS "Arm NN root is set to \"${ARMNN_ROOT}\"")
# Exclude default paths to ensure only the specified path is used.
find_path(ARMNN_LIB_INCLUDE armnn/ArmNN.hpp PATHS ${ARMNN_ROOT}/include NO_DEFAULT_PATH)
message(STATUS "Arm NN library include directory located at: ${ARMNN_LIB_INCLUDE}")
find_library(ARMNN_LIB
             NAMES armnn
             HINTS ${ARMNN_ROOT})
message(STATUS "Arm NN library location set to ${ARMNN_LIB}")

# Arm NN uses pthreads.
find_package(Threads)

# First test is the use of the Arm NN graph interface. We compile
# SimpleSample into a binary based on the libraries and includes we have
# found.
add_executable(SimpleSample ../../samples/SimpleSample.cpp)
# We're using BEFORE here to ensure no stray system libraries get used first.
target_include_directories(SimpleSample BEFORE PUBLIC ${ARMNN_LIB_INCLUDE})
target_link_libraries(SimpleSample ${ARMNN_LIB} ${CMAKE_THREAD_LIBS_INIT})

# TfLite parser.
find_library(ARMNN_TFLITE_LIB
             NAMES armnnTfLiteParser
             HINTS ${ARMNN_ROOT})
add_executable(TfLiteParserTest ./TfLiteParserTest.cpp)
target_include_directories(TfLiteParserTest BEFORE PUBLIC ${ARMNN_LIB_INCLUDE})
target_link_libraries(TfLiteParserTest ${ARMNN_LIB} ${ARMNN_TFLITE_LIB} ${CMAKE_THREAD_LIBS_INIT})

# Onnx parser
find_library(ARMNN_ONNX_LIB
             NAMES armnnOnnxParser
             HINTS ${ARMNN_ROOT})
add_executable(OnnxParserTest ./OnnxParserTest.cpp)
target_include_directories(OnnxParserTest BEFORE PUBLIC ${ARMNN_LIB_INCLUDE})
target_link_libraries(OnnxParserTest ${ARMNN_LIB} ${ARMNN_ONNX_LIB} ${CMAKE_THREAD_LIBS_INIT})


# These next two targets are for our TfLite delegates. The machanism used to
# compile and use them does not confirm to how Tf intends delegates to be
# used. We include these tests to highlight potential missing, or awkwrd,
# includes that could occur.

# Find Flatbuffers dependency
find_package(Flatbuffers REQUIRED MODULE)

# Find TfLite libraries and includes.
find_package(TfLite REQUIRED MODULE)
find_path(TFLITE_INCLUDE tensorflow/c/c_api.h PATHS ${TFLITE_INCLUDE_ROOT} NO_DEFAULT_PATH)

# Classic delegate
find_library(ARMNN_CLASSIC_DELEGATE_LIB
             NAMES armnnDelegate
             HINTS ${ARMNN_ROOT})
add_executable(ClassicDelegateTest ./ClassicDelegateTest.cpp)
find_path(ARMNN_COMMON_DELEGATE_INCLUDE DelegateOptions.hpp PATHS ${ARMNN_ROOT}/include/armnnDelegate/armnn/delegate/common/include/ NO_DEFAULT_PATH)
find_path(ARMNN_CLASSIC_DELEGATE_INCLUDE armnn_delegate.hpp PATHS ${ARMNN_ROOT}/include/armnnDelegate/armnn/delegate/classic/include/ NO_DEFAULT_PATH)
target_include_directories(ClassicDelegateTest BEFORE PUBLIC ${ARMNN_LIB_INCLUDE}
                                                             ${Flatbuffers_INCLUDE_DIR}
                                                             ${TFLITE_INCLUDE}
                                                             ${ARMNN_COMMON_DELEGATE_INCLUDE}
                                                             ${ARMNN_CLASSIC_DELEGATE_INCLUDE})
target_link_libraries(ClassicDelegateTest ${TfLite_LIB}
                                          ${ARMNN_CLASSIC_DELEGATE_LIB}
                                          ${ARMNN_LIB}
                                          ${Flatbuffers_LIB}
                                          ${CMAKE_THREAD_LIBS_INIT}
                                          ${CMAKE_DL_LIBS})

# Opaque delegate
find_library(ARMNN_OPAQUE_DELEGATE_LIB
             NAMES armnnOpaqueDelegate
             HINTS ${ARMNN_ROOT})
# Additional Absl Sync for Opaque Delegate
find_package(TfLiteAbsl REQUIRED MODULE)

add_executable(OpaqueDelegateTest ./OpaqueDelegateTest.cpp)
find_path(ARMNN_OPAQUE_DELEGATE_INCLUDE armnn_delegate.hpp PATHS ${ARMNN_ROOT}/include/armnnDelegate/armnn/delegate/opaque/include/ NO_DEFAULT_PATH)
# delegate_registry.h happens to use a ABSL mutex. We need to find and add its path too.
find_path(TFLITE_ABSL_INCLUDE absl/synchronization/mutex.h PATHS ${TFLITE_LIB_ROOT}/abseil-cpp/ NO_DEFAULT_PATH)
target_include_directories(OpaqueDelegateTest BEFORE PUBLIC ${ARMNN_LIB_INCLUDE}
                                                            ${Flatbuffers_INCLUDE_DIR}
                                                            ${TFLITE_INCLUDE}
                                                            ${TFLITE_ABSL_INCLUDE}
                                                            ${ARMNN_COMMON_DELEGATE_INCLUDE}
                                                            ${ARMNN_OPAQUE_DELEGATE_INCLUDE})
target_link_libraries(OpaqueDelegateTest ${ARMNN_OPAQUE_DELEGATE_LIB} 
                                         ${ARMNN_LIB}
                                         ${TfLite_Extra_Absl_LIB}
                                         ${TfLite_LIB}
                                         ${Flatbuffers_LIB}
                                         ${CMAKE_THREAD_LIBS_INIT}
                                         ${CMAKE_DL_LIBS})
