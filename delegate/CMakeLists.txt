#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required (VERSION 3.8.0)
project(armnnDelegate)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(armnnDelegate_sources)
list(APPEND armnnDelegate_sources
        include/armnn_delegate.hpp
        include/DelegateOptions.hpp
        src/armnn_delegate.cpp
        src/DelegateOptions.cpp
        src/Activation.hpp
        src/ArgMinMax.hpp
        src/BatchSpace.hpp
        src/Comparison.hpp
        src/Convolution.hpp
        src/Control.hpp
        src/DelegateUtils.hpp
        src/ElementwiseBinary.hpp
        src/ElementwiseUnary.hpp
        src/Fill.hpp
        src/FullyConnected.hpp
        src/Gather.hpp
        src/Lstm.hpp
        src/Normalization.hpp
        src/Pad.hpp
        src/Pooling.hpp
        src/Quantization.hpp
        src/Redefine.hpp
        src/Resize.hpp
        src/Round.hpp
        src/Slice.hpp
        src/Softmax.hpp
        src/SpaceDepth.hpp
        src/Transpose.hpp)

add_library(armnnDelegate SHARED ${armnnDelegate_sources})

target_include_directories(armnnDelegate
        PUBLIC
            $<INSTALL_INTERFACE:include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src)

include(GNUInstallDirs)

## Add Armnn as a Dependency
find_package(Armnn REQUIRED)
target_link_libraries(armnnDelegate Armnn::Armnn)

## Add Tensorflow v2.3.1 dependency
find_package(Tensorflow 2.3.1 REQUIRED MODULE)

target_link_libraries(armnnDelegate
        ${Tensorflow_LIB})

target_include_directories(armnnDelegate
        PRIVATE
            ${Tensorflow_INCLUDE_DIR})

## Add TfLite v2.3.1 dependency
find_package(TfLite REQUIRED MODULE)

target_link_libraries(armnnDelegate
        ${TfLite_LIB})

target_include_directories(armnnDelegate
        PRIVATE
            ${TfLite_INCLUDE_DIR})

## Add Flatbuffers dependency
find_package(Flatbuffers REQUIRED MODULE)

target_link_libraries(armnnDelegate
        ${Flatbuffers_LIB})

target_include_directories(armnnDelegate
        PRIVATE
            ${Flatbuffers_INCLUDE_DIR})

set(armnnDelegate_unittest_sources)
list(APPEND armnnDelegate_unittest_sources
        src/test/ArmnnDelegateTest.cpp
        src/test/ElementwiseBinaryTest.cpp
        src/test/ElementwiseBinaryTestHelper.hpp
        src/test/ElementwiseUnaryTest.cpp
        src/test/ElementwiseUnaryTestHelper.hpp)

add_executable(DelegateUnitTests ${armnnDelegate_unittest_sources})
target_include_directories(DelegateUnitTests PRIVATE src)
target_include_directories(DelegateUnitTests PRIVATE third-party)

target_link_libraries(DelegateUnitTests armnnDelegate)

target_include_directories(DelegateUnitTests
        PRIVATE
            ${TfLite_INCLUDE_DIR})

target_include_directories(DelegateUnitTests
        PRIVATE
            ${Flatbuffers_INCLUDE_DIR})

####################################################
## Export targets
set(armnn_delegate_export_targets)
list(APPEND armnn_delegate_export_targets
        armnnDelegate)

install(
        TARGETS ${armnn_delegate_export_targets}
        EXPORT  armnn-delegate-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

## Set export alias
set_target_properties(armnnDelegate
        PROPERTIES
        EXPORT_NAME ArmnnDelegate)

## Export target scrips
install(
        EXPORT      armnn-delegate-targets
        FILE        ArmnnDelegateTargets.cmake
        NAMESPACE   ArmnnDelegate::
        DESTINATION ${CMAKE_INSTALL_LIBDIR})

## Create ArmnnDelegateConfig.cmake
include(CMakePackageConfigHelpers)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR})
message(STATUS "CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}" )
message(STATUS "CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}" )
configure_package_config_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/ArmnnDelegateConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ArmnnDelegateConfig.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIGDIR})

## Install ArmNN Delegate config file
install(
        FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ArmnnDelegateConfig.cmake
        DESTINATION ${INSTALL_CONFIGDIR})

## Export from build tree
export(
        EXPORT      armnn-delegate-targets
        FILE        ${CMAKE_CURRENT_BINARY_DIR}/ArmnnDelegateTargets.cmake
        NAMESPACE   ArmnnDelegate::)

####################################################