#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required (VERSION 3.7.0)
project(armnnDelegate)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wextra -Werror -Wold-style-cast -Wno-missing-braces -Wconversion -Wsign-conversion")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

set(armnnDelegate_sources)
list(APPEND armnnDelegate_sources
        include/armnn_delegate.hpp
        include/DelegateOptions.hpp
        include/Version.hpp
        src/armnn_delegate.cpp
        src/armnn_external_delegate.cpp
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
        src/LogicalBinary.hpp
        src/Lstm.hpp
        src/MultiLayerFacade.hpp
        src/Normalization.hpp
        src/Pack.hpp
        src/Pad.hpp
        src/Pooling.hpp
        src/Prelu.hpp
        src/Quantization.hpp
        src/Redefine.hpp
        src/Reduce.hpp
        src/Resize.hpp
        src/Round.hpp
        src/Shape.hpp
        src/SharedFunctions.hpp
        src/SharedFunctions.cpp
        src/Slice.hpp
        src/Softmax.hpp
        src/SpaceDepth.hpp
        src/Split.hpp
        src/Unpack.hpp
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
if(NOT ARMNN_SUB_PROJECT)
    find_package(Armnn REQUIRED CONFIG HINTS ${Armnn_DIR})
endif()
target_link_libraries(armnnDelegate PUBLIC Armnn::Armnn)

## Add TfLite dependency
find_package(TfLiteSrc REQUIRED MODULE)
find_package(TfLite REQUIRED MODULE)

target_link_libraries(armnnDelegate PUBLIC ${TfLite_LIB})

#  lpthread and ldl are not required for Android
if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL Android)
    target_link_libraries(armnnDelegate PUBLIC -lpthread)
    target_link_libraries(armnnDelegate PUBLIC -ldl)
endif()

# Various tflite header files are not warning clean
# We can't change compilation flags on header files directly, so we need to add them to an interface library first
add_library(tflite_headers INTERFACE)
target_include_directories(tflite_headers INTERFACE $<BUILD_INTERFACE:${TfLite_INCLUDE_DIR}>
                                                    $<INSTALL_INTERFACE:include/tflite_headers>)

target_compile_options(tflite_headers INTERFACE -Wno-conversion
                                                -Wno-sign-conversion
                                                -Wno-unused-parameter
                                                -Wno-unused-function)

target_link_libraries(armnnDelegate PUBLIC tflite_headers)

## Add Flatbuffers dependency
find_package(Flatbuffers REQUIRED MODULE)

target_link_libraries(armnnDelegate PRIVATE
        ${Flatbuffers_LIB})

# include/flatbuffers/flatbuffers.h is not warning clean
# We can't change compilation flags on header files directly, so we need to add them to an interface library first
add_library(flatbuffer_headers INTERFACE)
target_include_directories(flatbuffer_headers INTERFACE $<BUILD_INTERFACE:${Flatbuffers_INCLUDE_DIR}>
                                                 $<INSTALL_INTERFACE:include/flatbuffer_headers>)
target_compile_options(flatbuffer_headers INTERFACE -Wno-sign-conversion)

target_link_libraries(armnnDelegate PUBLIC flatbuffer_headers)

# Add libraries from armnn third-party libraries
# Third-party header files are not warning clean
# We can't change compilation flags on header files directly, so we need to add them to an interface library first
add_library(thirdparty_headers INTERFACE)
target_include_directories(thirdparty_headers INTERFACE $<BUILD_INTERFACE:${ARMNN_SOURCE_DIR}/third-party>
                                                        $<INSTALL_INTERFACE:include/thirdparty_headers>)

target_compile_options(thirdparty_headers INTERFACE -Wno-old-style-cast)
target_link_libraries(armnnDelegate PUBLIC thirdparty_headers)

add_library(profiling_library_headers INTERFACE)
target_include_directories(profiling_library_headers INTERFACE $<BUILD_INTERFACE:${ARMNN_SOURCE_DIR}/profiling>
                                                              $<INSTALL_INTERFACE:include/profiling_library_headers>)
target_link_libraries(armnnDelegate PUBLIC profiling_library_headers)
target_link_libraries(armnnDelegate PUBLIC Armnn::armnnUtils)

set_target_properties(armnnDelegate PROPERTIES VERSION ${DELEGATE_LIB_VERSION} SOVERSION ${DELEGATE_LIB_SOVERSION})

option(BUILD_UNIT_TESTS "Build unit tests" ON)
if(BUILD_UNIT_TESTS)
    set(armnnDelegate_unittest_sources)
    list(APPEND armnnDelegate_unittest_sources
        src/test/ActivationTest.cpp
        src/test/ActivationTestHelper.hpp
        src/test/ArgMinMaxTest.cpp
        src/test/ArgMinMaxTestHelper.hpp
        src/test/ArmnnDelegateTest.cpp
        src/test/BatchSpaceTest.cpp
        src/test/BatchSpaceTestHelper.hpp
        src/test/CastTest.cpp
        src/test/CastTestHelper.hpp
        src/test/ComparisonTest.cpp
        src/test/ComparisonTestHelper.hpp
        src/test/ControlTest.cpp
        src/test/ControlTestHelper.hpp
        src/test/Convolution2dTest.cpp
        src/test/Convolution3dTest.cpp
        src/test/ConvolutionTestHelper.hpp
        src/test/DelegateOptionsTest.cpp
        src/test/DelegateOptionsTestHelper.hpp
        src/test/DepthwiseConvolution2dTest.cpp
        src/test/ElementwiseBinaryTest.cpp
        src/test/ElementwiseBinaryTestHelper.hpp
        src/test/ElementwiseUnaryTest.cpp
        src/test/ElementwiseUnaryTestHelper.hpp
        src/test/FillTest.cpp
        src/test/FillTestHelper.hpp
        src/test/FullyConnectedTest.cpp
        src/test/FullyConnectedTestHelper.hpp
        src/test/GatherTest.cpp
        src/test/GatherTestHelper.hpp
        src/test/LogicalTest.cpp
        src/test/LogicalTestHelper.hpp
        src/test/LstmTest.cpp
        src/test/LstmTestHelper.hpp
        src/test/MirrorPadTest.cpp
        src/test/NormalizationTest.cpp
        src/test/NormalizationTestHelper.hpp
        src/test/PackTest.cpp
        src/test/PackTestHelper.hpp
        src/test/PadTest.cpp
        src/test/PadTestHelper.hpp
        src/test/Pooling2dTest.cpp
        src/test/Pooling2dTestHelper.hpp
        src/test/PreluTest.cpp
        src/test/PreluTestHelper.hpp
        src/test/QuantizationTest.cpp
        src/test/QuantizationTestHelper.hpp
        src/test/RedefineTestHelper.hpp
        src/test/ReduceTest.cpp
        src/test/ReduceTestHelper.hpp
        src/test/ReshapeTest.cpp
        src/test/ResizeTest.cpp
        src/test/ResizeTestHelper.hpp
        src/test/RoundTest.cpp
        src/test/RoundTestHelper.hpp
        src/test/SoftmaxTest.cpp
        src/test/SoftmaxTestHelper.hpp
        src/test/SpaceDepthTest.cpp
        src/test/SpaceDepthTestHelper.hpp
        src/test/ShapeTest.cpp
        src/test/ShapeTestHelper.hpp
        src/test/SliceTest.cpp
        src/test/SliceTestHelper.hpp
        src/test/SplitTest.cpp
        src/test/SplitTestHelper.hpp
        src/test/TestUtils.hpp
        src/test/TestUtils.cpp
        src/test/TransposeTest.cpp
        src/test/TransposeTestHelper.hpp
        src/test/UnidirectionalSequenceLstmTest.cpp
        src/test/UnidirectionalSequenceLstmTestHelper.hpp
        src/test/UnpackTest.cpp
        src/test/UnpackTestHelper.hpp)

        # There's a known Android NDK bug which causes a subset of NeonLayerTests to
        # fail. We'll exclude these tests in NeonLayerTests_NDK_Bug.cpp if we're doing
        # a debug build and NDK is less than r21.
        # https://github.com/android/ndk/issues/1135

        # Default to always including these tests.
        set(INCLUDE_NDK_BUG_TESTS "ON")
        # Reconsider if we in a debug build.
        string( TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE_LOWERCASE )
        if ( NOT BUILD_TYPE_LOWERCASE STREQUAL "release" )
            message("CMAKE:: BUILD TYPE IS ${CMAKE_BUILD_TYPE}")
            # And NDK_VERSION has been set.
            if ( DEFINED NDK_VERSION )
                message("CMAKE:: NDK DEFINED")
                # And the version is less than r21.
                if ( ${NDK_VERSION} STRLESS "r21" )
                    message("CMAKE:: BUG TESTS OFF")
                    set(INCLUDE_NDK_BUG_TESTS "OFF")
                endif()
            endif()
        endif()

        if ( INCLUDE_NDK_BUG_TESTS STREQUAL "ON" )
           list(APPEND armnnDelegate_unittest_sources
                src/test/NeonDelegateTests_NDK_Issue.cpp
                )
        else()

        endif()

    add_executable(DelegateUnitTests ${armnnDelegate_unittest_sources})

    # Add half library from armnn third-party libraries
    target_link_libraries(DelegateUnitTests PRIVATE thirdparty_headers)

    target_link_libraries(DelegateUnitTests PRIVATE armnnDelegate)
    target_link_libraries(DelegateUnitTests PRIVATE Armnn::armnnUtils)

    target_link_libraries(DelegateUnitTests PRIVATE tflite_headers)
    target_link_libraries(DelegateUnitTests PRIVATE flatbuffer_headers)
    target_link_libraries(DelegateUnitTests PRIVATE profiling_library_headers)

endif()

####################################################
## Export targets
set(armnn_delegate_export_targets)
list(APPEND armnn_delegate_export_targets
            armnnDelegate
            tflite_headers
            flatbuffer_headers
            profiling_library_headers
            thirdparty_headers)

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
SET(Armnn_DIR "${Armnn_DIR}")

configure_package_config_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/ArmnnDelegateConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ArmnnDelegateConfig.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
        PATH_VARS  Armnn_DIR)

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
add_library(ArmnnDelegate::ArmnnDelegate ALIAS armnnDelegate)


####################################################
