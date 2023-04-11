#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

cmake_minimum_required (VERSION 3.7.0)
project(armnnDelegate)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Wextra -Werror -Wold-style-cast -Wno-missing-braces -Wconversion -Wsign-conversion -Wno-comment")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_CLASSIC_DELEGATE "Build classic delegate" ON)
option(BUILD_OPAQUE_DELEGATE "Build opaque delegate" OFF)
option(BUILD_SHARED_LIBS "Build share libs" ON)
option(BUILD_DELEGATE_JNI_INTERFACE "Builds a library to allow accessing the Arm NN delegate from Java code.
                                     This is an experimental feature." ON)

set(armnnDelegate_sources)
list(APPEND armnnDelegate_sources
        common/include/DelegateOptions.hpp
        common/src/DelegateOptions.cpp
        common/src/DelegateUtils.hpp)

## Add Armnn as a Dependency
if(NOT ARMNN_SUB_PROJECT)
    find_package(Armnn REQUIRED CONFIG HINTS ${Armnn_DIR})
endif()

if (BUILD_CLASSIC_DELEGATE)
    add_subdirectory(classic)
    add_library(armnnDelegate ${armnnDelegate_sources})

    target_include_directories(armnnDelegate
            PUBLIC
                $<INSTALL_INTERFACE:include>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/common/src)
endif()
if (BUILD_OPAQUE_DELEGATE)
    add_subdirectory(opaque)
    add_library(armnnOpaqueDelegate ${armnnDelegate_sources})

    target_include_directories(armnnOpaqueDelegate
            PUBLIC
                $<INSTALL_INTERFACE:include>
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common/include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/common/src)
endif()

include(GNUInstallDirs)

if (BUILD_CLASSIC_DELEGATE)
    target_link_libraries(armnnDelegate PUBLIC Armnn::Armnn)

    ## Add armnnClassicDelegateObject as a Dependency
    target_link_libraries(armnnDelegate PUBLIC armnnClassicDelegateObject)
endif()
if (BUILD_OPAQUE_DELEGATE)
    target_link_libraries(armnnOpaqueDelegate PUBLIC Armnn::Armnn)

    ## Add armnnOpaqueDelegateObject as a Dependency
    target_link_libraries(armnnOpaqueDelegate PUBLIC armnnOpaqueDelegateObject)
endif()

## Add TfLite dependency
find_package(TfLiteSrc REQUIRED MODULE)
find_package(TfLite REQUIRED MODULE)
if (BUILD_CLASSIC_DELEGATE)
    target_link_libraries(armnnDelegate PUBLIC ${TfLite_LIB})

    #  lpthread and ldl are not required for Android
    if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL Android)
        target_link_libraries(armnnDelegate PUBLIC -lpthread)
        target_link_libraries(armnnDelegate PUBLIC -ldl)
    endif()
endif()
if (BUILD_OPAQUE_DELEGATE)
    target_link_libraries(armnnOpaqueDelegate PUBLIC ${TfLite_LIB})

    #  lpthread and ldl are not required for Android
    if(NOT "${CMAKE_SYSTEM_NAME}" STREQUAL Android)
        target_link_libraries(armnnOpaqueDelegate PUBLIC -lpthread)
        target_link_libraries(armnnOpaqueDelegate PUBLIC -ldl)
    endif()
endif()

# Add libraries from armnn third-party libraries
# Third-party header files are not warning clean
# We can't change compilation flags on header files directly, so we need to add them to an interface library first
add_library(thirdparty_headers INTERFACE)
target_include_directories(thirdparty_headers INTERFACE $<BUILD_INTERFACE:${ARMNN_SOURCE_DIR}/third-party>
                                                        $<INSTALL_INTERFACE:include/thirdparty_headers>)

target_compile_options(thirdparty_headers INTERFACE -Wno-old-style-cast)
if (BUILD_CLASSIC_DELEGATE)
    target_link_libraries(armnnDelegate PUBLIC thirdparty_headers)
endif()
if (BUILD_OPAQUE_DELEGATE)
    target_link_libraries(armnnOpaqueDelegate PUBLIC thirdparty_headers)
endif()

add_library(profiling_library_headers INTERFACE)
target_include_directories(profiling_library_headers INTERFACE $<BUILD_INTERFACE:${ARMNN_SOURCE_DIR}/profiling>
                                                              $<INSTALL_INTERFACE:include/profiling_library_headers>)

if (BUILD_CLASSIC_DELEGATE)
    target_link_libraries(armnnDelegate PUBLIC profiling_library_headers)
    target_link_libraries(armnnDelegate PUBLIC Armnn::armnnUtils)

    set_target_properties(armnnDelegate PROPERTIES VERSION ${DELEGATE_LIB_VERSION} SOVERSION ${DELEGATE_LIB_SOVERSION})
endif()
if (BUILD_OPAQUE_DELEGATE)
    target_link_libraries(armnnOpaqueDelegate PUBLIC profiling_library_headers)
    target_link_libraries(armnnOpaqueDelegate PUBLIC Armnn::armnnUtils)

    set_target_properties(armnnOpaqueDelegate PROPERTIES VERSION ${OPAQUE_DELEGATE_LIB_VERSION} SOVERSION ${OPAQUE_DELEGATE_LIB_SOVERSION})
endif()

if(BUILD_UNIT_TESTS)
    set(commonDelegate_unittest_sources)
    list(APPEND commonDelegate_unittest_sources
        common/src/test/DelegateTestInterpreter.hpp
        common/src/test/DelegateTestInterpreterUtils.hpp
        test/ActivationTest.cpp
        test/ActivationTestHelper.hpp
        test/ArgMinMaxTest.cpp
        test/ArgMinMaxTestHelper.hpp
        test/BatchMatMulTest.cpp
        test/BatchMatMulTestHelper.hpp
        test/BatchSpaceTest.cpp
        test/BatchSpaceTestHelper.hpp
        test/CastTest.cpp
        test/CastTestHelper.hpp
        test/ComparisonTest.cpp
        test/ComparisonTestHelper.hpp
        test/ControlTest.cpp
        test/ControlTestHelper.hpp
        test/Convolution2dTest.cpp
        test/Convolution3dTest.cpp
        test/ConvolutionTestHelper.hpp
        test/DelegateOptionsTest.cpp
        test/DelegateOptionsTestHelper.hpp
        test/DepthwiseConvolution2dTest.cpp
        test/ElementwiseBinaryTest.cpp
        test/ElementwiseBinaryTestHelper.hpp
        test/ElementwiseUnaryTest.cpp
        test/ElementwiseUnaryTestHelper.hpp
        test/FillTest.cpp
        test/FillTestHelper.hpp
        test/FullyConnectedTest.cpp
        test/FullyConnectedTestHelper.hpp
        test/GatherTest.cpp
        test/GatherTestHelper.hpp
        test/GatherNdTest.cpp
        test/GatherNdTestHelper.hpp
        test/LogicalTest.cpp
        test/LogicalTestHelper.hpp
        test/LstmTest.cpp
        test/LstmTestHelper.hpp
        test/MirrorPadTest.cpp
        test/NormalizationTest.cpp
        test/NormalizationTestHelper.hpp
        test/PackTest.cpp
        test/PackTestHelper.hpp
        test/PadTest.cpp
        test/PadTestHelper.hpp
        test/Pooling2dTest.cpp
        test/Pooling2dTestHelper.hpp
        test/Pooling3dTest.cpp
        test/Pooling3dTestHelper.hpp
        test/PreluTest.cpp
        test/PreluTestHelper.hpp
        test/QuantizationTest.cpp
        test/QuantizationTestHelper.hpp
        test/RedefineTestHelper.hpp
        test/ReduceTest.cpp
        test/ReduceTestHelper.hpp
        test/ReshapeTest.cpp
        test/ResizeTest.cpp
        test/ResizeTestHelper.hpp
        test/RoundTest.cpp
        test/RoundTestHelper.hpp
        test/SoftmaxTest.cpp
        test/SoftmaxTestHelper.hpp
        test/SpaceDepthTest.cpp
        test/SpaceDepthTestHelper.hpp
        test/ShapeTest.cpp
        test/ShapeTestHelper.hpp
        test/SliceTest.cpp
        test/SliceTestHelper.hpp
        test/StridedSliceTest.cpp
        test/StridedSliceTestHelper.hpp
        test/SplitTest.cpp
        test/SplitTestHelper.hpp
        test/TestUtils.hpp
        test/TestUtils.cpp
        test/TransposeTest.cpp
        test/TransposeTestHelper.hpp
        test/UnidirectionalSequenceLstmTest.cpp
        test/UnidirectionalSequenceLstmTestHelper.hpp
        test/UnpackTest.cpp
        test/UnpackTestHelper.hpp)

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
        list(APPEND commonDelegate_unittest_sources
             test/NeonDelegateTests_NDK_Issue.cpp)
    endif()

    if (BUILD_CLASSIC_DELEGATE)
        set(classicDelegate_unittest_sources)
        list(APPEND classicDelegate_unittest_sources
             classic/src/test/ArmnnClassicDelegateTest.cpp
             classic/src/test/DelegateTestInterpreter.cpp)

        add_executable(DelegateUnitTests ${commonDelegate_unittest_sources} ${classicDelegate_unittest_sources})

        target_include_directories(DelegateUnitTests SYSTEM PRIVATE "${TF_LITE_SCHEMA_INCLUDE_PATH}")
        target_include_directories(DelegateUnitTests SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common/src/test")
        target_include_directories(DelegateUnitTests PUBLIC ${PROJECT_SOURCE_DIR})

        # Add half library from armnn third-party libraries
        target_link_libraries(DelegateUnitTests PRIVATE thirdparty_headers)
        target_link_libraries(DelegateUnitTests PRIVATE armnnDelegate)
        target_link_libraries(DelegateUnitTests PRIVATE Armnn::armnnUtils)
        target_link_libraries(DelegateUnitTests PRIVATE profiling_library_headers)
    endif()

    if (BUILD_OPAQUE_DELEGATE)
        set(opaqueDelegate_unittest_sources)
        list(APPEND opaqueDelegate_unittest_sources
             common/src/test/DelegateTestInterpreter.hpp
             common/src/test/DelegateTestInterpreterUtils.hpp
             opaque/src/test/ArmnnOpaqueDelegateTest.cpp
             opaque/src/test/DelegateTestInterpreter.cpp
             test/TestUtils.hpp
             test/TestUtils.cpp
             test/CastTest.cpp
             test/CastTestHelper.hpp)

        # Until all operators are supported, we have to add tests one by one above to opaqueDelegate_unittest_sources.
        # After we add can add commonDelegate_unittest_sources to the add_executable below.
        add_executable(OpaqueDelegateUnitTests ${opaqueDelegate_unittest_sources})

        target_include_directories(OpaqueDelegateUnitTests SYSTEM PRIVATE "${TF_LITE_SCHEMA_INCLUDE_PATH}")
        target_include_directories(OpaqueDelegateUnitTests SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/common/src/test")
        target_include_directories(OpaqueDelegateUnitTests PUBLIC ${PROJECT_SOURCE_DIR})

        # Add half library from armnn third-party libraries
        target_link_libraries(OpaqueDelegateUnitTests PRIVATE thirdparty_headers)
        target_link_libraries(OpaqueDelegateUnitTests PRIVATE armnnOpaqueDelegate)
        target_link_libraries(OpaqueDelegateUnitTests PRIVATE Armnn::armnnUtils)
        target_link_libraries(OpaqueDelegateUnitTests PRIVATE profiling_library_headers)
    endif()
endif()

if(BUILD_DELEGATE_JNI_INTERFACE)
    add_subdirectory(armnnDelegateJNI)
endif()

####################################################
## Export targets
set(armnn_delegate_export_targets)
list(APPEND armnn_delegate_export_targets
            armnnClassicDelegateObject
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
