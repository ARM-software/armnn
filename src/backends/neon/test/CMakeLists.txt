#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnNeonBackendUnitTests_sources
    NeonBackendTests.cpp
    NeonCreateWorkloadTests.cpp
    NeonEndToEndTests.cpp
    NeonJsonPrinterTests.cpp
    NeonLayerSupportTests.cpp
    NeonLayerTests.cpp
    NeonOptimizedNetworkTests.cpp
    NeonRuntimeTests.cpp
    NeonTensorHandleTests.cpp
    NeonTimerTest.cpp
    NeonWorkloadFactoryHelper.hpp
)

# There's a known Android NDK bug which causes a subset of NeonLayerTests to
# fail. We'll exclude these tests in NeonLayerTests_NDK_Bug.cpp if we're doing
# a debug build and NDK is less than r21.
# https://github.com/android/ndk/issues/1135

# Default to always including these tests.
set(INCLUDE_NDK_BUG_TESTS "ON")
# Reconsider if we in a debug build.
string( TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE_LOWERCASE )
if ( NOT BUILD_TYPE_LOWERCASE STREQUAL "release" )
    # And NDK_VERSION has been set.
    if ( DEFINED NDK_VERSION )
        # And the version is less than r21.
        if ( ${NDK_VERSION} STRLESS "r21" )
            set(INCLUDE_NDK_BUG_TESTS "OFF")
        endif()
    endif()
endif()

if ( INCLUDE_NDK_BUG_TESTS STREQUAL "ON" )
   list(APPEND armnnNeonBackendUnitTests_sources
        NeonLayerTests_NDK_Bug.cpp
        )
else()

endif()

if (ARMNNREF)
    list(APPEND armnnNeonBackendUnitTests_sources
        NeonFallbackTests.cpp
        NeonMemCopyTests.cpp
        )
endif()

add_library(armnnNeonBackendUnitTests OBJECT ${armnnNeonBackendUnitTests_sources})
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnTestUtils)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/client/include)
