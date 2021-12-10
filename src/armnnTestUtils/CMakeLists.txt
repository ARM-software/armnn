#
# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# armnnTestUtils library provides useful test functions for backend developers.
set(armnnTestUtils_sources)
list(APPEND armnnTestUtils_sources
        ../../include/armnnTestUtils/DataLayoutUtils.hpp
        ../../include/armnnTestUtils/LayerTestResult.hpp
        ../../include/armnnTestUtils/PredicateResult.hpp
        ../../include/armnnTestUtils/TensorCopyUtils.hpp
        ../../include/armnnTestUtils/WorkloadTestUtils.hpp
        TensorHelpers.hpp
        CreateWorkload.hpp
        CommonTestUtils.cpp
        CommonTestUtils.hpp
        DataTypeUtils.hpp
        GraphUtils.cpp
        GraphUtils.hpp
        TensorCopyUtils.cpp
        TestUtils.cpp
        TestUtils.hpp
        UnitTests.cpp
        UnitTests.hpp
        )

add_library_ex(armnnTestUtils SHARED ${armnnTestUtils_sources})

set_target_properties(armnnTestUtils PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

target_include_directories(armnnTestUtils
        PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(armnnTestUtils PRIVATE ../armnn)
target_include_directories(armnnTestUtils PRIVATE ../armnnUtils)
target_include_directories(armnnTestUtils PRIVATE ../backends)
target_include_directories(armnnTestUtils PRIVATE ../profiling)

install(TARGETS armnnTestUtils
        EXPORT  armnn-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_library(Armnn::armnnTestUtils ALIAS armnnTestUtils)