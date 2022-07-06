#
# Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# armnnTestUtils library provides useful test functions for backend developers.
set(armnnTestUtils_sources)
list(APPEND armnnTestUtils_sources
        ../../include/armnnTestUtils/DataLayoutUtils.hpp
        ../../include/armnnTestUtils/LayerTestResult.hpp
        ../../include/armnnTestUtils/MockBackend.hpp
        ../../include/armnnTestUtils/MockMemoryManager.hpp
        ../../include/armnnTestUtils/MockTensorHandle.hpp
        ../../include/armnnTestUtils/PredicateResult.hpp
        ../../include/armnnTestUtils/TensorCopyUtils.hpp
        ../../include/armnnTestUtils/WorkloadTestUtils.hpp
        ../../include/armnnTestUtils/TensorHelpers.hpp
        ../../include/armnnTestUtils/Version.hpp
        CreateWorkload.hpp
        CommonTestUtils.cpp
        CommonTestUtils.hpp
        DataTypeUtils.hpp
        GraphUtils.cpp
        GraphUtils.hpp
        MockBackend.cpp
        MockMemoryManager.cpp
        MockTensorHandle.cpp
        MockTensorHandleFactory.cpp
        TensorCopyUtils.cpp
        TestUtils.cpp
        TestUtils.hpp
        )

if(NOT BUILD_BARE_METAL AND NOT EXECUTE_NETWORK_STATIC)
list(APPEND armnnTestUtils_sources
        UnitTests.cpp
        UnitTests.hpp
        )
endif()


if(BUILD_BARE_METAL)
    add_library_ex(armnnTestUtils STATIC ${armnnTestUtils_sources})
elseif(EXECUTE_NETWORK_STATIC)
    add_library_ex(armnnTestUtils OBJECT ${armnnTestUtils_sources})
else()
    add_library_ex(armnnTestUtils SHARED ${armnnTestUtils_sources})
endif()

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

target_link_libraries(armnnTestUtils armnn)

install(TARGETS armnnTestUtils
        EXPORT  armnn-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

add_library(Armnn::armnnTestUtils ALIAS armnnTestUtils)

set_target_properties(armnnTestUtils PROPERTIES VERSION ${ARMNN_TEST_UTILS_LIB_VERSION} SOVERSION ${ARMNN_TEST_UTILS_LIB_SOVERSION})
