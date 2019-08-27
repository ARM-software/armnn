#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnBackendsCommonUnitTests_sources
    ActivationFixture.hpp
    ActivationTestImpl.hpp
    BackendIdTests.cpp
    BackendRegistryTests.cpp
    BatchNormTestImpl.hpp
    CommonTestUtils.cpp
    CommonTestUtils.hpp
    Conv2dTestImpl.hpp
    ConvertFp16ToFp32TestImpl.hpp
    ConvertFp32ToFp16TestImpl.hpp
    DebugTestImpl.hpp
    DequantizeTestImpl.hpp
    DequantizeEndToEndTestImpl.hpp
    DetectionPostProcessLayerTestImpl.hpp
    DetectionPostProcessTestImpl.hpp
    DynamicBackendTests.cpp
    DynamicBackendTests.hpp
    EndToEndTestImpl.hpp
    FullyConnectedTestImpl.hpp
    GatherTestImpl.hpp
    GatherEndToEndTestImpl.hpp
    IsLayerSupportedTestImpl.hpp
    JsonPrinterTestImpl.cpp
    JsonPrinterTestImpl.hpp
    LayerReleaseConstantDataTest.cpp
    LayerTests.cpp
    LayerTests.hpp
    LstmTestImpl.hpp
    NormTestImpl.hpp
    ConcatTestImpl.hpp
    MockBackend.cpp
    MockBackend.hpp
    MockBackendId.hpp
    OptimizeSubgraphViewTests.cpp
    OptimizationViewsTests.cpp
    PermuteTestImpl.hpp
    Pooling2dTestImpl.hpp
    PreluEndToEndTestImpl.hpp
    QuantizeHelper.hpp
    QuantizeTestImpl.hpp
    QuantizedLstmEndToEndTestImpl.hpp
    ResizeEndToEndTestImpl.hpp
    RuntimeTestImpl.hpp
    SoftmaxTestImpl.hpp
    SpaceToDepthTestImpl.hpp
    SpaceToDepthEndToEndTestImpl.hpp
    SplitterEndToEndTestImpl.hpp
    SplitterTestImpl.hpp
    StridedSliceTestImpl.hpp
    TensorCopyUtils.cpp
    TensorCopyUtils.hpp
    WorkloadFactoryHelper.hpp
    WorkloadTestUtils.hpp
)

if (ARMNNREF)
    list(APPEND armnnBackendsCommonUnitTests_sources
        OptimizedNetworkTests.cpp
        WorkloadDataValidation.cpp
        )
endif()

add_library(armnnBackendsCommonUnitTests OBJECT ${armnnBackendsCommonUnitTests_sources})
target_include_directories(armnnBackendsCommonUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnBackendsCommonUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnBackendsCommonUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

# Dummy shared object for testing.
# This is a simple library used to test the utility functions that will be used to handle the shared objects.

list(APPEND testSharedObject_sources
    TestSharedObject.cpp
    TestSharedObject.hpp
)

add_library_ex(TestSharedObject MODULE ${testSharedObject_sources})
set_target_properties(TestSharedObject PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testSharedObject)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/testSharedObject/libNoSharedObject.txt "This is not a shared object")

# Dummy dynamic backends for testing.
# Both a valid and a number of invalid dummy dynamic backends (covering various use cases) share the
# same source code. The various test cases are put together in the code using compiler directives.

list(APPEND testDynamicBackend_sources
    TestDynamicBackend.cpp
    TestDynamicBackend.hpp
)

add_library_ex(ValidTestDynamicBackend MODULE ${testDynamicBackend_sources})
target_compile_definitions(ValidTestDynamicBackend PRIVATE -DVALID_TEST_DYNAMIC_BACKEND_1)
target_include_directories(ValidTestDynamicBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(ValidTestDynamicBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(ValidTestDynamicBackend armnn)
set_target_properties(ValidTestDynamicBackend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testDynamicBackend)

add_library_ex(InvalidTestDynamicBackend1 MODULE ${testDynamicBackend_sources})
target_compile_definitions(InvalidTestDynamicBackend1 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_1)
target_include_directories(InvalidTestDynamicBackend1 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(InvalidTestDynamicBackend1 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(InvalidTestDynamicBackend1 armnn)
set_target_properties(InvalidTestDynamicBackend1 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testDynamicBackend)

add_library_ex(InvalidTestDynamicBackend2 MODULE ${testDynamicBackend_sources})
target_compile_definitions(InvalidTestDynamicBackend2 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_2)
target_include_directories(InvalidTestDynamicBackend2 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(InvalidTestDynamicBackend2 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(InvalidTestDynamicBackend2 armnn)
set_target_properties(InvalidTestDynamicBackend2 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testDynamicBackend)

add_library_ex(InvalidTestDynamicBackend3 MODULE ${testDynamicBackend_sources})
target_compile_definitions(InvalidTestDynamicBackend3 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_3)
target_include_directories(InvalidTestDynamicBackend3 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(InvalidTestDynamicBackend3 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(InvalidTestDynamicBackend3 armnn)
set_target_properties(InvalidTestDynamicBackend3 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testDynamicBackend)

add_library_ex(InvalidTestDynamicBackend4 MODULE ${testDynamicBackend_sources})
target_compile_definitions(InvalidTestDynamicBackend4 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_4)
target_include_directories(InvalidTestDynamicBackend4 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(InvalidTestDynamicBackend4 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(InvalidTestDynamicBackend4 armnn)
set_target_properties(InvalidTestDynamicBackend4 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testDynamicBackend)

add_library_ex(InvalidTestDynamicBackend5 MODULE ${testDynamicBackend_sources})
target_compile_definitions(InvalidTestDynamicBackend5 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_5)
target_include_directories(InvalidTestDynamicBackend5 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(InvalidTestDynamicBackend5 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(InvalidTestDynamicBackend5 armnn)
set_target_properties(InvalidTestDynamicBackend5 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testDynamicBackend)

add_library_ex(InvalidTestDynamicBackend6 MODULE ${testDynamicBackend_sources})
target_compile_definitions(InvalidTestDynamicBackend6 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_6)
target_include_directories(InvalidTestDynamicBackend6 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(InvalidTestDynamicBackend6 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(InvalidTestDynamicBackend6 armnn)
set_target_properties(InvalidTestDynamicBackend6 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testDynamicBackend)

add_library_ex(InvalidTestDynamicBackend7 MODULE ${testDynamicBackend_sources})
target_compile_definitions(InvalidTestDynamicBackend7 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_7)
target_include_directories(InvalidTestDynamicBackend7 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(InvalidTestDynamicBackend7 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(InvalidTestDynamicBackend7 armnn)
set_target_properties(InvalidTestDynamicBackend7 PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testDynamicBackend)

# Dummy files and directories for testing the dynamic backend file parsing.

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend.so          "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend.so.1        "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend.so.1.2      "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend.so.1.2.3    "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend.so.10.1.27  "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend.so.10.1.33. "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend.so.3.4..5   "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend.so.1,1.1    "Test file for dynamic backend file parsing")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm123_GpuAcc_backend.so       "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc456_backend.so       "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm%Co_GpuAcc_backend.so       "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_Gpu.Acc_backend.so         "Test file for dynamic backend file parsing")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/GpuAcc_backend.so              "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/_GpuAcc_backend.so             "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm__backend.so                "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc.so                  "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/__backend.so                   "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/__.so                          "Test file for dynamic backend file parsing")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend             "Test file for dynamic backend file parsing")
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath1/Arm_GpuAcc_backend_v1.2.so     "Test file for dynamic backend file parsing")

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath2/Arm_CpuAcc_backend.so          "Test file for dynamic backend file parsing")
add_custom_target(link_dummy_dynamic_backend ALL
                  COMMAND ${CMAKE_COMMAND} -E create_symlink Arm_CpuAcc_backend.so backendsTestPath2/Arm_CpuAcc_backend.so.1
                  COMMAND ${CMAKE_COMMAND} -E create_symlink Arm_CpuAcc_backend.so.1 backendsTestPath2/Arm_CpuAcc_backend.so.1.2
                  COMMAND ${CMAKE_COMMAND} -E create_symlink Arm_CpuAcc_backend.so.1.2 backendsTestPath2/Arm_CpuAcc_backend.so.1.2.3
                  COMMAND ${CMAKE_COMMAND} -E create_symlink nothing backendsTestPath2/Arm_no_backend.so)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath2/Arm_GpuAcc_backend.so          "Test file for dynamic backend file parsing")

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath3)

# Dummy dynamic backends for testing.
# These shared objects are used to test the creation of dynamic backend instances in ArmNN.

add_library_ex(Arm_TestValid2_backend MODULE ${testDynamicBackend_sources})
target_compile_definitions(Arm_TestValid2_backend PRIVATE -DVALID_TEST_DYNAMIC_BACKEND_2)
target_include_directories(Arm_TestValid2_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_TestValid2_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(Arm_TestValid2_backend armnn)
set_target_properties(Arm_TestValid2_backend PROPERTIES PREFIX "")
set_target_properties(Arm_TestValid2_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath5)
add_custom_command(TARGET Arm_TestValid2_backend POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Arm_TestValid2_backend> ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath6)

add_library_ex(Arm_TestValid3_backend MODULE ${testDynamicBackend_sources})
target_compile_definitions(Arm_TestValid3_backend PRIVATE -DVALID_TEST_DYNAMIC_BACKEND_3)
target_include_directories(Arm_TestValid3_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_TestValid3_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(Arm_TestValid3_backend armnn)
set_target_properties(Arm_TestValid3_backend PROPERTIES PREFIX "")
set_target_properties(Arm_TestValid3_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath5)

add_library_ex(Arm_TestInvalid8_backend MODULE ${testDynamicBackend_sources})
target_compile_definitions(Arm_TestInvalid8_backend PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_8)
target_include_directories(Arm_TestInvalid8_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_TestInvalid8_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(Arm_TestInvalid8_backend armnn)
set_target_properties(Arm_TestInvalid8_backend PROPERTIES PREFIX "")
set_target_properties(Arm_TestInvalid8_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath5)

add_library_ex(Arm_TestValid4_backend MODULE ${testDynamicBackend_sources})
target_compile_definitions(Arm_TestValid4_backend PRIVATE -DVALID_TEST_DYNAMIC_BACKEND_4)
target_include_directories(Arm_TestValid4_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_TestValid4_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(Arm_TestValid4_backend armnn)
set_target_properties(Arm_TestValid4_backend PROPERTIES PREFIX "")
set_target_properties(Arm_TestValid4_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath6)

add_library_ex(Arm_TestValid5_backend MODULE ${testDynamicBackend_sources})
target_compile_definitions(Arm_TestValid5_backend PRIVATE -DVALID_TEST_DYNAMIC_BACKEND_5)
target_include_directories(Arm_TestValid5_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_TestValid5_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(Arm_TestValid5_backend armnn)
set_target_properties(Arm_TestValid5_backend PROPERTIES PREFIX "")
set_target_properties(Arm_TestValid5_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath6)

add_library_ex(Arm_TestInvalid9_backend MODULE ${testDynamicBackend_sources})
target_compile_definitions(Arm_TestInvalid9_backend PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_9)
target_include_directories(Arm_TestInvalid9_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_TestInvalid9_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(Arm_TestInvalid9_backend armnn)
set_target_properties(Arm_TestInvalid9_backend PROPERTIES PREFIX "")
set_target_properties(Arm_TestInvalid9_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath6)

add_library_ex(Arm_TestInvalid10_backend MODULE ${testDynamicBackend_sources})
target_compile_definitions(Arm_TestInvalid10_backend PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_10)
target_include_directories(Arm_TestInvalid10_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_TestInvalid10_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(Arm_TestInvalid10_backend armnn)
set_target_properties(Arm_TestInvalid10_backend PROPERTIES PREFIX "")
set_target_properties(Arm_TestInvalid10_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath9)

add_library_ex(Arm_TestInvalid11_backend MODULE ${testDynamicBackend_sources})
target_compile_definitions(Arm_TestInvalid11_backend PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_11)
target_include_directories(Arm_TestInvalid11_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(Arm_TestInvalid11_backend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_link_libraries(Arm_TestInvalid11_backend armnn)
set_target_properties(Arm_TestInvalid11_backend PROPERTIES PREFIX "")
set_target_properties(Arm_TestInvalid11_backend PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath9)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/backendsTestPath7)
