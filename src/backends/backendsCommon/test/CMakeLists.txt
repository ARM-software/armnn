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
    OptimizedNetworkTests.cpp
    OptimizeSubgraphViewTests.cpp
    OptimizationViewsTests.cpp
    PermuteTestImpl.hpp
    Pooling2dTestImpl.hpp
    PreluEndToEndTestImpl.hpp
    QuantizeHelper.hpp
    QuantizeTestImpl.hpp
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
    WorkloadDataValidation.cpp
    WorkloadFactoryHelper.hpp
    WorkloadTestUtils.hpp
)

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
target_compile_definitions(ValidTestDynamicBackend PRIVATE -DVALID_TEST_DYNAMIC_BACKEND)
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
