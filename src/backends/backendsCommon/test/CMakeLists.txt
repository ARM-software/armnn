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

list(APPEND armnnTestSharedObject_sources
    TestSharedObject.cpp
    TestSharedObject.hpp
)

add_library_ex(armnnTestSharedObject SHARED ${armnnTestSharedObject_sources})
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/libarmnnNoSharedObject.txt "This is not a shared object")

# Dummy dynamic backends for testing.
# Both a valid and a number of invalid dummy dynamic backends (covering various use cases) share the
# same source code. The various test cases are put together in the code using compiler directives.

list(APPEND armnnTestDynamicBackend_sources
    TestDynamicBackend.cpp
    TestDynamicBackend.hpp
)

add_library_ex(armnnValidTestDynamicBackend SHARED ${armnnTestDynamicBackend_sources})
target_compile_definitions(armnnValidTestDynamicBackend PRIVATE -DVALID_TEST_DYNAMIC_BACKEND)
target_include_directories(armnnValidTestDynamicBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnValidTestDynamicBackend PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_library_ex(armnnInvalidTestDynamicBackend1 SHARED ${armnnTestDynamicBackend_sources})
target_compile_definitions(armnnInvalidTestDynamicBackend1 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_1)
target_include_directories(armnnInvalidTestDynamicBackend1 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnInvalidTestDynamicBackend1 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_library_ex(armnnInvalidTestDynamicBackend2 SHARED ${armnnTestDynamicBackend_sources})
target_compile_definitions(armnnInvalidTestDynamicBackend2 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_2)
target_include_directories(armnnInvalidTestDynamicBackend2 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnInvalidTestDynamicBackend2 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_library_ex(armnnInvalidTestDynamicBackend3 SHARED ${armnnTestDynamicBackend_sources})
target_compile_definitions(armnnInvalidTestDynamicBackend3 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_3)
target_include_directories(armnnInvalidTestDynamicBackend3 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnInvalidTestDynamicBackend3 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_library_ex(armnnInvalidTestDynamicBackend4 SHARED ${armnnTestDynamicBackend_sources})
target_compile_definitions(armnnInvalidTestDynamicBackend4 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_4)
target_include_directories(armnnInvalidTestDynamicBackend4 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnInvalidTestDynamicBackend4 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_library_ex(armnnInvalidTestDynamicBackend5 SHARED ${armnnTestDynamicBackend_sources})
target_compile_definitions(armnnInvalidTestDynamicBackend5 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_5)
target_include_directories(armnnInvalidTestDynamicBackend5 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnInvalidTestDynamicBackend5 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_library_ex(armnnInvalidTestDynamicBackend6 SHARED ${armnnTestDynamicBackend_sources})
target_compile_definitions(armnnInvalidTestDynamicBackend6 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_6)
target_include_directories(armnnInvalidTestDynamicBackend6 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnInvalidTestDynamicBackend6 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)

add_library_ex(armnnInvalidTestDynamicBackend7 SHARED ${armnnTestDynamicBackend_sources})
target_compile_definitions(armnnInvalidTestDynamicBackend7 PRIVATE -DINVALID_TEST_DYNAMIC_BACKEND_7)
target_include_directories(armnnInvalidTestDynamicBackend7 PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnInvalidTestDynamicBackend7 PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
