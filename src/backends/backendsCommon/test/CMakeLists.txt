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
    CommonTestUtils.hpp
    Conv2dTestImpl.hpp
    ConvertFp16ToFp32TestImpl.hpp
    ConvertFp32ToFp16TestImpl.hpp
    DebugTestImpl.hpp
    DequantizeTestImpl.hpp
    DetectionPostProcessLayerTestImpl.hpp
    DetectionPostProcessTestImpl.hpp
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
    MergerTestImpl.hpp
    MockBackend.cpp
    MockBackend.hpp
    MockBackendId.hpp
    OptimizedNetworkTests.cpp
    PermuteTestImpl.hpp
    Pooling2dTestImpl.hpp
    QuantizeHelper.hpp
    QuantizeTestImpl.hpp
    ReshapeTestImpl.hpp
    RuntimeTestImpl.hpp
    SoftmaxTestImpl.hpp
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
