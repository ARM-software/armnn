//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceDepthTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void DepthToSpaceFp32Test(std::vector<armnn::BackendId>& backends, int blockSize)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 2, 2, 4 };
    std::vector<int32_t> outputShape { 1, 4, 4, 1 };

    std::vector<float> inputValues = { 1.f,  2.f,  3.f,  4.f,
                                       5.f,  6.f,  7.f,  8.f,
                                       9.f, 10.f, 11.f, 12.f,
                                       13.f, 14.f, 15.f, 16.f };

    std::vector<float> expectedOutputValues = { 1.f,   2.f,   5.f,   6.f,
                                                3.f,   4.f,   7.f,   8.f,
                                                9.f,  10.f,  13.f,  14.f,
                                                11.f,  12.f,  15.f,  16.f };

    SpaceDepthTest<float>(tflite::BuiltinOperator_DEPTH_TO_SPACE,
                          ::tflite::TensorType_FLOAT32,
                          backends,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          blockSize);
}

void DepthToSpaceUint8Test(std::vector<armnn::BackendId>& backends, int blockSize)
{
    // Set input data
    std::vector<int32_t> inputShape { 2, 1, 1, 4 };
    std::vector<int32_t> outputShape { 2, 2, 2, 1 };

    std::vector<uint8_t> inputValues = { 1,  2,  3,  4,
                                         5,  6,  7,  8 };

    std::vector<uint8_t> expectedOutputValues = { 1,  2,  3,  4,
                                                  5,  6,  7,  8 };

    SpaceDepthTest<uint8_t>(tflite::BuiltinOperator_DEPTH_TO_SPACE,
                            ::tflite::TensorType_UINT8,
                            backends,
                            inputShape,
                            outputShape,
                            inputValues,
                            expectedOutputValues,
                            blockSize);
}

void SpaceToDepthFp32Test(std::vector<armnn::BackendId>& backends, int blockSize)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 2, 2, 2 };
    std::vector<int32_t> outputShape { 1, 1, 1, 8 };

    std::vector<float> inputValues = { 1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f };
    std::vector<float> expectedOutputValues = { 1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f };

    SpaceDepthTest<float>(tflite::BuiltinOperator_SPACE_TO_DEPTH,
                          ::tflite::TensorType_FLOAT32,
                          backends,
                          inputShape,
                          outputShape,
                          inputValues,
                          expectedOutputValues,
                          blockSize);
}

void SpaceToDepthUint8Test(std::vector<armnn::BackendId>& backends, int blockSize)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 1, 1, 4 };

    std::vector<uint8_t> inputValues = { 1, 2, 3, 2 };
    std::vector<uint8_t> expectedOutputValues = { 1, 2, 3, 2 };

    SpaceDepthTest<uint8_t>(tflite::BuiltinOperator_SPACE_TO_DEPTH,
                            ::tflite::TensorType_UINT8,
                            backends,
                            inputShape,
                            outputShape,
                            inputValues,
                            expectedOutputValues,
                            blockSize);
}

TEST_SUITE("DepthToSpace_CpuRefTests")
{

TEST_CASE ("DepthToSpaceFp32Test_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    DepthToSpaceFp32Test(backends, 2);
}

TEST_CASE ("DepthToSpaceUint8Test_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    DepthToSpaceUint8Test(backends, 2);
}

} // TEST_SUITE("DepthToSpace_CpuRefTests")


TEST_SUITE("DepthToSpace_CpuAccTests")
{

TEST_CASE ("DepthToSpaceFp32Test_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    DepthToSpaceFp32Test(backends, 2);
}

TEST_CASE ("DepthToSpaceUint8Test_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    DepthToSpaceUint8Test(backends, 2);
}

} // TEST_SUITE("DepthToSpace_CpuAccTests")

TEST_SUITE("DepthToSpace_GpuAccTests")
{

TEST_CASE ("DepthToSpaceFp32Test_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    DepthToSpaceFp32Test(backends, 2);
}

TEST_CASE ("DepthToSpaceUint8Test_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    DepthToSpaceUint8Test(backends, 2);
}

} // TEST_SUITE("DepthToSpace_GpuAccTests")

TEST_SUITE("SpaceToDepth_CpuRefTests")
{

TEST_CASE ("SpaceToDepthFp32Test_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    SpaceToDepthFp32Test(backends, 2);
}

TEST_CASE ("SpaceToDepthUint8Test_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    SpaceToDepthUint8Test(backends, 2);
}

} // TEST_SUITE("SpaceToDepth_CpuRefTests")

TEST_SUITE("SpaceToDepth_CpuAccTests")
{

TEST_CASE ("SpaceToDepthFp32Test_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    SpaceToDepthFp32Test(backends, 2);
}

TEST_CASE ("SpaceToDepthUint8Test_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    SpaceToDepthUint8Test(backends, 2);
}

} // TEST_SUITE("SpaceToDepth_CpuAccTests")

TEST_SUITE("SpaceToDepth_GpuAccTests")
{

TEST_CASE ("SpaceToDepthFp32Test_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    SpaceToDepthFp32Test(backends, 2);
}

TEST_CASE ("SpaceToDepthUint8Test_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    SpaceToDepthUint8Test(backends, 2);
}

} // TEST_SUITE("SpaceToDepth_GpuAccTests")

} // namespace armnnDelegate
