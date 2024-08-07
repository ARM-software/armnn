//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchSpaceTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

// BatchToSpaceND Operator
void BatchToSpaceNDFp32Test()
{
    std::vector<int32_t> inputShape { 4, 1, 1, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> expectedOutputValues { 1.0f, 2.0f, 3.0f, 4.0f };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    BatchSpaceTest<float>(tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                          ::tflite::TensorType_FLOAT32,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          crops,
                          expectedOutputValues);
}

void BatchToSpaceNDFp32BatchOneTest()
{
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 1 };

    std::vector<float> inputValues { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> expectedOutputValues { 1.0f, 2.0f, 3.0f, 4.0f };

    std::vector<unsigned int> blockShape({1, 1});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    BatchSpaceTest<float>(tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                          ::tflite::TensorType_FLOAT32,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          crops,
                          expectedOutputValues);
}

void BatchToSpaceNDUint8Test()
{
    std::vector<int32_t> inputShape { 4, 1, 1, 3 };
    std::vector<int32_t> expectedOutputShape { 1, 2, 2, 3 };

    std::vector<uint8_t> inputValues { 1, 2, 3, 4, 5, 6, 7 };
    std::vector<uint8_t> expectedOutputValues { 1, 2, 3, 4, 5, 6, 7 };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> crops = {{0, 0}, {0, 0}};

    BatchSpaceTest<uint8_t>(tflite::BuiltinOperator_BATCH_TO_SPACE_ND,
                          ::tflite::TensorType_UINT8,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          crops,
                          expectedOutputValues);
}

// SpaceToBatchND Operator
void SpaceToBatchNDFp32Test()
{
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> expectedOutputShape { 4, 1, 1, 1 };

    std::vector<float> inputValues { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> expectedOutputValues { 1.0f, 2.0f, 3.0f, 4.0f };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> padding = {{0, 0}, {0, 0}};

    BatchSpaceTest<float>(tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                          ::tflite::TensorType_FLOAT32,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          padding,
                          expectedOutputValues);
}

void SpaceToBatchNDFp32PaddingTest()
{
    std::vector<int32_t> inputShape { 2, 2, 4, 1 };
    std::vector<int32_t> expectedOutputShape { 8, 1, 3, 1 };

    std::vector<float> inputValues { 1.0f,  2.0f,  3.0f,  4.0f,
                                     5.0f,  6.0f,  7.0f,  8.0f,
                                     9.0f,  10.0f, 11.0f, 12.0f,
                                     13.0f, 14.0f, 15.0f, 16.0f };

    std::vector<float> expectedOutputValues { 0.0f, 1.0f, 3.0f,  0.0f, 9.0f, 11.0f,
                                              0.0f, 2.0f, 4.0f,  0.0f, 10.0f, 12.0f,
                                              0.0f, 5.0f, 7.0f,  0.0f, 13.0f, 15.0f,
                                              0.0f, 6.0f, 8.0f,  0.0f, 14.0f, 16.0f };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> padding = {{0, 0}, {2, 0}};

    BatchSpaceTest<float>(tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                          ::tflite::TensorType_FLOAT32,
                          inputShape,
                          expectedOutputShape,
                          inputValues,
                          blockShape,
                          padding,
                          expectedOutputValues);
}

void SpaceToBatchNDUint8Test()
{
    std::vector<int32_t> inputShape { 1, 2, 2, 3 };
    std::vector<int32_t> expectedOutputShape { 4, 1, 1, 3 };

    std::vector<uint8_t> inputValues { 1, 2, 3, 4, 5, 6, 7 };
    std::vector<uint8_t> expectedOutputValues { 1, 2, 3, 4, 5, 6, 7 };

    std::vector<unsigned int> blockShape({2, 2});
    std::vector<std::pair<unsigned int, unsigned int>> padding = {{0, 0}, {0, 0}};

    BatchSpaceTest<uint8_t>(tflite::BuiltinOperator_SPACE_TO_BATCH_ND,
                            ::tflite::TensorType_UINT8,
                            inputShape,
                            expectedOutputShape,
                            inputValues,
                            blockShape,
                            padding,
                            expectedOutputValues);
}

// BatchToSpaceND Tests
TEST_SUITE("BatchToSpaceNDTests")
{

TEST_CASE ("BatchToSpaceND_Fp32_Test")
{
    BatchToSpaceNDFp32Test();
}

TEST_CASE ("BatchToSpaceND_Fp32_BatchOne_Test")
{
    BatchToSpaceNDFp32BatchOneTest();
}

TEST_CASE ("BatchToSpaceND_Uint8_Test")
{
    BatchToSpaceNDUint8Test();
}

}

// SpaceToBatchND Tests
TEST_SUITE("SpaceToBatchND_Tests")
{

TEST_CASE ("SpaceToBatchND_Fp32_Test")
{
    SpaceToBatchNDFp32Test();
}

TEST_CASE ("SpaceToBatchND_Fp32_Padding_Test")
{
    SpaceToBatchNDFp32PaddingTest();
}

TEST_CASE ("SpaceToBatchND_Uint8_Test")
{
    SpaceToBatchNDUint8Test();
}

}

} // namespace armnnDelegate