//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PadTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void MirrorPadSymmetric2dTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 3, 3 };
    std::vector<int32_t> outputShape { 7, 7 };
    std::vector<int32_t> paddingShape { 2, 2 };

    std::vector<float> inputValues =
    {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };

    std::vector<float> expectedOutputValues =
    {
        5.0f, 4.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f,
        2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 2.0f,
        2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 3.0f, 2.0f,
        5.0f, 4.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f,
        8.0f, 7.0f, 7.0f, 8.0f, 9.0f, 9.0f, 8.0f,
        8.0f, 7.0f, 7.0f, 8.0f, 9.0f, 9.0f, 8.0f,
        5.0f, 4.0f, 4.0f, 5.0f, 6.0f, 6.0f, 5.0f
    };

    std::vector<int32_t> paddingDim = { 2, 2, 2, 2 };

    PadTest<float>(tflite::BuiltinOperator_MIRROR_PAD,
                   ::tflite::TensorType_FLOAT32,
                   backends,
                   inputShape,
                   paddingShape,
                   outputShape,
                   inputValues,
                   paddingDim,
                   expectedOutputValues,
                   0,    // Padding value - Not used in these tests.
                   1.0f, // Scale
                   0,    // Offset
                   tflite::MirrorPadMode_SYMMETRIC);
}

void MirrorPadReflect2dTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 3, 3 };
    std::vector<int32_t> outputShape { 7, 7 };
    std::vector<int32_t> paddingShape { 2, 2 };

    std::vector<float> inputValues =
    {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };

    std::vector<float> expectedOutputValues =
    {
        9.0f, 8.0f, 7.0f, 8.0f, 9.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 4.0f, 5.0f, 6.0f, 5.0f, 4.0f,
        3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f,
        6.0f, 5.0f, 4.0f, 5.0f, 6.0f, 5.0f, 4.0f,
        9.0f, 8.0f, 7.0f, 8.0f, 9.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 4.0f, 5.0f, 6.0f, 5.0f, 4.0f,
        3.0f, 2.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f
    };

    std::vector<int32_t> paddingDim = { 2, 2, 2, 2 };

    PadTest<float>(tflite::BuiltinOperator_MIRROR_PAD,
                   ::tflite::TensorType_FLOAT32,
                   backends,
                   inputShape,
                   paddingShape,
                   outputShape,
                   inputValues,
                   paddingDim,
                   expectedOutputValues,
                   0,    // Padding value - Not used in these tests.
                   1.0f, // Scale
                   0,    // Offset
                   tflite::MirrorPadMode_REFLECT);
}

void MirrorPadSymmetric3dTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 2, 2, 2 };
    std::vector<int32_t> outputShape { 4, 4, 4 };
    std::vector<int32_t> paddingShape { 3, 2 };

    std::vector<float> inputValues =
    {
        // Channel 0, Height (2) x Width (2)
        1.0f, 2.0f,
        3.0f, 4.0f,

        // Channel 1, Height (2) x Width (2)
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    std::vector<float> expectedOutputValues =
    {
        1.0f, 1.0f, 2.0f, 2.0f,
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
        3.0f, 3.0f, 4.0f, 4.0f,

        1.0f, 1.0f, 2.0f, 2.0f,
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
        3.0f, 3.0f, 4.0f, 4.0f,

        5.0f, 5.0f, 6.0f, 6.0f,
        5.0f, 5.0f, 6.0f, 6.0f,
        7.0f, 7.0f, 8.0f, 8.0f,
        7.0f, 7.0f, 8.0f, 8.0f,

        5.0f, 5.0f, 6.0f, 6.0f,
        5.0f, 5.0f, 6.0f, 6.0f,
        7.0f, 7.0f, 8.0f, 8.0f,
        7.0f, 7.0f, 8.0f, 8.0f
    };

    std::vector<int32_t> paddingDim = { 1, 1, 1, 1, 1, 1 };

    PadTest<float>(tflite::BuiltinOperator_MIRROR_PAD,
                   ::tflite::TensorType_FLOAT32,
                   backends,
                   inputShape,
                   paddingShape,
                   outputShape,
                   inputValues,
                   paddingDim,
                   expectedOutputValues,
                   0,    // Padding value - Not used in these tests.
                   1.0f, // Scale
                   0,    // Offset
                   tflite::MirrorPadMode_SYMMETRIC);
}

void MirrorPadReflect3dTest(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 2, 2, 2 };
    std::vector<int32_t> outputShape { 4, 4, 4 };
    std::vector<int32_t> paddingShape { 3, 2 };

    std::vector<float> inputValues =
    {
        // Channel 0, Height (2) x Width (2)
        1.0f, 2.0f,
        3.0f, 4.0f,

        // Channel 1, Height (2) x Width (2)
        5.0f, 6.0f,
        7.0f, 8.0f
    };

    std::vector<float> expectedOutputValues =
    {
        8.0f, 7.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 6.0f, 5.0f,
        8.0f, 7.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 6.0f, 5.0f,

        4.0f, 3.0f, 4.0f, 3.0f,
        2.0f, 1.0f, 2.0f, 1.0f,
        4.0f, 3.0f, 4.0f, 3.0f,
        2.0f, 1.0f, 2.0f, 1.0f,

        8.0f, 7.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 6.0f, 5.0f,
        8.0f, 7.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 6.0f, 5.0f,

        4.0f, 3.0f, 4.0f, 3.0f,
        2.0f, 1.0f, 2.0f, 1.0f,
        4.0f, 3.0f, 4.0f, 3.0f,
        2.0f, 1.0f, 2.0f, 1.0f
    };

    std::vector<int32_t> paddingDim = { 1, 1, 1, 1, 1, 1 };

    PadTest<float>(tflite::BuiltinOperator_MIRROR_PAD,
                   ::tflite::TensorType_FLOAT32,
                   backends,
                   inputShape,
                   paddingShape,
                   outputShape,
                   inputValues,
                   paddingDim,
                   expectedOutputValues,
                   0,    // Padding value - Not used in these tests.
                   1.0f, // Scale
                   0,    // Offset
                   tflite::MirrorPadMode_REFLECT);
}

void MirrorPadSymmetricUint8Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 3, 3 };
    std::vector<int32_t> outputShape { 5, 7 };
    std::vector<int32_t> paddingShape { 2, 2 };

    std::vector<uint8_t> inputValues =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<uint8_t> expectedOutputValues =
    {
        2, 1, 1, 2, 3, 3, 2,
        2, 1, 1, 2, 3, 3, 2,
        5, 4, 4, 5, 6, 6, 5,
        8, 7, 7, 8, 9, 9, 8,
        8, 7, 7, 8, 9, 9, 8,
    };

    std::vector<int32_t> paddingDim = { 1, 1, 2, 2 };

    PadTest<uint8_t>(tflite::BuiltinOperator_MIRROR_PAD,
                     ::tflite::TensorType_UINT8,
                     backends,
                     inputShape,
                     paddingShape,
                     outputShape,
                     inputValues,
                     paddingDim,
                     expectedOutputValues,
                     0,    // Padding value - Not used in these tests.
                     1.0f, // Scale
                     1,    // Offset
                     tflite::MirrorPadMode_SYMMETRIC);
}

void MirrorPadReflectInt8Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 3, 3 };
    std::vector<int32_t> outputShape { 7, 5 };
    std::vector<int32_t> paddingShape { 2, 2 };

    std::vector<int8_t> inputValues =
    {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    std::vector<int8_t> expectedOutputValues =
    {
        8, 7, 8, 9, 8,
        5, 4, 5, 6, 5,
        2, 1, 2, 3, 2,
        5, 4, 5, 6, 5,
        8, 7, 8, 9, 8,
        5, 4, 5, 6, 5,
        2, 1, 2, 3, 2
    };

    std::vector<int32_t> paddingDim = { 2, 2, 1, 1 };

    PadTest<int8_t>(tflite::BuiltinOperator_MIRROR_PAD,
                    ::tflite::TensorType_INT8,
                    backends,
                    inputShape,
                    paddingShape,
                    outputShape,
                    inputValues,
                    paddingDim,
                    expectedOutputValues,
                    0,    // Padding value - Not used in these tests.
                    1.0f, // Scale
                    1,    // Offset
                    tflite::MirrorPadMode_REFLECT);
}

TEST_SUITE("MirrorPad_CpuRefTests")
{

TEST_CASE ("MirrorPadSymmetric2d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MirrorPadSymmetric2dTest(backends);
}

TEST_CASE ("MirrorPadReflect2d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MirrorPadReflect2dTest(backends);
}

TEST_CASE ("MirrorPadSymmetric3d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MirrorPadSymmetric3dTest(backends);
}

TEST_CASE ("MirrorPadReflect3d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MirrorPadReflect3dTest(backends);
}

TEST_CASE ("MirrorPadSymmetricUint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MirrorPadSymmetricUint8Test(backends);
}

TEST_CASE ("MirrorPadSymmetricInt8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    MirrorPadReflectInt8Test(backends);
}

} // TEST_SUITE("MirrorPad_CpuRefTests")

} // namespace armnnDelegate