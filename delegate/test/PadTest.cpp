//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PadTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void Pad2dTest(std::vector<armnn::BackendId>& backends,
               tflite::BuiltinOperator padOperatorCode = tflite::BuiltinOperator_PAD,
               float pad = 0.0f)
{
    // Set input data
    std::vector<int32_t> inputShape { 2, 2, 2 };
    std::vector<int32_t> outputShape { 3, 5, 6 };
    std::vector<int32_t> paddingShape { 3, 2 };

    std::vector<float> inputValues = { 0.0f,  4.0f,
                                       2.0f, -5.0f,
                                       6.0f,  1.0f,
                                       5.0f, -2.0f };

    std::vector<float> expectedOutputValues = { pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, 0.0f, 4.0f, pad, pad,
                                                pad, pad, 2.0f, -5.0f, pad, pad,
                                                pad, pad, pad, pad, pad, pad,

                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, 6.0f, 1.0f, pad, pad,
                                                pad, pad, 5.0f, -2.0f, pad, pad,
                                                pad, pad, pad, pad, pad, pad,

                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad };

    std::vector<int32_t> paddingDim = { 0, 1, 2, 1, 2, 2 };

    PadTest<float>(padOperatorCode,
                   ::tflite::TensorType_FLOAT32,
                   backends,
                   inputShape,
                   paddingShape,
                   outputShape,
                   inputValues,
                   paddingDim,
                   expectedOutputValues,
                   pad);
}

void Pad3dTest(std::vector<armnn::BackendId>& backends,
               tflite::BuiltinOperator padOperatorCode = tflite::BuiltinOperator_PAD,
               float pad = 0.0f)
{
    // Set input data
    std::vector<int32_t> inputShape { 2, 2, 2 };
    std::vector<int32_t> outputShape { 3, 5, 6 };
    std::vector<int32_t> paddingShape { 3, 2 };

    std::vector<float> inputValues = { 0.0f, 4.0f,
                                       2.0f, 5.0f,
                                       6.0f, 1.0f,
                                       5.0f, 2.0f };

    std::vector<float> expectedOutputValues = { pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, 0.0f, 4.0f, pad, pad,
                                                pad, pad, 2.0f, 5.0f, pad, pad,
                                                pad, pad, pad, pad, pad, pad,

                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, 6.0f, 1.0f, pad, pad,
                                                pad, pad, 5.0f, 2.0f, pad, pad,
                                                pad, pad, pad, pad, pad, pad,

                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad,
                                                pad, pad, pad, pad, pad, pad };

    std::vector<int32_t> paddingDim = { 0, 1, 2, 1, 2, 2 };

    PadTest<float>(padOperatorCode,
                   ::tflite::TensorType_FLOAT32,
                   backends,
                   inputShape,
                   paddingShape,
                   outputShape,
                   inputValues,
                   paddingDim,
                   expectedOutputValues,
                   pad);
}

void Pad4dTest(std::vector<armnn::BackendId>& backends,
               tflite::BuiltinOperator padOperatorCode = tflite::BuiltinOperator_PAD,
               float pad = 0.0f)
{
    // Set input data
    std::vector<int32_t> inputShape { 2, 2, 3, 2 };
    std::vector<int32_t> outputShape { 4, 5, 7, 4 };
    std::vector<int32_t> paddingShape { 4, 2 };

    std::vector<float> inputValues = { 0.0f,  1.0f,
                                       2.0f,  3.0f,
                                       4.0f,  5.0f,

                                       6.0f,  7.0f,
                                       8.0f,  9.0f,
                                       10.0f, 11.0f,

                                       12.0f, 13.0f,
                                       14.0f, 15.0f,
                                       16.0f, 17.0f,

                                       18.0f, 19.0f,
                                       20.0f, 21.0f,
                                       22.0f, 23.0f };

    std::vector<float> expectedOutputValues = { pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, 0.0f, 1.0f, pad,
                                                pad, 2.0f, 3.0f, pad,
                                                pad, 4.0f, 5.0f, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, 6.0f, 7.0f, pad,
                                                pad, 8.0f, 9.0f, pad,
                                                pad, 10.0f, 11.0f, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, 12.0f, 13.0f, pad,
                                                pad, 14.0f, 15.0f, pad,
                                                pad, 16.0f, 17.0f, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, 18.0f, 19.0f, pad,
                                                pad, 20.0f, 21.0f, pad,
                                                pad, 22.0f, 23.0f, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,

                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad,
                                                pad, pad, pad, pad };

    std::vector<int32_t> paddingDim = { 1, 1, 2, 1, 3, 1, 1, 1 };

    PadTest<float>(padOperatorCode,
                   ::tflite::TensorType_FLOAT32,
                   backends,
                   inputShape,
                   paddingShape,
                   outputShape,
                   inputValues,
                   paddingDim,
                   expectedOutputValues,
                   pad);
}

void PadInt8Test(std::vector<armnn::BackendId>& backends,
                 tflite::BuiltinOperator padOperatorCode = tflite::BuiltinOperator_PAD,
                 int8_t paddingValue = 0,
                 int8_t p = 3,
                 float quantizationScale = -2.0f,
                 int32_t quantizationOffset = 3)
{
    // Set input data
    std::vector<int32_t> inputShape { 2, 2, 2 };
    std::vector<int32_t> outputShape { 3, 5, 6 };
    std::vector<int32_t> paddingShape { 3, 2 };

    std::vector<int8_t> inputValues = { 0,  4,
                                        2, -5,
                                        6,  1,
                                        5, -2 };

    std::vector<int8_t> expectedOutputValues = { p, p, p, p, p, p,
                                                 p, p, p, p, p, p,
                                                 p, p, 0, 4, p, p,
                                                 p, p, 2, -5, p, p,
                                                 p, p, p, p, p, p,

                                                 p, p, p, p, p, p,
                                                 p, p, p, p, p, p,
                                                 p, p, 6, 1, p, p,
                                                 p, p, 5, -2, p, p,
                                                 p, p, p, p, p, p,

                                                 p, p, p, p, p, p,
                                                 p, p, p, p, p, p,
                                                 p, p, p, p, p, p,
                                                 p, p, p, p, p, p,
                                                 p, p, p, p, p, p };

    std::vector<int32_t> paddingDim = { 0, 1, 2, 1, 2, 2 };

    PadTest<int8_t>(padOperatorCode,
                    ::tflite::TensorType_INT8,
                    backends,
                    inputShape,
                    paddingShape,
                    outputShape,
                    inputValues,
                    paddingDim,
                    expectedOutputValues,
                    paddingValue,
                    quantizationScale,
                    quantizationOffset);
}

void PadUint8Test(std::vector<armnn::BackendId>& backends,
                  tflite::BuiltinOperator padOperatorCode = tflite::BuiltinOperator_PAD,
                  uint8_t paddingValue = 0,
                  uint8_t p = 3,
                  float quantizationScale = -2.0f,
                  int32_t quantizationOffset = 3)
{
    // Set input data
    std::vector<int32_t> inputShape { 2, 2, 2 };
    std::vector<int32_t> outputShape { 3, 5, 6 };
    std::vector<int32_t> paddingShape { 3, 2 };

    std::vector<uint8_t> inputValues = { 0, 4,
                                         2, 5,
                                         6, 1,
                                         5, 2 };

    std::vector<uint8_t> expectedOutputValues = { p, p, p, p, p, p,
                                                  p, p, p, p, p, p,
                                                  p, p, 0, 4, p, p,
                                                  p, p, 2, 5, p, p,
                                                  p, p, p, p, p, p,

                                                  p, p, p, p, p, p,
                                                  p, p, p, p, p, p,
                                                  p, p, 6, 1, p, p,
                                                  p, p, 5, 2, p, p,
                                                  p, p, p, p, p, p,

                                                  p, p, p, p, p, p,
                                                  p, p, p, p, p, p,
                                                  p, p, p, p, p, p,
                                                  p, p, p, p, p, p,
                                                  p, p, p, p, p, p };

    std::vector<int32_t> paddingDim = { 0, 1, 2, 1, 2, 2 };

    PadTest<uint8_t>(padOperatorCode,
                     ::tflite::TensorType_UINT8,
                     backends,
                     inputShape,
                     paddingShape,
                     outputShape,
                     inputValues,
                     paddingDim,
                     expectedOutputValues,
                     paddingValue,
                     quantizationScale,
                     quantizationOffset);
}

TEST_SUITE("Pad_CpuRefTests")
{

TEST_CASE ("Pad2d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Pad2dTest(backends);
}

TEST_CASE ("Pad3d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Pad3dTest(backends);
}

TEST_CASE ("Pad4d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Pad4dTest(backends);
}

TEST_CASE ("Pad_Int8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    PadInt8Test(backends);
}

TEST_CASE ("Pad_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    PadUint8Test(backends);
}

TEST_CASE ("PadV22d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Pad2dTest(backends, tflite::BuiltinOperator_PADV2, -2.5);
}

TEST_CASE ("PadV23d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Pad3dTest(backends, tflite::BuiltinOperator_PADV2, 2.0);
}

TEST_CASE ("PadV24d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Pad4dTest(backends, tflite::BuiltinOperator_PADV2, -1.33);
}

TEST_CASE ("PadV2_Int8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    PadInt8Test(backends, tflite::BuiltinOperator_PADV2, -1, -1);
}

TEST_CASE ("PadV2_Uint8_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    PadUint8Test(backends, tflite::BuiltinOperator_PADV2, -1, -1);
}

} // TEST_SUITE("Pad_CpuRefTests")

TEST_SUITE("Pad_CpuAccTests")
{

TEST_CASE ("Pad2d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Pad2dTest(backends);
}

TEST_CASE ("Pad3d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Pad3dTest(backends);
}

TEST_CASE ("Pad4d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Pad4dTest(backends);
}

TEST_CASE ("Pad_Int8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    PadInt8Test(backends);
}

TEST_CASE ("Pad_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    PadUint8Test(backends);
}

TEST_CASE ("PadV22d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Pad2dTest(backends, tflite::BuiltinOperator_PADV2, -2.5);
}

TEST_CASE ("PadV23d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Pad3dTest(backends, tflite::BuiltinOperator_PADV2, 2.0);
}

TEST_CASE ("PadV24d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Pad4dTest(backends, tflite::BuiltinOperator_PADV2, -1.33);
}

TEST_CASE ("PadV2_Int8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    PadInt8Test(backends, tflite::BuiltinOperator_PADV2, -1, -1);
}

TEST_CASE ("PadV2_Uint8_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    PadUint8Test(backends, tflite::BuiltinOperator_PADV2, -1, -1);
}

} // TEST_SUITE("Pad_CpuAccTests")

TEST_SUITE("Pad_GpuAccTests")
{

TEST_CASE ("Pad2d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Pad2dTest(backends);
}

TEST_CASE ("Pad3d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Pad3dTest(backends);
}

TEST_CASE ("Pad4d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Pad4dTest(backends);
}

TEST_CASE ("Pad_Int8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    PadInt8Test(backends);
}

TEST_CASE ("Pad_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    PadUint8Test(backends);
}

TEST_CASE ("PadV22d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Pad2dTest(backends, tflite::BuiltinOperator_PADV2, -2.5);
}

TEST_CASE ("PadV23d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Pad3dTest(backends, tflite::BuiltinOperator_PADV2, 2.0);
}

TEST_CASE ("PadV24d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Pad4dTest(backends, tflite::BuiltinOperator_PADV2, -1.33);
}

TEST_CASE ("PadV2_Int8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    PadInt8Test(backends, tflite::BuiltinOperator_PADV2, -1, -1);
}

TEST_CASE ("PadV2_Uint8_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    PadUint8Test(backends, tflite::BuiltinOperator_PADV2, -1, -1);
}

} // TEST_SUITE("Pad_GpuAccTests")

} // namespace armnnDelegate