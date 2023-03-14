//
// Copyright © 2021, 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FillTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void Fill2dTest(std::vector<armnn::BackendId>& backends,
               tflite::BuiltinOperator fillOperatorCode = tflite::BuiltinOperator_FILL,
               float fill = 2.0f )
{
    std::vector<int32_t> inputShape { 2 };
    std::vector<int32_t> tensorShape { 2, 2 };
    std::vector<float> expectedOutputValues = { fill, fill,
                                                fill, fill };

    FillTest<float>(fillOperatorCode,
                    ::tflite::TensorType_FLOAT32,
                    backends,
                    inputShape,
                    tensorShape,
                    expectedOutputValues,
                    fill);
}

void Fill3dTest(std::vector<armnn::BackendId>& backends,
               tflite::BuiltinOperator fillOperatorCode = tflite::BuiltinOperator_FILL,
               float fill = 5.0f )
{
    std::vector<int32_t> inputShape { 3 };
    std::vector<int32_t> tensorShape { 3, 3, 3 };
    std::vector<float> expectedOutputValues = { fill, fill, fill,
                                                fill, fill, fill,
                                                fill, fill, fill,

                                                fill, fill, fill,
                                                fill, fill, fill,
                                                fill, fill, fill,

                                                fill, fill, fill,
                                                fill, fill, fill,
                                                fill, fill, fill };

    FillTest<float>(fillOperatorCode,
                    ::tflite::TensorType_FLOAT32,
                    backends,
                    inputShape,
                    tensorShape,
                    expectedOutputValues,
                    fill);
}

void Fill4dTest(std::vector<armnn::BackendId>& backends,
               tflite::BuiltinOperator fillOperatorCode = tflite::BuiltinOperator_FILL,
               float fill = 3.0f )
{
    std::vector<int32_t> inputShape { 4 };
    std::vector<int32_t> tensorShape { 2, 2, 4, 4 };
    std::vector<float> expectedOutputValues = { fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,

                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,

                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,

                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill,
                                                fill, fill, fill, fill };

    FillTest<float>(fillOperatorCode,
                    ::tflite::TensorType_FLOAT32,
                    backends,
                    inputShape,
                    tensorShape,
                    expectedOutputValues,
                    fill);
}

void FillInt32Test(std::vector<armnn::BackendId>& backends,
                  tflite::BuiltinOperator fillOperatorCode = tflite::BuiltinOperator_FILL,
                  int32_t fill = 2 )
{
    std::vector<int32_t> inputShape { 2 };
    std::vector<int32_t> tensorShape { 2, 2 };
    std::vector<int32_t> expectedOutputValues = { fill, fill,
                                                  fill, fill };

    FillTest<int32_t>(fillOperatorCode,
                      ::tflite::TensorType_INT32,
                      backends,
                      inputShape,
                      tensorShape,
                      expectedOutputValues,
                      fill);
}

TEST_SUITE("Fill_CpuRefTests")
{

TEST_CASE ("Fill2d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Fill2dTest(backends);
}

TEST_CASE ("Fill3d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Fill3dTest(backends);
}

TEST_CASE ("Fill3d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Fill3dTest(backends);
}

TEST_CASE ("Fill4d_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    Fill4dTest(backends);
}

TEST_CASE ("FillInt32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    FillInt32Test(backends);
}

}

TEST_SUITE("Fill_CpuAccTests")
{

TEST_CASE ("Fill2d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Fill2dTest(backends);
}

TEST_CASE ("Fill3d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Fill3dTest(backends);
}

TEST_CASE ("Fill3d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Fill3dTest(backends);
}

TEST_CASE ("Fill4d_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    Fill4dTest(backends);
}

TEST_CASE ("FillInt32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    FillInt32Test(backends);
}

}

TEST_SUITE("Fill_GpuAccTests")
{

TEST_CASE ("Fill2d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Fill2dTest(backends);
}

TEST_CASE ("Fill3d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Fill3dTest(backends);
}

TEST_CASE ("Fill3d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Fill3dTest(backends);
}

TEST_CASE ("Fill4d_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    Fill4dTest(backends);
}

TEST_CASE ("FillInt32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    FillInt32Test(backends);
}

}

} // namespace armnnDelegate