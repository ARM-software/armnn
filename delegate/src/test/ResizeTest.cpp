//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ResizeTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

void ResizeBiliniarFloat32Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<float> input1Values
        {
            0.0f, 1.0f, 2.0f,
            3.0f, 4.0f, 5.0f,
            6.0f, 7.0f, 8.0f
        };
    const std::vector<int32_t> input2NewShape { 5, 5 };

    // Calculate output data
    std::vector<float> expectedOutputValues
        {
            0.0f, 0.6f, 1.2f, 1.8f, 2.0f,
            1.8f, 2.4f, 3.0f, 3.6f, 3.8f,
            3.6f, 4.2f, 4.8f, 5.4f, 5.6f,
            5.4f, 6.0f, 6.6f, 7.2f, 7.4f,
            6.0f, 6.6f, 7.2f, 7.8f, 8.0f
        };

    const std::vector<int32_t> input1Shape { 1, 3, 3, 1 };
    const std::vector<int32_t> input2Shape { 2 };
    const std::vector<int32_t> expectedOutputShape = input2NewShape;

    ResizeFP32TestImpl(tflite::BuiltinOperator_RESIZE_BILINEAR,
                       backends,
                       input1Values,
                       input1Shape,
                       input2NewShape,
                       input2Shape,
                       expectedOutputValues,
                       expectedOutputShape);
}

void ResizeNearestNeighbourFloat32Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<float> input1Values {  1.0f, 2.0f, 3.0f, 4.0f }
    ;
    const std::vector<int32_t> input2NewShape { 1, 1 };

    // Calculate output data
    std::vector<float> expectedOutputValues { 1.0f };

    const std::vector<int32_t> input1Shape { 1, 2, 2, 1 };
    const std::vector<int32_t> input2Shape { 2 };
    const std::vector<int32_t> expectedOutputShape = input2NewShape;

    ResizeFP32TestImpl(tflite::BuiltinOperator_RESIZE_NEAREST_NEIGHBOR,
                       backends,
                       input1Values,
                       input1Shape,
                       input2NewShape,
                       input2Shape,
                       expectedOutputValues,
                       expectedOutputShape);
}

TEST_SUITE("ResizeTests_GpuAccTests")
{

TEST_CASE ("Resize_Biliniar_Float32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ResizeBiliniarFloat32Test(backends);
}

TEST_CASE ("Resize_NearestNeighbour_Float32_GpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
    ResizeNearestNeighbourFloat32Test(backends);
}

} // TEST_SUITE("ResizeTests_GpuAccTests")


TEST_SUITE("ResizeTests_CpuAccTests")
{

TEST_CASE ("Resize_Biliniar_Float32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ResizeBiliniarFloat32Test(backends);
}

TEST_CASE ("Resize_NearestNeighbour_Float32_CpuAcc_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
    ResizeNearestNeighbourFloat32Test(backends);
}

} // TEST_SUITE("ResizeTests_CpuAccTests")


TEST_SUITE("ResizeTests_CpuRefTests")
{

TEST_CASE ("Resize_Biliniar_Float32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ResizeBiliniarFloat32Test(backends);
}

TEST_CASE ("Resize_NearestNeighbour_Float32_CpuRef_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ResizeNearestNeighbourFloat32Test(backends);
}

} // TEST_SUITE("ResizeTests_CpuRefTests")

} // namespace armnnDelegate
