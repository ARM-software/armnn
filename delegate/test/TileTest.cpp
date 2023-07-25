//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TileTestHelper.hpp"

#include <armnn_delegate.hpp>
#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <schema_generated.h>
#include <tensorflow/lite/version.h>
#include <doctest/doctest.h>

namespace armnnDelegate
{
void TileFloat32Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<float> inputValues =
    {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f
    };

    // Set output data
    std::vector<float> expectedOutputValues =
    {
        0.f, 1.f, 2.f, 0.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 3.f, 4.f, 5.f,

        0.f, 1.f, 2.f, 0.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 3.f, 4.f, 5.f
    };

    // The multiples
    const std::vector<int32_t> multiplesValues = { 2, 2 };

    // Set shapes
    const std::vector<int32_t> inputShape = { 2, 3 };
    const std::vector<int32_t> multiplesShape = { 2 };
    const std::vector<int32_t> expectedOutputShape = { 4, 6 };

    TileFP32TestImpl(tflite::BuiltinOperator_TILE,
                     backends,
                     inputValues,
                     inputShape,
                     multiplesValues,
                     multiplesShape,
                     expectedOutputValues,
                     expectedOutputShape);
}

#if defined(TILE_GPUACC)
TEST_SUITE("TileTests_GpuAccTests")
{

    TEST_CASE ("Tile_Float32_GpuAcc_Test")
    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::GpuAcc };
        TileFloat32Test(backends);
    }

} // TEST_SUITE("Tile_Float32_GpuAcc_Test")
#endif

TEST_SUITE("TileTests_CpuAccTests")
{

    TEST_CASE ("Tile_Float32_CpuAcc_Test")
    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuAcc };
        TileFloat32Test(backends);
    }

} // TEST_SUITE("Tile_Float32_CpuAcc_Test")

TEST_SUITE("TileTests_CpuRefTests")
{

    TEST_CASE ("Tile_Float32_CpuRef_Test")
    {
        std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
        TileFloat32Test(backends);
    }

} // TEST_SUITE("Tile_Float32_CpuRef_Test")

} // namespace armnnDelegate