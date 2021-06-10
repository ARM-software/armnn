//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <doctest/doctest.h>

using namespace armnn::armcomputetensorutils;

TEST_SUITE("ArmComputeTensorUtils")
{
TEST_CASE("BuildArmComputeTensorInfoTest")
{

    const armnn::TensorShape tensorShape = { 1, 2, 3, 4 };
    const armnn::DataType dataType = armnn::DataType::QAsymmU8;

    const std::vector<float> quantScales = { 1.5f, 2.5f, 3.5f, 4.5f };
    const float quantScale = quantScales[0];
    const int32_t quantOffset = 128;

    // Tensor info with per-tensor quantization
    const armnn::TensorInfo tensorInfo0(tensorShape, dataType, quantScale, quantOffset);
    const arm_compute::TensorInfo aclTensorInfo0 = BuildArmComputeTensorInfo(tensorInfo0);

    const arm_compute::TensorShape& aclTensorShape = aclTensorInfo0.tensor_shape();
    CHECK(aclTensorShape.num_dimensions() == tensorShape.GetNumDimensions());
    for(unsigned int i = 0u; i < tensorShape.GetNumDimensions(); ++i)
    {
        // NOTE: arm_compute tensor dimensions are stored in the opposite order
        CHECK(aclTensorShape[i] == tensorShape[tensorShape.GetNumDimensions() - i - 1]);
    }

    CHECK(aclTensorInfo0.data_type() == arm_compute::DataType::QASYMM8);
    CHECK(aclTensorInfo0.quantization_info().scale()[0] == quantScale);

    // Tensor info with per-axis quantization
    const armnn::TensorInfo tensorInfo1(tensorShape, dataType, quantScales, 0);
    const arm_compute::TensorInfo aclTensorInfo1 = BuildArmComputeTensorInfo(tensorInfo1);

    CHECK(aclTensorInfo1.quantization_info().scale() == quantScales);
}

}
