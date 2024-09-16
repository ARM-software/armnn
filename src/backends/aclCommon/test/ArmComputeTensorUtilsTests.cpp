//
// Copyright © 2019, 2024 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

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
    const armnn::TensorInfo tensorInfo1(tensorShape, armnn::DataType::QSymmS8, quantScales, 0);
    const arm_compute::TensorInfo aclTensorInfo1 = BuildArmComputeTensorInfo(tensorInfo1);

    CHECK(aclTensorInfo1.quantization_info().scale() == quantScales);
    CHECK(aclTensorInfo1.data_type() == arm_compute::DataType::QSYMM8_PER_CHANNEL);
}

TEST_CASE("IsMultiAxesReduceSupportedQuantizationTest")
{
    // Input/Output shape used for the test
    const armnn::TensorShape inputShape = { 1, 2, 2, 2 };
    const armnn::TensorShape outputShape = { 1, 1, 1, 2 };

    // Descriptor used for the test
    armnn::ReduceDescriptor descriptor;
    descriptor.m_KeepDims = true;
    descriptor.m_vAxis = { 1, 2 };

    // Mock function to use instead of backend-specific NeonReduceWorkloadValidate/ClReduceWorkloadValidate
    auto MockSupportFunction = [](const armnn::TensorInfo& input,
                                   const armnn::TensorInfo& output,
                                   const armnn::ReduceDescriptor&) -> arm_compute::Status {
        // The Validate functions that would be called instead of this one are ACL backend-specific functions
        // performing all sorts of validation checks. This test checks specifically the ArmNN Utils function
        // IsMultiAxesReduceSupported, to ensure the output tensor info is considered as well as the input.
        // This mock function is a simplification of the ACL validation, checking only the type
        // and quantization information, which is sufficient to test IsMultiAxesReduceSupported itself.
        return (input.IsTypeSpaceMatch(output)) ? arm_compute::Status{} :
                                                  arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR};
    };

    // 1. Invalid case test - Input/Output quantization is different
    const armnn::TensorInfo input0(inputShape, armnn::DataType::QAsymmU8, 50.0f/255.0f, 0);
    const armnn::TensorInfo output0(outputShape, armnn::DataType::QAsymmU8, 5000.0f/255.0f, 0);

    arm_compute::Status status0;
    IsMultiAxesReduceSupported(MockSupportFunction, input0, output0, descriptor, status0);

    CHECK(!status0);

    // 2. Valid case test - Input/Output quantization is the same
    const armnn::TensorInfo input1(inputShape, armnn::DataType::QAsymmU8, 50.0f/255.0f, 0);
    const armnn::TensorInfo output1(outputShape, armnn::DataType::QAsymmU8, 50.0f/255.0f, 0);

    arm_compute::Status status1;
    IsMultiAxesReduceSupported(MockSupportFunction, input1, output1, descriptor, status1);

    CHECK(status1);
}

}
