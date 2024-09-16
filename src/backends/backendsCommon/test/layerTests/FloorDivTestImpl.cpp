//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FloorDivTestImpl.hpp"

#include "ElementwiseTestImpl.hpp"

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> SimpleFloorDivTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    unsigned int shape[] = { 1, 1, 2, 2 };

    armnn::TensorInfo inputTensorInfo0({1, 1, 2, 2}, ArmnnType);
    inputTensorInfo0.SetQuantizationScale(0.1f);

    armnn::TensorInfo inputTensorInfo1({1, 1, 2, 2}, ArmnnType);
    inputTensorInfo1.SetQuantizationScale(0.1f);

    armnn::TensorInfo outputTensorInfo({1, 1, 2, 2}, ArmnnType);
    outputTensorInfo.SetQuantizationScale(0.1f);

    std::vector<T> input0 = ConvertToDataType<ArmnnType>(
    {
        15, -17, -22, 21
    }, inputTensorInfo0);

    std::vector<T> input1 = ConvertToDataType<ArmnnType>(
    {
        5, 2, -7, 3
    }, inputTensorInfo1);

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<T> expectedOutput = ConvertToDataType<ArmnnType>(
    {
        3, -9, 3, 7
    },outputTensorInfo);

    return ElementwiseTestHelper<4, ArmnnType, ArmnnType>(
        workloadFactory,
        memoryManager,
        armnn::BinaryOperation::FloorDiv,
        shape,
        input0,
        shape,
        input1,
        shape,
        expectedOutput,
        tensorHandleFactory);
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
    SimpleFloorDivTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
    SimpleFloorDivTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 4>
    SimpleFloorDivTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);