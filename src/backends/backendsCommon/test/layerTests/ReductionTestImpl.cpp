//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReductionTestImpl.hpp"

#include <DataTypeUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <iostream>

namespace
{

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<float, 4> ReductionTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::TensorInfo inputTensorInfo,
        const armnn::TensorInfo outputTensorInfo,
        const std::vector<float>& inputData,
        const std::vector<float>& outputData,
        const std::vector<int32_t> vAxis,
        const armnn::ReduceOperation reduceOperation,
        bool keepDims = false)
{
    IgnoreUnused(memoryManager);
    auto inputTensor = ConvertToDataType<ArmnnType>(inputData, inputTensorInfo);

    std::vector<float> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ReduceQueueDescriptor descriptor;
    std::vector<uint32_t> updated_idx;
    uint32_t resolvedAxis = 0;
    for (uint32_t i = 0; i < vAxis.size(); ++i)
    {
        if (vAxis[i] <  0)
        {
            resolvedAxis = inputTensorInfo.GetNumDimensions() + static_cast<uint32_t>(vAxis[i]);
        } else
        {
            resolvedAxis = static_cast<uint32_t>(vAxis[i]);
        }

        updated_idx.push_back(resolvedAxis);
    }

    descriptor.m_Parameters.m_vAxis = updated_idx;
    descriptor.m_Parameters.m_ReduceOperation = reduceOperation;
    descriptor.m_Parameters.m_KeepDims = keepDims;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Reduce,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputTensor.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<float, 4>(actualOutput,
                                     outputData,
                                     outputHandle->GetShape(),
                                     outputTensorInfo.GetShape());
}

} // namespace

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceMaxSimpleTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 1, 2, 3 };
    const armnn::TensorShape outputShape{ 1, 1, 1, 3};

        armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues
    ({
        1001.0f, 11.0f,   1003.0f,
        10.0f,   1002.0f, 12.0f
    });
    std::vector<float> outputValues
    ({
        1001.0f, 1002.0f, 1003.0f
    });

    return ReductionTestCommon<ArmnnType>(workloadFactory,
                                       memoryManager,
                                       tensorHandleFactory,
                                       inputTensorInfo,
                                       outputTensorInfo,
                                       inputValues,
                                       outputValues,
                                       { 2 },
                                       armnn::ReduceOperation::Max);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceMaxNegativeAxisTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 1, 2, 3 };
    const armnn::TensorShape outputShape{ 1, 1, 2, 1};

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues
    ({
         1001.0f, 11.0f,   1003.0f,
         10.0f,   1002.0f, 12.0f
    });
    std::vector<float> outputValues
    ({
        1003.0f, 1002.0f
     });

    return ReductionTestCommon<ArmnnType>(workloadFactory,
                                          memoryManager,
                                          tensorHandleFactory,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          inputValues,
                                          outputValues,
                                          { -1 },
                                          armnn::ReduceOperation::Max,
                                          true);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceMaxSimpleTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 1, 2, 3 };
    const armnn::TensorShape outputShape{ 1, 1, 2, 1 };

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues
    ({
         1.0f, 3.0f, 2.0f,
         6.0f, 4.0f, 5.0f
    });

    std::vector<float> outputValues
    ({
        3.0f, 6.0f
    });

    return ReductionTestCommon<ArmnnType>(workloadFactory,
                                          memoryManager,
                                          tensorHandleFactory,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          inputValues,
                                          outputValues,
                                          { 3 },
                                          armnn::ReduceOperation::Max,
                                          true);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceMinSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape  { 1, 1, 2, 3 };
    const armnn::TensorShape outputShape { 1, 1, 1, 3};

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues
    ({
        1001.0f, 11.0f,   1003.0f,
        10.0f,   1002.0f, 12.0f
    });
    std::vector<float> outputValues
    ({
        10.0f, 11.0f, 12.0f
    });

    return ReductionTestCommon<ArmnnType>(workloadFactory,
                                          memoryManager,
                                          tensorHandleFactory,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          inputValues,
                                          outputValues,
                                          { 2 },
                                          armnn::ReduceOperation::Min);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<float, 4> ReduceMinNegativeAxisTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputShape{ 1, 1, 2, 3 };
    const armnn::TensorShape outputShape{ 1, 1, 2, 1};

    armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(1.0f);
        inputTensorInfo.SetQuantizationOffset(0);
    }

    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    std::vector<float> inputValues
    ({
         1001.0f, 11.0f,   1003.0f,
         10.0f,   1002.0f, 12.0f
    });
    std::vector<float> outputValues
    ({
        11.0f, 10.0f
     });

    return ReductionTestCommon<ArmnnType>(workloadFactory,
                                          memoryManager,
                                          tensorHandleFactory,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          inputValues,
                                          outputValues,
                                          { -1 },
                                          armnn::ReduceOperation::Min,
                                          true);
}

// Explicit template specializations
template LayerTestResult<float, 4>
ReduceMaxSimpleTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<float, 4>
ReduceMaxNegativeAxisTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<float, 4>
ReduceMaxSimpleTest2<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<float, 4>
ReduceMinSimpleTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<float, 4>
ReduceMinNegativeAxisTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

