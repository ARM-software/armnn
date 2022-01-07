//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace
{

template<armnn::DataType ArmnnType, typename T, std::size_t InputDim, std::size_t OutputDim>
LayerTestResult<T, OutputDim> MeanTestHelper(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const unsigned int* inputShape,
        const std::vector<float>& inputData,
        const std::vector<unsigned int>& axis,
        bool keepDims,
        const unsigned int* outputShape,
        const std::vector<float>& outputData,
        float scale = 1.0f,
        int32_t offset = 0)
{
    IgnoreUnused(memoryManager);

    armnn::TensorInfo inputTensorInfo(InputDim, inputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(OutputDim, outputShape, ArmnnType);

    inputTensorInfo.SetQuantizationScale(scale);
    inputTensorInfo.SetQuantizationOffset(offset);

    outputTensorInfo.SetQuantizationScale(scale);
    outputTensorInfo.SetQuantizationOffset(offset);

    auto input = ConvertToDataType<ArmnnType>(inputData, inputTensorInfo);

    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());
    std::vector<T> expectedOutput = ConvertToDataType<ArmnnType>(outputData, outputTensorInfo);

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::MeanQueueDescriptor data;
    data.m_Parameters.m_Axis = axis;
    data.m_Parameters.m_KeepDims = keepDims;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Mean, data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, OutputDim>(actualOutput,
                                         expectedOutput,
                                         outputHandle->GetShape(),
                                         outputTensorInfo.GetShape());
}

} // anonymous namespace

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 1> MeanSimpleTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 3, 2 };
    const unsigned int outputShape[] = { 1 };

    std::vector<float> input({ 1.5f, 1.5f, 2.5f, 2.5f, 3.5f, 3.5f });
    std::vector<float> output({ 2.5f });

    return MeanTestHelper<ArmnnType, T, 2, 1>(
            workloadFactory, memoryManager, tensorHandleFactory, inputShape, input, {}, false, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> MeanSimpleAxisTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 2, 3, 1, 2 };
    const unsigned int outputShape[] = { 3, 1, 2 };

    std::vector<float> input({ 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f });
    std::vector<float> output({ 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f });

    return MeanTestHelper<ArmnnType, T, 4, 3>(
            workloadFactory, memoryManager, tensorHandleFactory, inputShape, input, { 0 }, false, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> MeanKeepDimsTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 1, 1, 3, 2 };
    const unsigned int outputShape[] = { 1, 1, 1, 2 };

    std::vector<float> input({ 1.5f, 1.5f, 2.5f, 2.5f, 3.5f, 3.5f });
    std::vector<float> output({ 2.5f, 2.5f });

    return MeanTestHelper<ArmnnType, T, 4, 4>(
            workloadFactory, memoryManager, tensorHandleFactory, inputShape, input, { 2 }, true, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> MeanMultipleDimsTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 2, 3, 1, 2 };
    const unsigned int outputShape[] = { 1, 3, 1, 1 };

    std::vector<float> input({ 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5 });
    std::vector<float> output({ 2.0f, 4.0f, 6.0f });

    return MeanTestHelper<ArmnnType, T, 4, 4>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputShape, input, { 0, 3 }, true, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 1> MeanVts1Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 4, 3, 2 };
    const unsigned int outputShape[] = { 2 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                               15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f });
    std::vector<float> output({ 12.0f, 13.0f });

    return MeanTestHelper<ArmnnType, T, 3, 1>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputShape, input, { 0, 1 }, false, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> MeanVts2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 4, 3, 2 };
    const unsigned int outputShape[] = { 1, 3, 1 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f,
                               15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f });
    std::vector<float> output({ 10.5f, 12.5f, 14.5f });

    return MeanTestHelper<ArmnnType, T, 3, 3>(
            workloadFactory, memoryManager, tensorHandleFactory,
            inputShape, input, { 0, 2 }, true, outputShape, output);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> MeanVts3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const unsigned int inputShape[] = { 1, 2, 2, 1 };
    const unsigned int outputShape[] = { 1, 2, 1 };

    std::vector<float> input({ 1.0f, 2.0f, 3.0f, 4.0f });
    std::vector<float> output({ 1.5f, 3.5f });

    return MeanTestHelper<ArmnnType, T, 4, 3>(
            workloadFactory, memoryManager, tensorHandleFactory, inputShape, input, { 2 }, false, outputShape, output);
}
