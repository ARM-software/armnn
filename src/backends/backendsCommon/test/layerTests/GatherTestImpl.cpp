//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherTestImpl.hpp"

#include <ResolveType.hpp>

#include <armnn/ArmNN.hpp>

#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

namespace
{

template <armnn::DataType ArmnnType,
          typename T = armnn::ResolveType<ArmnnType>,
          size_t ParamsDim,
          size_t IndicesDim,
          size_t OutputDim>
LayerTestResult<T, OutputDim> GatherTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::TensorInfo& paramsInfo,
    const armnn::TensorInfo& indicesInfo,
    const armnn::TensorInfo& outputInfo,
    const std::vector<T>& paramsData,
    const std::vector<int32_t>& indicesData,
    const std::vector<T>& outputData)
{
    auto params  = MakeTensor<T, ParamsDim>(paramsInfo, paramsData);
    auto indices = MakeTensor<int32_t, IndicesDim>(indicesInfo, indicesData);

    LayerTestResult<T, OutputDim> result(outputInfo);
    result.outputExpected = MakeTensor<T, OutputDim>(outputInfo, outputData);

    std::unique_ptr<armnn::ITensorHandle> paramsHandle = workloadFactory.CreateTensorHandle(paramsInfo);
    std::unique_ptr<armnn::ITensorHandle> indicesHandle = workloadFactory.CreateTensorHandle(indicesInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputInfo);

    armnn::GatherQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, paramsInfo, paramsHandle.get());
    AddInputToWorkload(data, info, indicesInfo, indicesHandle.get());
    AddOutputToWorkload(data, info, outputInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateGather(data, info);

    paramsHandle->Allocate();
    indicesHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(paramsHandle.get(), params.origin());
    CopyDataToITensorHandle(indicesHandle.get(), indices.origin());

    workload->Execute();

    CopyDataFromITensorHandle(result.output.origin(), outputHandle.get());

    return result;
}

template <armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 1> Gather1dParamsTestImpl(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo paramsInfo({ 8 }, ArmnnType);
    armnn::TensorInfo indicesInfo({ 4 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 4 }, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        paramsInfo.SetQuantizationScale(1.0f);
        paramsInfo.SetQuantizationOffset(1);
        outputInfo.SetQuantizationScale(1.0f);
        outputInfo.SetQuantizationOffset(1);
    }
    const std::vector<T> params         = std::vector<T>({ 1, 2, 3, 4, 5, 6, 7, 8 });
    const std::vector<int32_t> indices  = std::vector<int32_t>({ 0, 2, 1, 5 });
    const std::vector<T> expectedOutput = std::vector<T>({ 1, 3, 2, 6 });

    return GatherTestImpl<ArmnnType, T, 1, 1, 1>(
        workloadFactory,
        memoryManager,
        paramsInfo,
        indicesInfo,
        outputInfo,
        params,
        indices,
        expectedOutput);
}

template <armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> GatherMultiDimParamsTestImpl(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo paramsInfo({ 5, 2 }, ArmnnType);
    armnn::TensorInfo indicesInfo({ 3 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 3, 2 }, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        paramsInfo.SetQuantizationScale(1.0f);
        paramsInfo.SetQuantizationOffset(1);
        outputInfo.SetQuantizationScale(1.0f);
        outputInfo.SetQuantizationOffset(1);
    }

    const std::vector<T> params         = std::vector<T>({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
    const std::vector<int32_t> indices  = std::vector<int32_t>({ 1, 3, 4 });
    const std::vector<T> expectedOutput = std::vector<T>({ 3, 4, 7, 8, 9, 10 });

    return GatherTestImpl<ArmnnType, T, 2, 1, 2>(
        workloadFactory,
        memoryManager,
        paramsInfo,
        indicesInfo,
        outputInfo,
        params,
        indices,
        expectedOutput);
}

template <armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> GatherMultiDimParamsMultiDimIndicesTestImpl(
    armnn::IWorkloadFactory& workloadFactory, const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorInfo paramsInfo({ 3, 2, 3}, ArmnnType);
    armnn::TensorInfo indicesInfo({ 2, 3 }, armnn::DataType::Signed32);
    armnn::TensorInfo outputInfo({ 2, 3, 2, 3 }, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        paramsInfo.SetQuantizationScale(1.0f);
        paramsInfo.SetQuantizationOffset(1);
        outputInfo.SetQuantizationScale(1.0f);
        outputInfo.SetQuantizationOffset(1);
    }

    const std::vector<T> params =
    {
         1,  2,  3,
         4,  5,  6,

         7,  8,  9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18
    };

    const std::vector<int32_t> indices = { 1, 2, 1, 2, 1, 0 };

    const std::vector<T> expectedOutput =
    {
         7,  8,  9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
         7,  8,  9,
        10, 11, 12,

        13, 14, 15,
        16, 17, 18,
         7,  8,  9,
        10, 11, 12,
         1,  2,  3,
         4,  5,  6
    };

    return GatherTestImpl<ArmnnType, T, 3, 2, 4>(
        workloadFactory,
        memoryManager,
        paramsInfo,
        indicesInfo,
        outputInfo,
        params,
        indices,
        expectedOutput);
}

} // anonymous namespace

LayerTestResult<float, 1> Gather1dParamsFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Gather1dParamsTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 1> Gather1dParamsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Gather1dParamsTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 1> Gather1dParamsInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Gather1dParamsTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> GatherMultiDimParamsFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> GatherMultiDimParamsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsTestImpl<armnn::DataType::QuantisedAsymm8>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 2> GatherMultiDimParamsInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsTestImpl<armnn::DataType::QuantisedSymm16>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> GatherMultiDimParamsMultiDimIndicesFloatTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsMultiDimIndicesTestImpl<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 4> GatherMultiDimParamsMultiDimIndicesUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsMultiDimIndicesTestImpl<armnn::DataType::QuantisedAsymm8>(
        workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 4> GatherMultiDimParamsMultiDimIndicesInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return GatherMultiDimParamsMultiDimIndicesTestImpl<armnn::DataType::QuantisedSymm16>(
        workloadFactory, memoryManager);
}
