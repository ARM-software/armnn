//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherTestImpl.hpp"

#include <ResolveType.hpp>


#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

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
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::TensorInfo& paramsInfo,
    const armnn::TensorInfo& indicesInfo,
    const armnn::TensorInfo& outputInfo,
    const std::vector<T>& paramsData,
    const std::vector<int32_t>& indicesData,
    const std::vector<T>& outputData)
{
    IgnoreUnused(memoryManager);

    std::vector<T> actualOutput(outputInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> paramsHandle = tensorHandleFactory.CreateTensorHandle(paramsInfo);
    std::unique_ptr<armnn::ITensorHandle> indicesHandle = tensorHandleFactory.CreateTensorHandle(indicesInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    armnn::GatherQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data,  info, paramsInfo, paramsHandle.get());
    AddInputToWorkload(data, info, indicesInfo, indicesHandle.get());
    AddOutputToWorkload(data, info, outputInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Gather, data, info);

    paramsHandle->Allocate();
    indicesHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(paramsHandle.get(), paramsData.data());
    CopyDataToITensorHandle(indicesHandle.get(), indicesData.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, OutputDim>(actualOutput,
                                         outputData,
                                         outputHandle->GetShape(),
                                         outputInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
struct GatherTestHelper
{
    static LayerTestResult<T, 1> Gather1dParamsTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
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
            tensorHandleFactory,
            paramsInfo,
            indicesInfo,
            outputInfo,
            params,
            indices,
            expectedOutput);
    }

    static LayerTestResult<T, 2> GatherMultiDimParamsTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
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
            tensorHandleFactory,
            paramsInfo,
            indicesInfo,
            outputInfo,
            params,
            indices,
            expectedOutput);
    }

    static LayerTestResult<T, 4> GatherMultiDimParamsMultiDimIndicesTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
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
            tensorHandleFactory,
            paramsInfo,
            indicesInfo,
            outputInfo,
            params,
            indices,
            expectedOutput);
    }
};

template<typename T>
struct GatherTestHelper<armnn::DataType::Float16, T>
{
    static LayerTestResult<T, 1> Gather1dParamsTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
    {
        using namespace half_float::literal;

        armnn::TensorInfo paramsInfo({ 8 }, armnn::DataType::Float16);
        armnn::TensorInfo indicesInfo({ 4 }, armnn::DataType::Signed32);
        armnn::TensorInfo outputInfo({ 4 }, armnn::DataType::Float16);

        const std::vector<T> params = std::vector<T>({ 1._h, 2._h, 3._h, 4._h, 5._h, 6._h, 7._h, 8._h });
        const std::vector<int32_t> indices  = std::vector<int32_t>({ 0, 2, 1, 5 });
        const std::vector<T> expectedOutput = std::vector<T>({ 1._h, 3._h, 2._h, 6._h });

        return GatherTestImpl<armnn::DataType::Float16, T, 1, 1, 1>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            paramsInfo,
            indicesInfo,
            outputInfo,
            params,
            indices,
            expectedOutput);
    }

    static LayerTestResult<T, 2> GatherMultiDimParamsTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
    {
        using namespace half_float::literal;

        armnn::TensorInfo paramsInfo({ 5, 2 }, armnn::DataType::Float16);
        armnn::TensorInfo indicesInfo({ 3 }, armnn::DataType::Signed32);
        armnn::TensorInfo outputInfo({ 3, 2 }, armnn::DataType::Float16);

        const std::vector<T> params = std::vector<T>({ 1._h, 2._h, 3._h, 4._h, 5._h, 6._h, 7._h, 8._h, 9._h, 10._h });

        const std::vector<int32_t> indices  = std::vector<int32_t>({ 1, 3, 4 });
        const std::vector<T> expectedOutput = std::vector<T>({ 3._h, 4._h, 7._h, 8._h, 9._h, 10._h });

        return GatherTestImpl<armnn::DataType::Float16, T, 2, 1, 2>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            paramsInfo,
            indicesInfo,
            outputInfo,
            params,
            indices,
            expectedOutput);
    }

    static LayerTestResult<T, 4> GatherMultiDimParamsMultiDimIndicesTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
    {
        using namespace half_float::literal;

        armnn::TensorInfo paramsInfo({ 3, 2, 3 }, armnn::DataType::Float16);
        armnn::TensorInfo indicesInfo({ 2, 3 }, armnn::DataType::Signed32);
        armnn::TensorInfo outputInfo({ 2, 3, 2, 3 }, armnn::DataType::Float16);

        const std::vector<T> params =
        {
            1._h,  2._h,  3._h,
            4._h,  5._h,  6._h,

            7._h,  8._h,  9._h,
            10._h, 11._h, 12._h,

            13._h, 14._h, 15._h,
            16._h, 17._h, 18._h
        };

        const std::vector<int32_t> indices = { 1, 2, 1, 2, 1, 0 };

        const std::vector<T> expectedOutput =
        {
            7._h,  8._h,  9._h,
            10._h, 11._h, 12._h,
            13._h, 14._h, 15._h,
            16._h, 17._h, 18._h,
            7._h,  8._h,  9._h,
            10._h, 11._h, 12._h,

            13._h, 14._h, 15._h,
            16._h, 17._h, 18._h,
            7._h,  8._h,  9._h,
            10._h, 11._h, 12._h,
            1._h,  2._h,  3._h,
            4._h,  5._h,  6._h
        };

        return GatherTestImpl<armnn::DataType::Float16, T, 3, 2, 4>(
            workloadFactory,
            memoryManager,
            tensorHandleFactory,
            paramsInfo,
            indicesInfo,
            outputInfo,
            params,
            indices,
            expectedOutput);
    }
};

} // anonymous namespace

LayerTestResult<float, 1> Gather1dParamsFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::Float32>::Gather1dParamsTestImpl(
            workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<armnn::Half, 1> Gather1dParamsFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::Float16>::Gather1dParamsTestImpl(
            workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<uint8_t, 1> Gather1dParamsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::QAsymmU8>::Gather1dParamsTestImpl(
            workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 1> Gather1dParamsInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::QSymmS16>::Gather1dParamsTestImpl(
            workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int32_t, 1> Gather1dParamsInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::Signed32>::Gather1dParamsTestImpl(
            workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<float, 2> GatherMultiDimParamsFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::Float32>::GatherMultiDimParamsTestImpl(
            workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<armnn::Half, 2> GatherMultiDimParamsFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::Float16>::GatherMultiDimParamsTestImpl(
            workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<uint8_t, 2> GatherMultiDimParamsUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::QAsymmU8>::GatherMultiDimParamsTestImpl(
        workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 2> GatherMultiDimParamsInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::QSymmS16>::GatherMultiDimParamsTestImpl(
        workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int32_t, 2> GatherMultiDimParamsInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::Signed32>::GatherMultiDimParamsTestImpl(
            workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<float, 4> GatherMultiDimParamsMultiDimIndicesFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::Float32>::GatherMultiDimParamsMultiDimIndicesTestImpl(
        workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<armnn::Half, 4> GatherMultiDimParamsMultiDimIndicesFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::Float16>::GatherMultiDimParamsMultiDimIndicesTestImpl(
        workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> GatherMultiDimParamsMultiDimIndicesUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::QAsymmU8>::GatherMultiDimParamsMultiDimIndicesTestImpl(
        workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int16_t, 4> GatherMultiDimParamsMultiDimIndicesInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::QSymmS16>::GatherMultiDimParamsMultiDimIndicesTestImpl(
        workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<int32_t, 4> GatherMultiDimParamsMultiDimIndicesInt32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return GatherTestHelper<armnn::DataType::Signed32>::GatherMultiDimParamsMultiDimIndicesTestImpl(
            workloadFactory, memoryManager, tensorHandleFactory);
}