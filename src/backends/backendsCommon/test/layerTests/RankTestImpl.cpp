//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RankTestImpl.hpp"

#include <backendsCommon/test/DataTypeUtils.hpp>
#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

template<typename T, std::size_t n>
LayerTestResult<int32_t, 1> RankTest(
        armnn::TensorInfo inputTensorInfo,
        boost::multi_array<T, n> input,
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);

    const armnn::TensorShape outputShape{armnn::Dimensionality::Scalar};
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Signed32);

    LayerTestResult<int32_t , 1> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<uint32_t, 1>(outputTensorInfo, { n });

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::RankQueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateRank(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.origin());

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0], outputHandle.get());

    return ret;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 1> RankDimSize1Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo({6}, ArmnnType, 1.0f, 0);
    auto input = MakeTensor<T, 1>(inputTensorInfo, ConvertToDataType<ArmnnType>(
            { -37.5f, -15.2f, -8.76f, -2.0f, -1.3f, -0.5f },
            inputTensorInfo));

    return RankTest<T, 1>(inputTensorInfo, input, workloadFactory, memoryManager, tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 1> RankDimSize2Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo({1, 3}, ArmnnType, 1.0f, 0);
    auto input = MakeTensor<T, 2>(inputTensorInfo, ConvertToDataType<ArmnnType>(
            { -37.5f, -15.2f, -8.76f },
            inputTensorInfo));

    return RankTest<T, 2>(inputTensorInfo, input, workloadFactory, memoryManager, tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 1> RankDimSize3Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo({1, 3, 2}, ArmnnType, 1.0f, 0);
    auto input = MakeTensor<T, 3>(inputTensorInfo, ConvertToDataType<ArmnnType>(
            { -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f},
            inputTensorInfo));

    return RankTest<T, 3>(inputTensorInfo, input, workloadFactory, memoryManager, tensorHandleFactory);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<int32_t, 1> RankDimSize4Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    armnn::TensorInfo inputTensorInfo({1, 3, 2, 3}, ArmnnType, 1.0f, 0);
    auto input = MakeTensor<T, 4>(inputTensorInfo, ConvertToDataType<ArmnnType>(
            { -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f, 0.0f,
              1.0f, 0.4f, 0.5f, 1.3f, 1.5f, 2.0f, 8.76f, 15.2f, 37.5f },
            inputTensorInfo));

    return RankTest<T, 4>(inputTensorInfo, input, workloadFactory, memoryManager, tensorHandleFactory);
}

template LayerTestResult<int32_t, 1>
RankDimSize4Test<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize4Test<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize4Test<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize4Test<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize4Test<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize4Test<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize4Test<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize4Test<armnn::DataType::BFloat16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize3Test<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize3Test<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize3Test<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize3Test<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize3Test<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize3Test<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize3Test<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize3Test<armnn::DataType::BFloat16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize2Test<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize2Test<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize2Test<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize2Test<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize2Test<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize2Test<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize2Test<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize2Test<armnn::DataType::BFloat16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize1Test<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize1Test<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize1Test<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize1Test<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize1Test<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize1Test<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize1Test<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<int32_t, 1>
RankDimSize1Test<armnn::DataType::BFloat16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);