//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DequantizeTestImpl.hpp"

#include <ResolveType.hpp>


#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<typename T, std::size_t Dim, typename T1=float>
LayerTestResult<T1, Dim> DequantizeTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::TensorInfo& inputTensorInfo,
        const armnn::TensorInfo& outputTensorInfo,
        const std::vector<T>& inputData,
        const std::vector<T1>& expectedOutputData,
        armnn::DequantizeQueueDescriptor descriptor)
{
    IgnoreUnused(memoryManager);

    std::vector<T1> actualOutput(outputTensorInfo.GetNumElements());

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);
    ARMNN_NO_DEPRECATE_WARN_END

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Dequantize,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T1, Dim>(actualOutput,
                                    expectedOutputData,
                                    outputHandle->GetShape(),
                                    outputTensorInfo.GetShape());
}

template <armnn::DataType ArmnnInputType,
          armnn::DataType ArmnnOutputType=armnn::DataType::Float32,
          typename OutType=armnn::ResolveType<ArmnnOutputType>>
LayerTestResult<OutType, 4> DequantizeSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    using T = armnn::ResolveType<ArmnnInputType>;

    armnn::DequantizeQueueDescriptor desc;

    const armnn::TensorInfo inputTensorInfo({1, 2, 2, 3}, ArmnnInputType, 0.5f, 0);
    const armnn::TensorInfo outputTensorInfo({1, 2, 2, 3}, ArmnnOutputType);

    std::vector<T> inputData = std::vector<T>(
    {
         2,  4,  6,
         8, 10, 12,
        14, 16, 18,
        20, 22, 24,
    });

    std::vector<OutType> expectedOutputData;
    for (OutType i = OutType(1); i <= OutType(12); ++i)
    {
        expectedOutputData.push_back(i);
    }

    return DequantizeTestImpl<T, 4, OutType>(workloadFactory,
                                             memoryManager,
                                             inputTensorInfo,
                                             outputTensorInfo,
                                             inputData,
                                             expectedOutputData,
                                             desc);
}

template <armnn::DataType ArmnnInputType>
LayerTestResult<float, 4> DequantizeOffsetTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    using T = armnn::ResolveType<ArmnnInputType>;

    armnn::DequantizeQueueDescriptor desc;

    const armnn::TensorInfo inputTensorInfo({1, 2, 2, 3}, ArmnnInputType, 0.5f, 1);
    const armnn::TensorInfo outputTensorInfo({1, 2, 2, 3}, armnn::DataType::Float32);

    std::vector<T> inputData = std::vector<T>(
    {
         3,  5,  7,
         9, 11, 13,
        15, 17, 19,
        21, 23, 25,
    });

    std::vector<float> expectedOutputData = std::vector<float>(
    {
        1.0f,   2.0f,  3.0f,
        4.0f,   5.0f,  6.0f,
        7.0f,   8.0f,  9.0f,
        10.0f, 11.0f, 12.0f,
    });

    return DequantizeTestImpl<T, 4>(workloadFactory,
                                    memoryManager,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    inputData,
                                    expectedOutputData,
                                    desc);
}

} // anonymous namespace

LayerTestResult<float, 4> DequantizeSimpleUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeSimpleTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> DequantizeOffsetUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeOffsetTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> DequantizeSimpleAsymmInt8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeSimpleTest<armnn::DataType::QAsymmS8>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> DequantizeOffsetAsymmInt8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeOffsetTest<armnn::DataType::QAsymmS8>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> DequantizeSimpleInt8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeSimpleTest<armnn::DataType::QSymmS8>(workloadFactory, memoryManager);
}

LayerTestResult<float, 4> DequantizeSimpleInt16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeSimpleTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager);
}

LayerTestResult<armnn::Half, 4> DequantizeSimpleUint8ToFp16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeSimpleTest<armnn::DataType::QAsymmU8, armnn::DataType::Float16>(workloadFactory,
                                                                                            memoryManager);
}

LayerTestResult<armnn::Half, 4> DequantizeSimpleInt8ToFp16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeSimpleTest<armnn::DataType::QSymmS8, armnn::DataType::Float16>(workloadFactory, memoryManager);
}

LayerTestResult<armnn::Half, 4> DequantizeSimpleInt16ToFp16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return DequantizeSimpleTest<armnn::DataType::QSymmS16, armnn::DataType::Float16>(workloadFactory,
                                                                                            memoryManager);
}
