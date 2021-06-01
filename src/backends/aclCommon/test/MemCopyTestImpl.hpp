//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>

#include <backendsCommon/test/LayerTests.hpp>
#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

namespace
{

template<armnn::DataType dataType, typename T = armnn::ResolveType<dataType>>
LayerTestResult<T, 4> MemCopyTest(armnn::IWorkloadFactory& srcWorkloadFactory,
                                  armnn::IWorkloadFactory& dstWorkloadFactory,
                                  bool withSubtensors)
{
    const std::array<unsigned int, 4> shapeData = { { 1u, 1u, 6u, 5u } };
    const armnn::TensorShape tensorShape(4, shapeData.data());
    const armnn::TensorInfo tensorInfo(tensorShape, dataType);
    std::vector<T> inputData =
    {
         1,  2,  3,  4,  5,
         6,  7,  8,  9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
        26, 27, 28, 29, 30,
    };

    LayerTestResult<T, 4> ret(tensorInfo);
    ret.m_ExpectedData = inputData;

    std::vector<T> actualOutput(tensorInfo.GetNumElements());

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    auto inputTensorHandle = srcWorkloadFactory.CreateTensorHandle(tensorInfo);
    auto outputTensorHandle = dstWorkloadFactory.CreateTensorHandle(tensorInfo);
    ARMNN_NO_DEPRECATE_WARN_END

    AllocateAndCopyDataToITensorHandle(inputTensorHandle.get(), inputData.data());
    outputTensorHandle->Allocate();

    armnn::MemCopyQueueDescriptor memCopyQueueDesc;
    armnn::WorkloadInfo workloadInfo;

    const unsigned int origin[4] = {};

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    auto workloadInput = (withSubtensors && srcWorkloadFactory.SupportsSubTensors())
                         ? srcWorkloadFactory.CreateSubTensorHandle(*inputTensorHandle, tensorShape, origin)
                         : std::move(inputTensorHandle);
    auto workloadOutput = (withSubtensors && dstWorkloadFactory.SupportsSubTensors())
                          ? dstWorkloadFactory.CreateSubTensorHandle(*outputTensorHandle, tensorShape, origin)
                          : std::move(outputTensorHandle);
    ARMNN_NO_DEPRECATE_WARN_END

    AddInputToWorkload(memCopyQueueDesc, workloadInfo, tensorInfo, workloadInput.get());
    AddOutputToWorkload(memCopyQueueDesc, workloadInfo, tensorInfo, workloadOutput.get());

    dstWorkloadFactory.CreateMemCopy(memCopyQueueDesc, workloadInfo)->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), workloadOutput.get());
    ret.m_ActualData = actualOutput;

    return ret;
}

template<typename SrcWorkloadFactory,
         typename DstWorkloadFactory,
         armnn::DataType dataType,
         typename T = armnn::ResolveType<dataType>>
LayerTestResult<T, 4> MemCopyTest(bool withSubtensors)
{
    armnn::IBackendInternal::IMemoryManagerSharedPtr srcMemoryManager =
        WorkloadFactoryHelper<SrcWorkloadFactory>::GetMemoryManager();

    armnn::IBackendInternal::IMemoryManagerSharedPtr dstMemoryManager =
        WorkloadFactoryHelper<DstWorkloadFactory>::GetMemoryManager();

    SrcWorkloadFactory srcWorkloadFactory = WorkloadFactoryHelper<SrcWorkloadFactory>::GetFactory(srcMemoryManager);
    DstWorkloadFactory dstWorkloadFactory = WorkloadFactoryHelper<DstWorkloadFactory>::GetFactory(dstMemoryManager);

    return MemCopyTest<dataType>(srcWorkloadFactory, dstWorkloadFactory, withSubtensors);
}

} // anonymous namespace
