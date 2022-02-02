//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceToDepthTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>


#include <armnnUtils/Permute.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<typename T>
LayerTestResult<T, 4> SpaceToDepthTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::TensorInfo& inputTensorInfo,
    armnn::TensorInfo& outputTensorInfo,
    std::vector<float>& inputData,
    std::vector<float>& outputExpectedData,
    armnn::SpaceToDepthQueueDescriptor descriptor,
    const float qScale = 1.0f,
    const int32_t qOffset = 0)
{
    IgnoreUnused(memoryManager);
    const armnn::PermutationVector NHWCToNCHW = {0, 2, 3, 1};

    if (descriptor.m_Parameters.m_DataLayout == armnn::DataLayout::NCHW)
    {
        inputTensorInfo = armnnUtils::Permuted(inputTensorInfo, NHWCToNCHW);
        outputTensorInfo = armnnUtils::Permuted(outputTensorInfo, NHWCToNCHW);

        std::vector<float> inputTmp(inputData.size());
        armnnUtils::Permute(inputTensorInfo.GetShape(), NHWCToNCHW,
            inputData.data(), inputTmp.data(), sizeof(float));
        inputData = inputTmp;

        std::vector<float> outputTmp(outputExpectedData.size());
        armnnUtils::Permute(outputTensorInfo.GetShape(), NHWCToNCHW,
            outputExpectedData.data(), outputTmp.data(), sizeof(float));
        outputExpectedData = outputTmp;
    }

    if(armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(qScale);
        inputTensorInfo.SetQuantizationOffset(qOffset);
        outputTensorInfo.SetQuantizationScale(qScale);
        outputTensorInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(outputExpectedData, qScale, qOffset);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::SpaceToDepth,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SpaceToDepthSimpleTest1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout = armnn::DataLayout::NHWC)
{
    unsigned int inputShape[] = {1, 2, 2, 1};
    unsigned int outputShape[] = {1, 1, 1, 4};

    std::vector<float> input = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        1.0f, 2.0f, 3.0f, 4.0f
    });

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    armnn::SpaceToDepthQueueDescriptor desc;
    desc.m_Parameters.m_DataLayout = dataLayout;
    desc.m_Parameters.m_BlockSize = 2;

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    return SpaceToDepthTestImpl<T>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> SpaceToDepthSimpleTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout = armnn::DataLayout::NHWC)
{
    unsigned int inputShape[]  = {1, 2, 2, 2};
    unsigned int outputShape[] = {1, 1, 1, 8};

    std::vector<float> input = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    std::vector<float> outputExpected = std::vector<float>(
    {
        1.4f, 2.3f, 3.2f, 4.1f, 5.4f, 6.3f, 7.2f, 8.1f
    });

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    armnn::SpaceToDepthQueueDescriptor desc;
    desc.m_Parameters.m_DataLayout = dataLayout;
    desc.m_Parameters.m_BlockSize = 2;

    inputTensorInfo = armnn::TensorInfo(4, inputShape, ArmnnType);
    outputTensorInfo = armnn::TensorInfo(4, outputShape, ArmnnType);

    return SpaceToDepthTestImpl<T>(
        workloadFactory, memoryManager, tensorHandleFactory,
        inputTensorInfo, outputTensorInfo, input, outputExpected, desc);
}

} // anonymous namespace

LayerTestResult<uint8_t, 4> SpaceToDepthNhwcAsymmQ8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory);
}

LayerTestResult<uint8_t, 4> SpaceToDepthNchwAsymmQ8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::QAsymmU8>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        armnn::DataLayout::NCHW);
}

LayerTestResult<armnn::Half, 4> SpaceToDepthNhwcFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory);
}

LayerTestResult<armnn::Half, 4> SpaceToDepthNchwFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::Float16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        armnn::DataLayout::NCHW);
}

LayerTestResult<float, 4> SpaceToDepthNhwcFloat32Test1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory);
}

LayerTestResult<float, 4> SpaceToDepthNchwFloat32Test1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest1<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        armnn::DataLayout::NCHW);
}

LayerTestResult<float, 4> SpaceToDepthNhwcFloat32Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest2<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory);
}

LayerTestResult<float, 4> SpaceToDepthNchwFloat32Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest2<armnn::DataType::Float32>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        armnn::DataLayout::NCHW);
}

LayerTestResult<int16_t, 4> SpaceToDepthNhwcQSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest2<armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory);
}

LayerTestResult<int16_t, 4> SpaceToDepthNchwQSymm16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return SpaceToDepthSimpleTest2<armnn::DataType::QSymmS16>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        armnn::DataLayout::NCHW);
}
