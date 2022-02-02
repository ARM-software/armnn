//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DepthToSpaceTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>


#include <armnnTestUtils/DataLayoutUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<typename T>
LayerTestResult<T, 4> DepthToSpaceTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::TensorInfo& inputInfo,
    armnn::TensorInfo& outputInfo,
    std::vector<float>& inputData,
    std::vector<float>& expectedOutputData,
    armnn::DepthToSpaceQueueDescriptor descriptor,
    const float qScale = 1.0f,
    const int32_t qOffset = 0)
{
    IgnoreUnused(memoryManager);
    if (descriptor.m_Parameters.m_DataLayout == armnn::DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw<float>(inputInfo, inputData);
        PermuteTensorNhwcToNchw<float>(outputInfo, expectedOutputData);
    }

    if(armnn::IsQuantizedType<T>())
    {
        inputInfo.SetQuantizationScale(qScale);
        inputInfo.SetQuantizationOffset(qOffset);
        outputInfo.SetQuantizationScale(qScale);
        outputInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);

    std::vector<T> actualOutput(outputInfo.GetNumElements());
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    std::unique_ptr<armnn::ITensorHandle> inputHandle  = workloadFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputInfo);
    ARMNN_NO_DEPRECATE_WARN_END

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::DepthToSpace,
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
                                 outputInfo.GetShape());
}

} // anonymous namespace

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> DepthToSpaceTest1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout)
{
    unsigned int inputShape[]  = { 1, 1, 1, 8 };
    unsigned int outputShape[] = { 1, 2, 2, 2 };

    // in:
    // [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
    //
    // out:
    // [[[[1, 2, 3], [4, 5, 6]],
    //   [[7, 8, 9], [10, 11, 12]]]]

    std::vector<float> input          = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f };
    std::vector<float> expectedOutput = input;

    armnn::DepthToSpaceQueueDescriptor desc;
    desc.m_Parameters.m_DataLayout = dataLayout;
    desc.m_Parameters.m_BlockSize  = 2;

    armnn::TensorInfo inputInfo(4, inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(4, outputShape, ArmnnType);

    return DepthToSpaceTestImpl<T>(workloadFactory, memoryManager, inputInfo, outputInfo, input, expectedOutput, desc);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> DepthToSpaceTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout)
{
    unsigned int inputShape[]  = { 1, 2, 2, 4 };
    unsigned int outputShape[] = { 1, 4, 4, 1 };

    // in:
    // [[[[1, 2, 3, 4],
    //   [5, 6, 7, 8]],
    //  [[9, 10, 11, 12],
    //   [13, 14, 15, 16]]]]
    //
    // out:
    // [[[ [1],   [2],  [5],  [6]],
    //  [ [3],   [4],  [7],  [8]],
    //  [ [9],  [10], [13],  [14]],
    //  [ [11], [12], [15],  [16]]]]

    std::vector<float> input =
    {
        1.f,  2.f,  3.f,  4.f,

        5.f,  6.f,  7.f,  8.f,

        9.f, 10.f, 11.f, 12.f,

        13.f, 14.f, 15.f, 16.f
    };

    std::vector<float> expectedOutput
    {
         1.f,   2.f,   5.f,   6.f,
         3.f,   4.f,   7.f,   8.f,
         9.f,  10.f,  13.f,  14.f,
        11.f,  12.f,  15.f,  16.f
    };

    armnn::DepthToSpaceQueueDescriptor desc;
    desc.m_Parameters.m_DataLayout = dataLayout;
    desc.m_Parameters.m_BlockSize  = 2;

    armnn::TensorInfo inputInfo(4, inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(4, outputShape, ArmnnType);

    return DepthToSpaceTestImpl<T>(workloadFactory, memoryManager, inputInfo, outputInfo, input, expectedOutput, desc);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> DepthToSpaceTest3(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout)
{
    unsigned int inputShape[]  = { 2, 1, 1, 4 };
    unsigned int outputShape[] = { 2, 2, 2, 1 };

    std::vector<float> input =
    {
        1.f, 2.f, 3.f, 4.f, // batch 0
        5.f, 6.f, 7.f, 8.f  // batch 1
    };

    std::vector<float> expectedOutput = input;

    armnn::DepthToSpaceQueueDescriptor desc;
    desc.m_Parameters.m_DataLayout = dataLayout;
    desc.m_Parameters.m_BlockSize  = 2;

    armnn::TensorInfo inputInfo(4, inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(4, outputShape, ArmnnType);

    return DepthToSpaceTestImpl<T>(workloadFactory, memoryManager, inputInfo, outputInfo, input, expectedOutput, desc);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> DepthToSpaceTest4(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout)
{
    unsigned int inputShape[]  = { 2, 2, 2, 4 };
    unsigned int outputShape[] = { 2, 4, 4, 1 };

    std::vector<float> input =
    {
        1.f,  2.f,  3.f,  4.f,

        5.f,  6.f,  7.f,  8.f,

        9.f, 10.f, 11.f, 12.f,

        13.f, 14.f, 15.f, 16.f,


        17.f, 18.f, 19.f, 20.f,

        21.f, 22.f, 23.f, 24.f,

        25.f, 26.f, 27.f, 28.f,

        29.f, 30.f, 31.f, 32.f
    };

    std::vector<float> expectedOutput
    {
         1.f,   2.f,   5.f,   6.f,
         3.f,   4.f,   7.f,   8.f,
         9.f,  10.f,  13.f,  14.f,
        11.f,  12.f,  15.f,  16.f,


        17.f,  18.f,  21.f,  22.f,
        19.f,  20.f,  23.f,  24.f,
        25.f,  26.f,  29.f,  30.f,
        27.f,  28.f,  31.f,  32.f
    };

    armnn::DepthToSpaceQueueDescriptor desc;
    desc.m_Parameters.m_DataLayout = dataLayout;
    desc.m_Parameters.m_BlockSize  = 2;

    armnn::TensorInfo inputInfo(4, inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(4, outputShape, ArmnnType);

    return DepthToSpaceTestImpl<T>(workloadFactory, memoryManager, inputInfo, outputInfo, input, expectedOutput, desc);
}

// Float32
template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthToSpaceTest1<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthToSpaceTest2<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthToSpaceTest3<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
DepthToSpaceTest4<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

// Float16
template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
DepthToSpaceTest1<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
DepthToSpaceTest2<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
DepthToSpaceTest3<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
DepthToSpaceTest4<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

// QuantisedAsymm8
template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
DepthToSpaceTest1<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
DepthToSpaceTest2<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
DepthToSpaceTest3<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
DepthToSpaceTest4<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

// QuantisedAsymmS8
template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
DepthToSpaceTest1<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
DepthToSpaceTest2<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
DepthToSpaceTest3<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
DepthToSpaceTest4<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

// QuantisedSymm16
template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
DepthToSpaceTest1<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
DepthToSpaceTest2<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
DepthToSpaceTest3<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
DepthToSpaceTest4<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::DataLayout dataLayout);
