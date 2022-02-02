//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SliceTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>


#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<typename T, std::size_t NumDims>
LayerTestResult<T, NumDims> SliceTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    armnn::TensorInfo& inputInfo,
    armnn::TensorInfo& outputInfo,
    std::vector<float>& inputData,
    std::vector<float>& expectedOutputData,
    armnn::SliceQueueDescriptor descriptor,
    const float qScale = 1.0f,
    const int qOffset = 0)
{
    IgnoreUnused(memoryManager);
    if(armnn::IsQuantizedType<T>())
    {
        inputInfo.SetQuantizationScale(qScale);
        inputInfo.SetQuantizationOffset(qOffset);

        outputInfo.SetQuantizationScale(qScale);
        outputInfo.SetQuantizationOffset(qOffset);
    }

    std::vector<T> input = armnnUtils::QuantizedVector<T>(inputData, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(expectedOutputData, qScale, qOffset);
    std::vector<T> actualOutput(outputInfo.GetNumElements());

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    std::unique_ptr<armnn::ITensorHandle> inputHandle  = workloadFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputInfo);
    ARMNN_NO_DEPRECATE_WARN_END

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Slice,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, NumDims>(actualOutput,
                                       expectedOutput,
                                       outputHandle->GetShape(),
                                       outputInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> Slice4dTest(armnn::IWorkloadFactory& workloadFactory,
                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorShape inputShape  = { 3, 2, 3, 5 };
    armnn::TensorShape outputShape = { 2, 1, 2, 3 };

    armnn::SliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin  = { 1, 0, 1, 2 };
    desc.m_Parameters.m_Size   = { 2, 1, 2, 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<float> input =
    {
         0.f,  1.f,  2.f,  3.f,  4.f,
         5.f,  6.f,  7.f,  8.f,  9.f,
        10.f, 11.f, 12.f, 13.f, 14.f,

        15.f, 16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f, 24.f,
        25.f, 26.f, 27.f, 28.f, 29.f,


        30.f, 31.f, 32.f, 33.f, 34.f,
        35.f, 36.f, 37.f, 38.f, 39.f,
        40.f, 41.f, 42.f, 43.f, 44.f,

        45.f, 46.f, 47.f, 48.f, 49.f,
        50.f, 51.f, 52.f, 53.f, 54.f,
        55.f, 56.f, 57.f, 58.f, 59.f,


        60.f, 61.f, 62.f, 63.f, 64.f,
        65.f, 66.f, 67.f, 68.f, 69.f,
        70.f, 71.f, 72.f, 73.f, 74.f,

        75.f, 76.f, 77.f, 78.f, 79.f,
        80.f, 81.f, 82.f, 83.f, 84.f,
        85.f, 86.f, 87.f, 88.f, 89.f
    };

    std::vector<float> expectedOutput =
    {
        37.f, 38.f, 39.f,
        42.f, 43.f, 44.f,


        67.f, 68.f, 69.f,
        72.f, 73.f, 74.f
    };

    return SliceTestImpl<T, 4>(workloadFactory, memoryManager, inputInfo, outputInfo, input, expectedOutput, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> Slice3dTest(armnn::IWorkloadFactory& workloadFactory,
                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorShape inputShape  = { 2, 3, 5 };
    armnn::TensorShape outputShape = { 1, 2, 3 };

    armnn::SliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin  = { 0, 1, 2 };
    desc.m_Parameters.m_Size   = { 1, 2, 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<float> input =
    {
         0.f,  1.f,  2.f,  3.f,  4.f,
         5.f,  6.f,  7.f,  8.f,  9.f,
        10.f, 11.f, 12.f, 13.f, 14.f,

        15.f, 16.f, 17.f, 18.f, 19.f,
        20.f, 21.f, 22.f, 23.f, 24.f,
        25.f, 26.f, 27.f, 28.f, 29.f,
    };

    std::vector<float> expectedOutput =
    {
         7.f,  8.f,  9.f,
        12.f, 13.f, 14.f
    };

    return SliceTestImpl<T, 3>(workloadFactory, memoryManager, inputInfo, outputInfo, input, expectedOutput, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> Slice2dTest(armnn::IWorkloadFactory& workloadFactory,
                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorShape inputShape  = { 3, 5 };
    armnn::TensorShape outputShape = { 2, 3 };

    armnn::SliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin  = { 1, 2 };
    desc.m_Parameters.m_Size   = { 2, 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<float> input =
    {
         0.f,  1.f,  2.f,  3.f,  4.f,
         5.f,  6.f,  7.f,  8.f,  9.f,
        10.f, 11.f, 12.f, 13.f, 14.f
    };

    std::vector<float> expectedOutput =
    {
         7.f,  8.f,  9.f,
        12.f, 13.f, 14.f
    };

    return SliceTestImpl<T, 2>(workloadFactory, memoryManager, inputInfo, outputInfo, input, expectedOutput, desc);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 1> Slice1dTest(armnn::IWorkloadFactory& workloadFactory,
                                  const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    armnn::TensorShape inputShape  = { 5 };
    armnn::TensorShape outputShape = { 3 };

    armnn::SliceQueueDescriptor desc;
    desc.m_Parameters.m_Begin  = { 2 };
    desc.m_Parameters.m_Size   = { 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<float> input =
    {
         0.f,  1.f,  2.f,  3.f,  4.f
    };

    std::vector<float> expectedOutput =
    {
         2.f,  3.f,  4.f
    };

    return SliceTestImpl<T, 1>(workloadFactory, memoryManager, inputInfo, outputInfo, input, expectedOutput, desc);
}

} // anonymous namespace

// Float32 tests
LayerTestResult<float, 4> Slice4dFloat32Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice4dTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 3> Slice3dFloat32Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice3dTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 2> Slice2dFloat32Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice2dTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

LayerTestResult<float, 1> Slice1dFloat32Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice1dTest<armnn::DataType::Float32>(workloadFactory, memoryManager);
}

// Uint8 tests
LayerTestResult<uint8_t, 4> Slice4dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice4dTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 3> Slice3dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice3dTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 2> Slice2dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice2dTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager);
}

LayerTestResult<uint8_t, 1> Slice1dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice1dTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager);
}

// Int16 tests
LayerTestResult<int16_t, 4> Slice4dInt16Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice4dTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 3> Slice3dInt16Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice3dTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 2> Slice2dInt16Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice2dTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager);
}

LayerTestResult<int16_t, 1> Slice1dInt16Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager)
{
    return Slice1dTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager);
}
