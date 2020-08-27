//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PadTestImpl.hpp"

#include <QuantizeHelper.hpp>

#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

//
// Implementation templates
//

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Pad2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    const float customPaddingValue)
{
    IgnoreUnused(memoryManager);
    const armnn::TensorShape inputShape{ 3, 3 };
    const armnn::TensorShape outputShape{ 7, 7 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>(
        {
            // Height (3) x Width (3)
            4, 8, 6,
            7, 4, 4,
            3, 2, 4
        },
        qScale, qOffset);

    auto p = customPaddingValue;
    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>(
        {
            p, p, p, p, p, p, p,
            p, p, p, p, p, p, p,
            p, p, 4, 8, 6, p, p,
            p, p, 7, 4, 4, p, p,
            p, p, 3, 2, 4, p, p,
            p, p, p, p, p, p, p,
            p, p, p, p, p, p, p
        },
        qScale, qOffset);

    auto inputTensor = MakeTensor<T, 2>(inputTensorInfo, std::vector<T>(inputValues));

    LayerTestResult<T, 2> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 2>(outputTensorInfo, std::vector<T>(expectedOutputValues));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);


    armnn::PadQueueDescriptor descriptor;

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));

    descriptor.m_Parameters.m_PadList = padList;
    descriptor.m_Parameters.m_PadValue = customPaddingValue;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePad(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> Pad3dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset)
{
    IgnoreUnused(memoryManager);
    const armnn::TensorShape inputShape{ 2, 2, 2 };
    const armnn::TensorShape outputShape{ 3, 5, 6 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>(
        {
            // Channel 0, Height (2) x Width (2)
            0, 4,
            2, 5,

            // Channel 1, Height (2) x Width (2)
            6, 1,
            5, 2
        },
        qScale, qOffset);

    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>(
        {
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 4, 0, 0,
            0, 0, 2, 5, 0, 0,
            0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 6, 1, 0, 0,
            0, 0, 5, 2, 0, 0,
            0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0
       },
       qScale, qOffset);

    auto inputTensor = MakeTensor<T, 3>(inputTensorInfo, std::vector<T>(inputValues));

    LayerTestResult<T, 3> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 3>(outputTensorInfo, std::vector<T>(expectedOutputValues));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);


    armnn::PadQueueDescriptor descriptor;

    std::vector<std::pair<unsigned int, unsigned int>> PadList;
    PadList.push_back(std::pair<unsigned int, unsigned int>(0,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(2,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(2,2));

    descriptor.m_Parameters.m_PadList = PadList;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePad(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0], outputHandle.get());

    return result;
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> Pad4dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset)
{
    IgnoreUnused(memoryManager);
    const armnn::TensorShape inputShape{ 2, 2, 3, 2 };
    const armnn::TensorShape outputShape{ 4, 5, 7, 4 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>(
        {
            // Batch 0, Channel 0, Height (3) x Width (2)
             0,  1,
             2,  3,
             4,  5,

            // Batch 0, Channel 1, Height (3) x Width (2)
             6,  7,
             8,  9,
            10, 11,

            // Batch 1, Channel 0, Height (3) x Width (2)
            12, 13,
            14, 15,
            16, 17,

            // Batch 1, Channel 1, Height (3) x Width (2)
            18, 19,
            20, 21,
            22, 23
        },
        qScale, qOffset);

    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>(
        {
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 1, 0,
            0, 2, 3, 0,
            0, 4, 5, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 6, 7, 0,
            0, 8, 9, 0,
            0, 10, 11, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 12, 13, 0,
            0, 14, 15, 0,
            0, 16, 17, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 18, 19, 0,
            0, 20, 21, 0,
            0, 22, 23, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,

            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        },
        qScale, qOffset);

    auto inputTensor = MakeTensor<T, 4>(inputTensorInfo, std::vector<T>(inputValues));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected = MakeTensor<T, 4>(outputTensorInfo, std::vector<T>(expectedOutputValues));

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PadQueueDescriptor descriptor;

    std::vector<std::pair<unsigned int, unsigned int>> PadList;
    PadList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(2,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(3,1));
    PadList.push_back(std::pair<unsigned int, unsigned int>(1,1));

    descriptor.m_Parameters.m_PadList = PadList;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePad(descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &inputTensor[0][0][0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}

//
// Explicit template specializations
//

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
Pad2dTestCommon<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset,
    const float customPaddingValue);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
Pad3dTestCommon<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
Pad4dTestCommon<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    float qScale,
    int32_t qOffset);

//
// Implementation functions
//

LayerTestResult<uint8_t, 2> PadUint82dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad2dTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 1.0f, 0);
}

LayerTestResult<uint8_t, 2> PadUint82dCustomPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad2dTestCommon<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, 1.0f, 0, 1.0f);
}

LayerTestResult<uint8_t, 3> PadUint83dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad3dTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 1.0f, 0);
}

LayerTestResult<uint8_t, 4> PadUint84dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad4dTestCommon<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 1.0f, 0);
}

LayerTestResult<float, 2> PadFloat322dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad2dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<float, 2> PadFloat322dCustomPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad2dTestCommon<armnn::DataType::Float32>(
            workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0, 1.0f);
}

LayerTestResult<float, 3> PadFloat323dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad3dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<float, 4> PadFloat324dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad4dTestCommon<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<armnn::BFloat16, 2> PadBFloat162dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad2dTestCommon<armnn::DataType::BFloat16>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<armnn::BFloat16, 2> PadBFloat162dCustomPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad2dTestCommon<armnn::DataType::BFloat16>(
            workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0, 1.0f);
}

LayerTestResult<armnn::BFloat16, 3> PadBFloat163dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad3dTestCommon<armnn::DataType::BFloat16>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<armnn::BFloat16, 4> PadBFloat164dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad4dTestCommon<armnn::DataType::BFloat16>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<int8_t, 2> PadInt82dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad2dTestCommon<armnn::DataType::QSymmS8>(
            workloadFactory, memoryManager, tensorHandleFactory, 1.0f, 0);
}

LayerTestResult<int8_t, 2> PadInt82dCustomPaddingTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad2dTestCommon<armnn::DataType::QSymmS8>(
            workloadFactory, memoryManager, tensorHandleFactory, 1.0f, 0, 1.0f);
}

LayerTestResult<int8_t, 3> PadInt83dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad3dTestCommon<armnn::DataType::QSymmS8>(workloadFactory, memoryManager, tensorHandleFactory, 1.0f, 0);
}

LayerTestResult<int8_t, 4> PadInt84dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return Pad4dTestCommon<armnn::DataType::QSymmS8>(workloadFactory, memoryManager, tensorHandleFactory, 1.0f, 0);
}
