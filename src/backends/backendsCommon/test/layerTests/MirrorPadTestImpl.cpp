//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MirrorPadTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

//
// Implementation templates
//

template<typename T>
LayerTestResult<T, 2> MirrorPad2dTestCommon(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::TensorInfo& inputTensorInfo,
    const armnn::TensorInfo& outputTensorInfo,
    const std::vector<T>& inputValues,
    const std::vector<T>& expectedOutputValues,
    const std::vector<std::pair<unsigned int, unsigned int>>& padList,
    const armnn::PaddingMode paddingMode)
{
    IgnoreUnused(memoryManager);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PadQueueDescriptor descriptor;

    descriptor.m_Parameters.m_PadList = padList;
    descriptor.m_Parameters.m_PaddingMode = paddingMode;
    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Pad,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputValues.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 2>(actualOutput,
                                 expectedOutputValues,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<typename T>
LayerTestResult<T, 3> MirrorPad3dTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::TensorInfo& inputTensorInfo,
        const armnn::TensorInfo& outputTensorInfo,
        const std::vector<T>& inputValues,
        const std::vector<T>& expectedOutputValues,
        const std::vector<std::pair<unsigned int, unsigned int>>& padList,
        const armnn::PaddingMode paddingMode)
{
    IgnoreUnused(memoryManager);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PadQueueDescriptor descriptor;
    descriptor.m_Parameters.m_PadList = padList;
    descriptor.m_Parameters.m_PaddingMode = paddingMode;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Pad,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputValues.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 3>(actualOutput,
                                 expectedOutputValues,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<typename T>
LayerTestResult<T, 4> MirrorPad4dTestCommon(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const armnn::TensorInfo& inputTensorInfo,
        const armnn::TensorInfo& outputTensorInfo,
        const std::vector<T>& inputValues,
        const std::vector<T>& expectedOutputValues,
        const std::vector<std::pair<unsigned int, unsigned int>>& padList,
        const armnn::PaddingMode paddingMode)
{
    IgnoreUnused(memoryManager);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PadQueueDescriptor descriptor;
    descriptor.m_Parameters.m_PadList = padList;
    descriptor.m_Parameters.m_PaddingMode = paddingMode;

    armnn::WorkloadInfo info;
    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::Pad,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputValues.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutputValues,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> PadSymmetric2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 3, 3 };
    const armnn::TensorShape outputShape{ 7, 7 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>(
    {
        // Height (3) x Width (3)
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    },
    qScale, qOffset);

    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>(
    {
        5, 4, 4, 5, 6, 6, 5,
        2, 1, 1, 2, 3, 3, 2,
        2, 1, 1, 2, 3, 3, 2,
        5, 4, 4, 5, 6, 6, 5,
        8, 7, 7, 8, 9, 9, 8,
        8, 7, 7, 8, 9, 9, 8,
        5, 4, 4, 5, 6, 6, 5
    },
    qScale, qOffset);

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));

    return MirrorPad2dTestCommon<T>(workloadFactory,
                                    memoryManager,
                                    tensorHandleFactory,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    inputValues,
                                    expectedOutputValues,
                                    padList,
                                    armnn::PaddingMode::Symmetric);
}

template<armnn::DataType ArmnnType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 2> PadReflect2dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 3, 3 };
    const armnn::TensorShape outputShape{ 7, 7 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>(
    {
        // Height (3) x Width (3)
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    },
    qScale, qOffset);

    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>(
    {
        9, 8, 7, 8, 9, 8, 7,
        6, 5, 4, 5, 6, 5, 4,
        3, 2, 1, 2, 3, 2, 1,
        6, 5, 4, 5, 6, 5, 4,
        9, 8, 7, 8, 9, 8, 7,
        6, 5, 4, 5, 6, 5, 4,
        3, 2, 1, 2, 3, 2, 1
    },
    qScale, qOffset);

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));

    return MirrorPad2dTestCommon<T>(workloadFactory,
                                    memoryManager,
                                    tensorHandleFactory,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    inputValues,
                                    expectedOutputValues,
                                    padList,
                                    armnn::PaddingMode::Reflect);
}

template<armnn::DataType ArmnnType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> PadSymmetric3dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 2, 2, 2 };
    const armnn::TensorShape outputShape{ 4, 4, 4 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>(
    {
        // Channel 0, Height (2) x Width (2)
        1, 2,
        3, 4,

        // Channel 1, Height (2) x Width (2)
        5, 6,
        7, 8
    },
    qScale, qOffset);

    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>(
    {
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4,

        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4,

        5, 5, 6, 6,
        5, 5, 6, 6,
        7, 7, 8, 8,
        7, 7, 8, 8,

        5, 5, 6, 6,
        5, 5, 6, 6,
        7, 7, 8, 8,
        7, 7, 8, 8
    },
    qScale, qOffset);

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));

    return MirrorPad3dTestCommon<T>(workloadFactory,
                                    memoryManager,
                                    tensorHandleFactory,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    inputValues,
                                    expectedOutputValues,
                                    padList,
                                    armnn::PaddingMode::Symmetric);
}

template<armnn::DataType ArmnnType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 3> PadReflect3dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 2, 2, 2 };
    const armnn::TensorShape outputShape{ 4, 4, 4 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>(
    {
        // Channel 0, Height (2) x Width (2)
        1, 2,
        3, 4,

        // Channel 1, Height (2) x Width (2)
        5, 6,
        7, 8
    },
    qScale, qOffset);

    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>(
    {
        8, 7, 8, 7,
        6, 5, 6, 5,
        8, 7, 8, 7,
        6, 5, 6, 5,

        4, 3, 4, 3,
        2, 1, 2, 1,
        4, 3, 4, 3,
        2, 1, 2, 1,

        8, 7, 8, 7,
        6, 5, 6, 5,
        8, 7, 8, 7,
        6, 5, 6, 5,

        4, 3, 4, 3,
        2, 1, 2, 1,
        4, 3, 4, 3,
        2, 1, 2, 1
    },
    qScale, qOffset);

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));

    return MirrorPad3dTestCommon<T>(workloadFactory,
                                    memoryManager,
                                    tensorHandleFactory,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    inputValues,
                                    expectedOutputValues,
                                    padList,
                                    armnn::PaddingMode::Reflect);
}

template<armnn::DataType ArmnnType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> PadSymmetric4dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 2, 2, 2, 2 };
    const armnn::TensorShape outputShape{ 6, 6, 6, 6 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>(
    {
        // Batch 0, Channel 0, Height (2) x Width (2)
        1, 2,
        3, 4,

        // Batch 0, Channel 1, Height (2) x Width (2)
        5, 6,
        7, 8,

        // Batch 1, Channel 0, Height (2) x Width (2)
        9, 10,
        11, 12,

        // Batch 1, Channel 1, Height (2) x Width (2)
        13, 14,
        15, 16,
    },
    qScale, qOffset);

    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>(
    {
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,
        14, 13, 13, 14, 14, 13,
        16, 15, 15, 16, 16, 15,
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,

        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,
        10,  9,  9, 10, 10,  9,
        12, 11, 11, 12, 12, 11,
        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,

        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,
        10,  9,  9, 10, 10,  9,
        12, 11, 11, 12, 12, 11,
        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,

        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,
        14, 13, 13, 14, 14, 13,
        16, 15, 15, 16, 16, 15,
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,

        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,
        14, 13, 13, 14, 14, 13,
        16, 15, 15, 16, 16, 15,
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,

        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,
        10,  9,  9, 10, 10,  9,
        12, 11, 11, 12, 12, 11,
        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,


        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,
        6,  5,  5,  6,  6,  5,
        8,  7,  7,  8,  8,  7,
        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,

        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,
        2,  1,  1,  2,  2,  1,
        4,  3,  3,  4,  4,  3,
        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,

        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,
        2,  1,  1,  2,  2,  1,
        4,  3,  3,  4,  4,  3,
        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,

        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,
        6,  5,  5,  6,  6,  5,
        8,  7,  7,  8,  8,  7,
        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,

        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,
        6,  5,  5,  6,  6,  5,
        8,  7,  7,  8,  8,  7,
        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,

        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,
        2,  1,  1,  2,  2,  1,
        4,  3,  3,  4,  4,  3,
        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,


        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,
        6,  5,  5,  6,  6,  5,
        8,  7,  7,  8,  8,  7,
        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,

        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,
        2,  1,  1,  2,  2,  1,
        4,  3,  3,  4,  4,  3,
        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,

        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,
        2,  1,  1,  2,  2,  1,
        4,  3,  3,  4,  4,  3,
        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,

        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,
        6,  5,  5,  6,  6,  5,
        8,  7,  7,  8,  8,  7,
        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,

        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,
        6,  5,  5,  6,  6,  5,
        8,  7,  7,  8,  8,  7,
        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,

        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,
        2,  1,  1,  2,  2,  1,
        4,  3,  3,  4,  4,  3,
        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,


        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,
        14, 13, 13, 14, 14, 13,
        16, 15, 15, 16, 16, 15,
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,

        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,
        10,  9,  9, 10, 10,  9,
        12, 11, 11, 12, 12, 11,
        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,

        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,
        10,  9,  9, 10, 10,  9,
        12, 11, 11, 12, 12, 11,
        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,

        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,
        14, 13, 13, 14, 14, 13,
        16, 15, 15, 16, 16, 15,
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,

        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,
        14, 13, 13, 14, 14, 13,
        16, 15, 15, 16, 16, 15,
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,

        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,
        10,  9,  9, 10, 10,  9,
        12, 11, 11, 12, 12, 11,
        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,


        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,
        14, 13, 13, 14, 14, 13,
        16, 15, 15, 16, 16, 15,
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,

        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,
        10,  9,  9, 10, 10,  9,
        12, 11, 11, 12, 12, 11,
        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,

        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,
        10,  9,  9, 10, 10,  9,
        12, 11, 11, 12, 12, 11,
        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,

        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,
        14, 13, 13, 14, 14, 13,
        16, 15, 15, 16, 16, 15,
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,

        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,
        14, 13, 13, 14, 14, 13,
        16, 15, 15, 16, 16, 15,
        16, 15, 15, 16, 16, 15,
        14, 13, 13, 14, 14, 13,

        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,
        10,  9,  9, 10, 10,  9,
        12, 11, 11, 12, 12, 11,
        12, 11, 11, 12, 12, 11,
        10,  9,  9, 10, 10,  9,


        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,
        6,  5,  5,  6,  6,  5,
        8,  7,  7,  8,  8,  7,
        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,

        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,
        2,  1,  1,  2,  2,  1,
        4,  3,  3,  4,  4,  3,
        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,

        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,
        2,  1,  1,  2,  2,  1,
        4,  3,  3,  4,  4,  3,
        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,

        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,
        6,  5,  5,  6,  6,  5,
        8,  7,  7,  8,  8,  7,
        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,

        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,
        6,  5,  5,  6,  6,  5,
        8,  7,  7,  8,  8,  7,
        8,  7,  7,  8,  8,  7,
        6,  5,  5,  6,  6,  5,

        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1,
        2,  1,  1,  2,  2,  1,
        4,  3,  3,  4,  4,  3,
        4,  3,  3,  4,  4,  3,
        2,  1,  1,  2,  2,  1
    },
    qScale, qOffset);

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));

    return MirrorPad4dTestCommon<T>(workloadFactory,
                                    memoryManager,
                                    tensorHandleFactory,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    inputValues,
                                    expectedOutputValues,
                                    padList,
                                    armnn::PaddingMode::Symmetric);
}

template<armnn::DataType ArmnnType,
        typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> PadReflect4dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        float qScale,
        int32_t qOffset)
{
    const armnn::TensorShape inputShape{ 2, 2, 2, 2 };
    const armnn::TensorShape outputShape{ 4, 4, 4, 4 };

    const armnn::TensorInfo inputTensorInfo(inputShape, ArmnnType, qScale, qOffset);
    const armnn::TensorInfo outputTensorInfo(outputShape, ArmnnType, qScale, qOffset);

    std::vector<T> inputValues = armnnUtils::QuantizedVector<T>(
    {
        // Batch 0, Channel 0, Height (2) x Width (2)
        1, 2,
        3, 4,

        // Batch 0, Channel 1, Height (2) x Width (2)
        5, 6,
        7, 8,

        // Batch 1, Channel 0, Height (2) x Width (2)
        9, 10,
        11, 12,

        // Batch 1, Channel 1, Height (2) x Width (2)
        13, 14,
        15, 16,
    },
    qScale, qOffset);

    std::vector<T> expectedOutputValues = armnnUtils::QuantizedVector<T>(
    {
        16, 15, 16, 15,
        14, 13, 14, 13,
        16, 15, 16, 15,
        14, 13, 14, 13,

        12, 11, 12, 11,
        10,  9, 10,  9,
        12, 11, 12, 11,
        10,  9, 10,  9,

        16, 15, 16, 15,
        14, 13, 14, 13,
        16, 15, 16, 15,
        14, 13, 14, 13,

        12, 11, 12, 11,
        10,  9, 10,  9,
        12, 11, 12, 11,
        10,  9, 10,  9,


        8,  7,  8,  7,
        6,  5,  6,  5,
        8,  7,  8,  7,
        6,  5,  6,  5,

        4,  3,  4,  3,
        2,  1,  2,  1,
        4,  3,  4,  3,
        2,  1,  2,  1,

        8,  7,  8,  7,
        6,  5,  6,  5,
        8,  7,  8,  7,
        6,  5,  6,  5,

        4,  3,  4,  3,
        2,  1,  2,  1,
        4,  3,  4,  3,
        2,  1,  2,  1,


        16, 15, 16, 15,
        14, 13, 14, 13,
        16, 15, 16, 15,
        14, 13, 14, 13,

        12, 11, 12, 11,
        10,  9, 10,  9,
        12, 11, 12, 11,
        10,  9, 10,  9,

        16, 15, 16, 15,
        14, 13, 14, 13,
        16, 15, 16, 15,
        14, 13, 14, 13,

        12, 11, 12, 11,
        10,  9, 10,  9,
        12, 11, 12, 11,
        10,  9, 10,  9,


        8,  7,  8,  7,
        6,  5,  6,  5,
        8,  7,  8,  7,
        6,  5,  6,  5,

        4,  3,  4,  3,
        2,  1,  2,  1,
        4,  3,  4,  3,
        2,  1,  2,  1,

        8,  7,  8,  7,
        6,  5,  6,  5,
        8,  7,  8,  7,
        6,  5,  6,  5,

        4,  3,  4,  3,
        2,  1,  2,  1,
        4,  3,  4,  3,
        2,  1,  2,  1
    },
    qScale, qOffset);

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));

    return MirrorPad4dTestCommon<T>(workloadFactory,
                                    memoryManager,
                                    tensorHandleFactory,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    inputValues,
                                    expectedOutputValues,
                                    padList,
                                    armnn::PaddingMode::Reflect);
}

LayerTestResult<armnn::Half, 2> PadSymmetricFloat16(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    const armnn::TensorShape inputShape{ 3, 3 };
    const armnn::TensorShape outputShape{ 5, 7 };

    const armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float16);
    const armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float16);

    const std::vector<armnn::Half> inputValues =
    {
        1._h,  2._h,  3._h,
        4._h,  5._h,  6._h,
        7._h,  8._h,  9._h
    };

    std::vector<armnn::Half> expectedOutputValues =
    {
        2._h, 1._h, 1._h, 2._h, 3._h, 3._h, 2._h,
        2._h, 1._h, 1._h, 2._h, 3._h, 3._h, 2._h,
        5._h, 4._h, 4._h, 5._h, 6._h, 6._h, 5._h,
        8._h, 7._h, 7._h, 8._h, 9._h, 9._h, 8._h,
        8._h, 7._h, 7._h, 8._h, 9._h, 9._h, 8._h,
    };

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));

    return MirrorPad2dTestCommon<armnn::Half>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              inputTensorInfo,
                                              outputTensorInfo,
                                              inputValues,
                                              expectedOutputValues,
                                              padList,
                                              armnn::PaddingMode::Symmetric);
}

LayerTestResult<armnn::Half, 2> PadReflectFloat16(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    using namespace half_float::literal;

    const armnn::TensorShape inputShape{ 3, 3 };
    const armnn::TensorShape outputShape{ 7, 5 };

    const armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float16);
    const armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float16);

    const std::vector<armnn::Half> inputValues =
    {
        1._h,  2._h,  3._h,
        4._h,  5._h,  6._h,
        7._h,  8._h,  9._h
    };

    std::vector<armnn::Half> expectedOutputValues =
    {
        8._h, 7._h, 8._h, 9._h, 8._h,
        5._h, 4._h, 5._h, 6._h, 5._h,
        2._h, 1._h, 2._h, 3._h, 2._h,
        5._h, 4._h, 5._h, 6._h, 5._h,
        8._h, 7._h, 8._h, 9._h, 8._h,
        5._h, 4._h, 5._h, 6._h, 5._h,
        2._h, 1._h, 2._h, 3._h, 2._h,
    };

    std::vector<std::pair<unsigned int, unsigned int>> padList;
    padList.push_back(std::pair<unsigned int, unsigned int>(2,2));
    padList.push_back(std::pair<unsigned int, unsigned int>(1,1));

    return MirrorPad2dTestCommon<armnn::Half>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              inputTensorInfo,
                                              outputTensorInfo,
                                              inputValues,
                                              expectedOutputValues,
                                              padList,
                                              armnn::PaddingMode::Reflect);
}

//
// Implementation functions
//

LayerTestResult<float, 2> PadSymmetric2dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetric2dTest<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<float, 2> PadReflect2dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflect2dTest<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<float, 3> PadSymmetric3dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetric3dTest<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<float, 3> PadReflect3dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflect3dTest<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<uint8_t, 3> PadSymmetric3dUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetric3dTest<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 128);
}

LayerTestResult<uint8_t, 3> PadReflect3dUint8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflect3dTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 128);
}

LayerTestResult<int8_t, 3> PadSymmetric3dInt8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetric3dTest<armnn::DataType::QAsymmS8>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 64);
}

LayerTestResult<int8_t, 3> PadReflect3dInt8Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflect3dTest<armnn::DataType::QAsymmS8>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 64);
}

LayerTestResult<float, 4> PadSymmetric4dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetric4dTest<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<float, 4> PadReflect4dFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflect4dTest<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<armnn::BFloat16, 4> PadSymmetric4dBFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetric4dTest<armnn::DataType::BFloat16>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<armnn::BFloat16, 4> PadReflect4dBFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflect4dTest<armnn::DataType::BFloat16>(workloadFactory, memoryManager, tensorHandleFactory, 0.0f, 0);
}

LayerTestResult<uint8_t, 4> PadSymmetric4dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetric4dTest<armnn::DataType::QAsymmU8>(
            workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 128);
}

LayerTestResult<uint8_t, 4> PadReflect4dUint8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflect4dTest<armnn::DataType::QAsymmU8>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 128);
}

LayerTestResult<int8_t, 4> PadSymmetric4dInt8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetric4dTest<armnn::DataType::QAsymmS8>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 64);
}

LayerTestResult<int8_t, 4> PadReflect4dInt8Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflect4dTest<armnn::DataType::QAsymmS8>(workloadFactory, memoryManager, tensorHandleFactory, 0.1f, 64);
}

LayerTestResult<int16_t, 4> PadSymmetric4dInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetric4dTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 2.0f, 0);
}

LayerTestResult<int16_t, 4> PadReflect4dInt16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflect4dTest<armnn::DataType::QSymmS16>(workloadFactory, memoryManager, tensorHandleFactory, 2.0f, 0);
}

LayerTestResult<armnn::Half, 2> PadSymmetricFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadSymmetricFloat16(workloadFactory, memoryManager, tensorHandleFactory);
}

LayerTestResult<armnn::Half, 2> PadReflectFloat16Test(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    return PadReflectFloat16(workloadFactory, memoryManager, tensorHandleFactory);
}
