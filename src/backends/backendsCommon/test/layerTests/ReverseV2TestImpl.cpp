//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReverseV2TestImpl.hpp"

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/WorkloadTestUtils.hpp>
#include <armnnUtils/QuantizeHelper.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnn/Optional.hpp>
#include <armnn/BackendHelper.hpp>

namespace
{
template<armnn::DataType ArmnnType, typename T, std::size_t NumDims>
LayerTestResult<T, NumDims> ReverseV2TestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const std::vector<T>& input,
    const std::vector<int>& axis,
    const std::vector<T>& outputExpected,
    const armnn::TensorInfo& inputInfo,
    const armnn::TensorInfo& axisInfo,
    const armnn::TensorInfo& outputInfo)
{
    LayerTestResult<T, NumDims> result(outputInfo);
    std::vector<T> outputActual(outputInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> axisHandle = tensorHandleFactory.CreateTensorHandle(axisInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    armnn::ReverseV2QueueDescriptor queueDescriptor;
    armnn::WorkloadInfo workloadInfo;

    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, inputHandle.get());
    AddInputToWorkload(queueDescriptor, workloadInfo, axisInfo, axisHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, outputHandle.get());

    // Don't execute if ReverseV2 is not supported, as an exception will be raised.
    const armnn::BackendId& backend = workloadFactory.GetBackendId();
    std::string reasonIfUnsupported;
    armnn::LayerSupportHandle handle = armnn::GetILayerSupportByBackendId(backend);
    result.m_Supported = handle.IsReverseV2Supported(inputInfo,
                                                     axisInfo,
                                                     outputInfo,
                                                     reasonIfUnsupported);
    if (!result.m_Supported)
    {
        return result;
    }

    auto workload = workloadFactory.CreateWorkload(armnn::LayerType::ReverseV2, queueDescriptor, workloadInfo);

    inputHandle->Allocate();
    axisHandle->Allocate();
    outputHandle->Allocate();

    if (input.data() != nullptr)
    {
        CopyDataToITensorHandle(inputHandle.get(), input.data());
    }
    if (axis.data() != nullptr)
    {
        CopyDataToITensorHandle(axisHandle.get(), axis.data());
    }

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    if (outputActual.data() != nullptr)
    {
        CopyDataFromITensorHandle(outputActual.data(), outputHandle.get());
    }

    return LayerTestResult<T, NumDims>(outputActual,
                                       outputExpected,
                                       outputHandle->GetShape(),
                                       outputInfo.GetShape());

}
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTestEmptyAxis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    // Simple test with no axes set so output is the same as input

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2,
        3, 4
    }, qScale, qOffset);

    std::vector<int> axis = armnnUtils::QuantizedVector<int>({}, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        1, 2,
        3, 4
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTestEmptyTensor(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    // Simple test with empty input tensor

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({0}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({0}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({0}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({}, qScale, qOffset);
    std::vector<int> axis = armnnUtils::QuantizedVector<int>({}, qScale, qOffset);
    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({}, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest1Dim(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({4}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({4}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2,
        3, 4
    }, qScale, qOffset);

    std::vector<int> axis = {0};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        4, 3,
        2, 1
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest2Dim1Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2,
        3, 4
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        2, 1,
        4, 3
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest2Dim2Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2,
        3, 4
    }, qScale, qOffset);

    std::vector<int> axis = {1,0};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        4, 3,
        2, 1
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest3Dim1Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 3, 4}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 3, 4}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        9, 10, 11, 12,
        5, 6, 7, 8,
        1, 2, 3, 4,
        21, 22, 23, 24,
        17, 18, 19, 20,
        13, 14, 15, 16
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest3Dim2Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 3, 4}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 3, 4}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    }, qScale, qOffset);

    std::vector<int> axis = {0, 1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        21, 22, 23, 24,
        17, 18, 19, 20,
        13, 14, 15, 16,
        9, 10, 11, 12,
        5, 6, 7, 8,
        1, 2, 3, 4
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest3Dim3Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 3, 4}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({3}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 3, 4}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    }, qScale, qOffset);

    std::vector<int> axis = {1, 0, 2};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        24, 23, 22, 21,
        20, 19, 18, 17,
        16, 15, 14, 13,
        12, 11, 10, 9,
        8, 7, 6, 5,
        4, 3, 2, 1
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest4Dim1Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 2, 2, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 2, 2, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
        19, 20, 21,
        22, 23, 24
    }, qScale, qOffset);

    std::vector<int> axis = {0};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        13, 14, 15,
        16, 17, 18,
        19, 20, 21,
        22, 23, 24,
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest4Dim2Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 2, 2, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 2, 2, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
        19, 20, 21,
        22, 23, 24
    }, qScale, qOffset);

    std::vector<int> axis = {0, 1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        19, 20, 21,
        22, 23, 24,
        13, 14, 15,
        16, 17, 18,
        7, 8, 9,
        10, 11, 12,
        1, 2, 3,
        4, 5, 6
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest4Dim3Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 2, 2, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({3}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 2, 2, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
        19, 20, 21,
        22, 23, 24
    }, qScale, qOffset);

    std::vector<int> axis = {0, 1, 2};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        22, 23, 24,
        19, 20, 21,
        16, 17, 18,
        13, 14, 15,
        10, 11, 12,
        7, 8, 9,
        4, 5, 6,
        1, 2, 3
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2SimpleTest4Dim4Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 2, 2, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({4}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 2, 2, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,
        19, 20, 21,
        22, 23, 24
    }, qScale, qOffset);

    std::vector<int> axis = {0, 1, 2, 3};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
       24, 23, 22,
       21, 20, 19,
       18, 17, 16,
       15, 14, 13,
       12, 11, 10,
       9, 8, 7,
       6, 5, 4,
       3, 2, 1
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2EvenRowOddColTest2Dim(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3,
        4, 5, 6
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        3, 2, 1,
        6, 5, 4
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2EvenRowOddColTest3Dim(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 3, 1}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 3, 1}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3,
        4, 5, 6
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        3, 2, 1,
        6, 5, 4
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2EvenRowEvenColTest2Dim(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 4}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 4}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3, 4,
        5, 6, 7, 8
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        4, 3, 2, 1,
        8, 7, 6, 5
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2EvenRowEvenColTest3Dim(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 4, 1}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 4, 1}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3, 4,
        5, 6, 7, 8
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        4, 3, 2, 1,
        8, 7, 6, 5
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2OddRowOddColTest2Dim(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        3, 2, 1,
        6, 5, 4,
        9, 8, 7
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2OddRowOddColTest3Dim(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3, 1}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3, 1}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        3, 2, 1,
        6, 5, 4,
        9, 8, 7
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2OddRowEvenColTest2Dim(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 4}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 4}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        4, 3, 2, 1,
        8, 7, 6, 5,
        12, 11, 10, 9
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2OddRowEvenColTest3Dim(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 4, 1}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 4, 1}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    }, qScale, qOffset);

    std::vector<int> axis = {1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        4, 3, 2, 1,
        8, 7, 6, 5,
        12, 11, 10, 9
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2NegAxisTest2Dim1Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 4}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 4}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3, 4,
        5, 6, 7, 8,
    }, qScale, qOffset);

    std::vector<int> axis = {-1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        4, 3, 2, 1,
        8, 7, 6, 5
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ReverseV2NegAxisTest3Dim2Axis(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 4, 1}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo axisInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 4, 1}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({
        1, 2, 3, 4,
        5, 6, 7, 8,
    }, qScale, qOffset);

    std::vector<int> axis = {1, -1};

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        4, 3, 2, 1,
        8, 7, 6, 5
    }, qScale, qOffset);

    return ReverseV2TestImpl<ArmnnType, T, 2>(workloadFactory,
                                              memoryManager,
                                              tensorHandleFactory,
                                              input,
                                              axis,
                                              outputExpected,
                                              inputInfo,
                                              axisInfo,
                                              outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTestEmptyAxis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTestEmptyTensor<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest1Dim<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest2Dim1Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest2Dim2Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest3Dim1Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest3Dim2Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest3Dim3Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest4Dim1Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest4Dim2Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest4Dim3Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2SimpleTest4Dim4Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2EvenRowOddColTest2Dim<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2EvenRowOddColTest3Dim<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2EvenRowEvenColTest2Dim<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2EvenRowEvenColTest3Dim<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2OddRowOddColTest2Dim<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2OddRowOddColTest3Dim<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2OddRowEvenColTest2Dim<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2OddRowEvenColTest3Dim<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2NegAxisTest2Dim1Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ReverseV2NegAxisTest3Dim2Axis<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
ReverseV2SimpleTest2Dim2Axis<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
ReverseV2SimpleTest2Dim2Axis<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
ReverseV2SimpleTest2Dim2Axis<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
ReverseV2SimpleTest2Dim2Axis<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);
