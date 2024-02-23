//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ScatterNdTestImpl.hpp"

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
template<armnn::DataType ArmnnType, typename T, typename TInput, std::size_t NumDims>
LayerTestResult<T, NumDims> ScatterNdTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        const std::vector<TInput>& input,
        const std::vector<int>& indices,
        const std::vector<T>& updates,
        const std::vector<T>& outputExpected,
        const armnn::TensorInfo& inputInfo,
        const armnn::TensorInfo& indicesInfo,
        const armnn::TensorInfo& updatesInfo,
        const armnn::TensorInfo& outputInfo,
        const armnn::ScatterNdDescriptor &descriptor)
{
    LayerTestResult<T, NumDims> result(outputInfo);
    std::vector<T> outputActual(outputInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> indicesHandle = tensorHandleFactory.CreateTensorHandle(indicesInfo);
    std::unique_ptr<armnn::ITensorHandle> updatesHandle = tensorHandleFactory.CreateTensorHandle(updatesInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    armnn::ScatterNdQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;

    armnn::WorkloadInfo workloadInfo;

    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, inputHandle.get());
    AddInputToWorkload(queueDescriptor, workloadInfo, indicesInfo, indicesHandle.get());
    AddInputToWorkload(queueDescriptor, workloadInfo, updatesInfo, updatesHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, outputHandle.get());

    // Don't execute if ScatterNd is not supported, as an exception will be raised.
    const armnn::BackendId& backend = workloadFactory.GetBackendId();
    std::string reasonIfUnsupported;
    armnn::LayerSupportHandle handle = armnn::GetILayerSupportByBackendId(backend);
    result.m_Supported = handle.IsScatterNdSupported(inputInfo,
                                                     indicesInfo,
                                                     updatesInfo,
                                                     outputInfo,
                                                     descriptor,
                                                     reasonIfUnsupported);
    if (!result.m_Supported)
    {
        return result;
    }

    auto workload = workloadFactory.CreateWorkload(armnn::LayerType::ScatterNd, queueDescriptor, workloadInfo);

    inputHandle->Allocate();
    indicesHandle->Allocate();
    updatesHandle->Allocate();
    outputHandle->Allocate();

    if (input.data() != nullptr)
    {
        CopyDataToITensorHandle(inputHandle.get(), input.data());
    }
    if (indices.data() != nullptr)
    {
        CopyDataToITensorHandle(indicesHandle.get(), indices.data());
    }
    if (updates.data() != nullptr)
    {
        CopyDataToITensorHandle(updatesHandle.get(), updates.data());
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
LayerTestResult<T, 1> ScatterNd1DimUpdateWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({5}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({5}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 0, 0, 0, 0, 0 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({0, 1, 2}, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 2, 3, 0, 0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 1>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 1> ScatterNd1DimUpdateNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({5}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 5 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 1, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 2, 3, 0, 0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 1>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimUpdateWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                            1, 1, 1,
                                                            1, 1, 1 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                                     1, 2, 1,
                                                                     1, 1, 3 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 2>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimUpdateNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 0, 0,
                                                                     0, 2, 0,
                                                                     0, 0, 3 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 2>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2Dim1Outter1InnerUpdateWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({2, 1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({2, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                            1, 1, 1,
                                                            1, 1, 1 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 1 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                              2, 2, 2 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({  1, 1, 1,
                                                                      2, 2, 2,
                                                                      1, 1, 1 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 2>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2Dim1Outter1InnerUpdateNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({2, 1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({2, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 1 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                              1, 1, 1 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                                     1, 1, 1,
                                                                     0, 0, 0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 2>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> ScatterNd3DimUpdateWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 3}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0, 0,
                                                                  1, 1, 1,
                                                                  2, 2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 0, 0,
                                                                     0, 0, 0,
                                                                     0, 0, 0,

                                                                     0, 0, 0,
                                                                     0, 2, 0,
                                                                     0, 0, 0,

                                                                     0, 0, 0,
                                                                     0, 0, 0,
                                                                     0, 0, 3 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 3>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> ScatterNd3DimUpdateNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({3}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 3}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0, 0,
                                                                  1, 1, 1,
                                                                  2, 2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 0, 0,
                                                                     0, 0, 0,
                                                                     0, 0, 0,

                                                                     0, 0, 0,
                                                                     0, 2, 0,
                                                                     0, 0, 0,

                                                                     0, 0, 0,
                                                                     0, 0, 0,
                                                                     0, 0, 3 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 3>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> ScatterNd3Dim1Outter2InnerUpdateWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({2, 1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({2, 3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 1 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                              1, 1, 1,
                                                              1, 1, 1,

                                                              2, 2, 2,
                                                              2, 2, 2,
                                                              2, 2, 2 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                                     1, 1, 1,
                                                                     1, 1, 1,

                                                                     2, 2, 2,
                                                                     2, 2, 2,
                                                                     2, 2, 2,

                                                                     0, 0, 0,
                                                                     0, 0, 0,
                                                                     0, 0, 0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 3>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> ScatterNd3Dim1Outter2InnerUpdateNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({3}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({2, 1}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({2, 3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 1 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                              1, 1, 1,
                                                              1, 1, 1,

                                                              2, 2, 2,
                                                              2, 2, 2,
                                                              2, 2, 2 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                                     1, 1, 1,
                                                                     1, 1, 1,

                                                                     2, 2, 2,
                                                                     2, 2, 2,
                                                                     2, 2, 2,

                                                                     0, 0, 0,
                                                                     0, 0, 0,
                                                                     0, 0, 0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 3>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> ScatterNd3Dim2Outter1InnerUpdateWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({2, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({2, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1, }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                              2, 2, 2 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                                     0, 0, 0,
                                                                     0, 0, 0,

                                                                     0, 0, 0,
                                                                     2, 2, 2,
                                                                     0, 0, 0,

                                                                     0, 0, 0,
                                                                     0, 0, 0,
                                                                     0, 0, 0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 3>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> ScatterNd3Dim2Outter1InnerUpdateNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({3}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({2, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({2, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1, }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                              2, 2, 2 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                                     0, 0, 0,
                                                                     0, 0, 0,

                                                                     0, 0, 0,
                                                                     2, 2, 2,
                                                                     0, 0, 0,

                                                                     0, 0, 0,
                                                                     0, 0, 0,
                                                                     0, 0, 0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 3>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ScatterNd4DimUpdateWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({2, 3, 3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 4}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 3, 3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0,

                                                            0, 0, 0,
                                                            0, 0, 0,
                                                            0, 0, 0 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0, 0, 0,
                                                                  0, 1, 1, 1,
                                                                  1, 1, 1, 1 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({  1, 0, 0,
                                                                      0, 0, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 2, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 0, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 0, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 3, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 0, 0,
                                                                      0, 0, 0  }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 4>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> ScatterNd4DimUpdateNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({4}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 4}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({2, 3, 3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 2, 3, 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0, 0, 0,
                                                                  0, 1, 1, 1,
                                                                  1, 1, 1, 1 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({  1, 0, 0,
                                                                      0, 0, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 2, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 0, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 0, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 3, 0,
                                                                      0, 0, 0,

                                                                      0, 0, 0,
                                                                      0, 0, 0,
                                                                      0, 0, 0  }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Update, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 4>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimAddWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                            1, 1, 1,
                                                            1, 1, 1 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 2, 1, 1,
                                                                     1, 3, 1,
                                                                     1, 1, 4 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Add, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 2>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimAddNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1, 0, 0,
                                                                     0, 2, 0,
                                                                     0, 0, 3 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Add, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 2>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimSubWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                            1, 1, 1,
                                                            1, 1, 1 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 0,  1,  1,
                                                                     1, -1,  1,
                                                                     1,  1, -2 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Sub, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 2>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimSubNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ -1,  0,  0,
                                                                      0, -2,  0,
                                                                      0,  0, -3 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Sub, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 2>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimMaxWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                            1, 1, 1,
                                                            1, 1, 1 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 0, 1, 2 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1,  1,  1,
                                                                     1,  1,  1,
                                                                     1,  1,  2 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Max, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 2>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimMaxNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ -1, 0, 1 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({  0,  0,  0,
                                                                      0,  0,  0,
                                                                      0,  0,  1 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Max, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 2>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimMinWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                            1, 1, 1,
                                                            1, 1, 1 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 0, 1, 2 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 0,  1,  1,
                                                                     1,  1,  1,
                                                                     1,  1,  1 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Min, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 2>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimMinNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ -1, 0, 1 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ -1,  0,  0,
                                                                      0,  0,  0,
                                                                      0,  0,  0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Min, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 2>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimMulWithInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo inputInfo({3, 3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<T> input = armnnUtils::QuantizedVector<T>({ 1, 1, 1,
                                                            1, 1, 1,
                                                            1, 1, 1 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 1,  1,  1,
                                                                     1,  2,  1,
                                                                     1,  1,  3 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Mul, true);

    return ScatterNdTestImpl<ArmnnType, T, T, 2>(workloadFactory,
                                                 memoryManager,
                                                 tensorHandleFactory,
                                                 input,
                                                 indices,
                                                 updates,
                                                 outputExpected,
                                                 inputInfo,
                                                 indicesInfo,
                                                 updatesInfo,
                                                 outputInfo,
                                                 descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> ScatterNd2DimMulNoInput(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorInfo shapeInfo({2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo indicesInfo({3, 2}, armnn::DataType::Signed32, qScale, qOffset);
    armnn::TensorInfo updatesInfo({3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({3, 3}, ArmnnType, qScale, qOffset);

    std::vector<int> shape = armnnUtils::QuantizedVector<int>({ 3, 3 }, qScale, qOffset);

    std::vector<int> indices = armnnUtils::QuantizedVector<int>({ 0, 0,
                                                                  1, 1,
                                                                  2, 2 }, qScale, qOffset);

    std::vector<T> updates = armnnUtils::QuantizedVector<T>({ 1, 2, 3 }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({ 0,  0,  0,
                                                                     0,  0,  0,
                                                                     0,  0,  0 }, qScale, qOffset);

    armnn::ScatterNdDescriptor descriptor(armnn::ScatterNdFunction::Mul, false);

    return ScatterNdTestImpl<ArmnnType, T, int, 2>(workloadFactory,
                                                   memoryManager,
                                                   tensorHandleFactory,
                                                   shape,
                                                   indices,
                                                   updates,
                                                   outputExpected,
                                                   shapeInfo,
                                                   indicesInfo,
                                                   updatesInfo,
                                                   outputInfo,
                                                   descriptor);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 1>
ScatterNd1DimUpdateWithInput<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 1>
ScatterNd1DimUpdateNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimUpdateWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimUpdateNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2Dim1Outter1InnerUpdateWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2Dim1Outter1InnerUpdateNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
ScatterNd3DimUpdateWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
ScatterNd3DimUpdateNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
ScatterNd3Dim1Outter2InnerUpdateWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
ScatterNd3Dim1Outter2InnerUpdateNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
ScatterNd3Dim2Outter1InnerUpdateWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
ScatterNd3Dim2Outter1InnerUpdateNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ScatterNd4DimUpdateWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
ScatterNd4DimUpdateNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimAddWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimAddNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimSubWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimSubNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimMaxWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimMaxNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimMinWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimMinNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimMulWithInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
ScatterNd2DimMulNoInput<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
ScatterNd3DimUpdateWithInput<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
ScatterNd3DimUpdateNoInput<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
ScatterNd3DimUpdateWithInput<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
ScatterNd3DimUpdateNoInput<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
ScatterNd3DimUpdateWithInput<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
ScatterNd3DimUpdateNoInput<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 3>
ScatterNd3DimUpdateWithInput<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 3>
ScatterNd3DimUpdateNoInput<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
ScatterNd3DimUpdateWithInput<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
ScatterNd3DimUpdateNoInput<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 3>
ScatterNd3DimUpdateWithInput<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 3>
ScatterNd3DimUpdateNoInput<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);