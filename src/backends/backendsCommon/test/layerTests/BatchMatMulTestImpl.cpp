//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchMatMulTestImpl.hpp"

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/WorkloadTestUtils.hpp>
#include <armnnUtils/QuantizeHelper.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnn/Optional.hpp>


template<armnn::DataType ArmnnType, typename T, std::size_t NumDims>
LayerTestResult<T, NumDims> BatchMatMulTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::BatchMatMulDescriptor descriptor,
    const std::vector<T>& inputX,
    const std::vector<T>& inputY,
    const std::vector<T>& outputExpected,
    const armnn::TensorInfo& inputXInfo,
    const armnn::TensorInfo& inputYInfo,
    const armnn::TensorInfo& outputInfo)
{
    std::vector<T> outputActual(outputInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputXHandle = tensorHandleFactory.CreateTensorHandle(inputXInfo);
    std::unique_ptr<armnn::ITensorHandle> inputYHandle = tensorHandleFactory.CreateTensorHandle(inputYInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    armnn::BatchMatMulQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = descriptor;
    armnn::WorkloadInfo workloadInfo;

    AddInputToWorkload(queueDescriptor, workloadInfo, inputXInfo, inputXHandle.get());
    AddInputToWorkload(queueDescriptor, workloadInfo, inputYInfo, inputYHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, outputHandle.get());

    auto workload = workloadFactory.CreateWorkload(armnn::LayerType::BatchMatMul, queueDescriptor, workloadInfo);

    inputXHandle->Allocate();
    inputYHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputXHandle.get(), inputX.data());
    CopyDataToITensorHandle(inputYHandle.get(), inputY.data());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(outputActual.data(), outputHandle.get());

    return LayerTestResult<T, NumDims>(outputActual,
                                       outputExpected,
                                       outputHandle->GetShape(),
                                       outputInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> BatchMatMul2DSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(); // Arbitrary layout with no transpose/adjointing

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
       1, 2,
       3, 4
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        5, 6,
        7, 8
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        19, 22,
        43, 50
    }, qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 2>(workloadFactory,
                                             memoryManager,
                                             tensorHandleFactory,
                                             descriptor,
                                             inputX,
                                             inputY,
                                             outputExpected,
                                             inputXInfo,
                                             inputYInfo,
                                             outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 2>
BatchMatMul2DSimpleTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
BatchMatMul2DSimpleTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
BatchMatMul2DSimpleTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
BatchMatMul2DSimpleTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
BatchMatMul2DSimpleTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
BatchMatMul2DSimpleTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> BatchMatMul3DSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(); // Arbitrary layout with no transpose/adjointing

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({1,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({1,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({1,2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
       1, 2,
       3, 4
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        5, 6,
        7, 8
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        19, 22,
        43, 50
    },qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 3>(workloadFactory,
                                                memoryManager,
                                                tensorHandleFactory,
                                                descriptor,
                                                inputX,
                                                inputY,
                                                outputExpected,
                                                inputXInfo,
                                                inputYInfo,
                                                outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 3>
BatchMatMul3DSimpleTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
BatchMatMul3DSimpleTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
BatchMatMul3DSimpleTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
BatchMatMul3DSimpleTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
BatchMatMul3DSimpleTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
BatchMatMul3DSimpleTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> BatchMatMulNCHWSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(
        armnn::Optional<armnn::DataLayout>(armnn::DataLayout::NCHW),
        armnn::Optional<armnn::DataLayout>(armnn::DataLayout::NCHW));

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({1,1,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({1,1,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({1,1,2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
       1, 2,
       3, 4
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        5, 6,
        7, 8
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        19, 22,
        43, 50
    },qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 4>(workloadFactory,
                                                memoryManager,
                                                tensorHandleFactory,
                                                descriptor,
                                                inputX,
                                                inputY,
                                                outputExpected,
                                                inputXInfo,
                                                inputYInfo,
                                                outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 4>
BatchMatMulNCHWSimpleTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
BatchMatMulNCHWSimpleTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
BatchMatMulNCHWSimpleTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
BatchMatMulNCHWSimpleTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
BatchMatMulNCHWSimpleTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
BatchMatMulNCHWSimpleTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> BatchMatMulNHWCSimpleTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(
        armnn::Optional<armnn::DataLayout>(armnn::DataLayout::NHWC),
        armnn::Optional<armnn::DataLayout>(armnn::DataLayout::NHWC));

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({1,2,2,1}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({1,2,2,1}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({1,2,2,1}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
       1, 2,
       3, 4
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        5, 6,
        7, 8
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        19, 22,
        43, 50
    },qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 4>(workloadFactory,
                                                memoryManager,
                                                tensorHandleFactory,
                                                descriptor,
                                                inputX,
                                                inputY,
                                                outputExpected,
                                                inputXInfo,
                                                inputYInfo,
                                                outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 4>
BatchMatMulNHWCSimpleTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
BatchMatMulNHWCSimpleTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
BatchMatMulNHWCSimpleTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
BatchMatMulNHWCSimpleTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
BatchMatMulNHWCSimpleTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
BatchMatMulNHWCSimpleTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> BatchMatMul3DBatchTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(); // Arbitrary layout with no transpose/adjointing

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({2,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({2,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({2,2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
       1, 2,
       3, 4,

       9, 10,
       11, 12
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        5, 6,
        7, 8,

        13, 14,
        15, 16
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        19, 22,
        43, 50,

        267, 286,
        323, 346
    },qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 3>(workloadFactory,
                                                memoryManager,
                                                tensorHandleFactory,
                                                descriptor,
                                                inputX,
                                                inputY,
                                                outputExpected,
                                                inputXInfo,
                                                inputYInfo,
                                                outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 3>
BatchMatMul3DBatchTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
BatchMatMul3DBatchTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
BatchMatMul3DBatchTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
BatchMatMul3DBatchTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
BatchMatMul3DBatchTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
BatchMatMul3DBatchTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> BatchMatMul3DBroadcastTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(); // Arbitrary layout with no transpose/adjointing

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({2,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({1,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({2,2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
       1, 2,
       3, 4,

       9, 10,
       11, 12
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        13, 14,
        15, 16
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        43, 46,
        99, 106,

        267, 286,
        323, 346
    },qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 3>(workloadFactory,
                                                memoryManager,
                                                tensorHandleFactory,
                                                descriptor,
                                                inputX,
                                                inputY,
                                                outputExpected,
                                                inputXInfo,
                                                inputYInfo,
                                                outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 3>
BatchMatMul3DBroadcastTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
BatchMatMul3DBroadcastTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
BatchMatMul3DBroadcastTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
BatchMatMul3DBroadcastTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
BatchMatMul3DBroadcastTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
BatchMatMul3DBroadcastTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> BatchMatMul3D2DBroadcastTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(); // Arbitrary layout with no transpose/adjointing

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({2,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({2,2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
       1, 2,
       3, 4,

       9, 10,
       11, 12
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        13, 14,
        15, 16
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        43, 46,
        99, 106,

        267, 286,
        323, 346
    },qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 3>(workloadFactory,
                                                memoryManager,
                                                tensorHandleFactory,
                                                descriptor,
                                                inputX,
                                                inputY,
                                                outputExpected,
                                                inputXInfo,
                                                inputYInfo,
                                                outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 3>
BatchMatMul3D2DBroadcastTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
BatchMatMul3D2DBroadcastTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
BatchMatMul3D2DBroadcastTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
BatchMatMul3D2DBroadcastTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
BatchMatMul3D2DBroadcastTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
BatchMatMul3D2DBroadcastTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 5> BatchMatMulNDHWCNHWCTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(
        armnn::Optional<armnn::DataLayout>(armnn::DataLayout::NDHWC),
        armnn::Optional<armnn::DataLayout>(armnn::DataLayout::NHWC));

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({1,1,2,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({1,2,2,2}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({1,1,2,2,2}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
        1, 20,
        3, 22,

        2, 21,
        4, 23
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        5, 24,
        7, 26,

        6, 25,
        8, 27
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
       23, 1030,
       31, 1114,

       34, 1079,
       46, 1167
    },qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 5>(workloadFactory,
                                                memoryManager,
                                                tensorHandleFactory,
                                                descriptor,
                                                inputX,
                                                inputY,
                                                outputExpected,
                                                inputXInfo,
                                                inputYInfo,
                                                outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 5>
BatchMatMulNDHWCNHWCTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 5>
BatchMatMulNDHWCNHWCTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 5>
BatchMatMulNDHWCNHWCTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 5>
BatchMatMulNDHWCNHWCTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 5>
BatchMatMulNDHWCNHWCTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 5>
BatchMatMulNDHWCNHWCTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> BatchMatMul2DTinyTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(); // Arbitrary layout with no transpose/adjointing

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({1,1}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({1,1}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({1,1}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
       3
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        5
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        15
    }, qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 2>(workloadFactory,
                                             memoryManager,
                                             tensorHandleFactory,
                                             descriptor,
                                             inputX,
                                             inputY,
                                             outputExpected,
                                             inputXInfo,
                                             inputYInfo,
                                             outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 2>
BatchMatMul2DTinyTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
BatchMatMul2DTinyTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
BatchMatMul2DTinyTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
BatchMatMul2DTinyTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
BatchMatMul2DTinyTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
BatchMatMul2DTinyTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> BatchMatMul3DNonSquareTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BatchMatMulDescriptor(); // Arbitrary layout with no transpose/adjointing

    float qScale = 0.0f;
    int32_t qOffset = 0;

    switch(ArmnnType)
    {
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS16:
            qScale = 1.0f;
            break;
        default:
            break;
    }

    armnn::TensorInfo inputXInfo({2,5,3}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo inputYInfo({2,3,4}, ArmnnType, qScale, qOffset);
    armnn::TensorInfo outputInfo({2,5,4}, ArmnnType, qScale, qOffset);

    std::vector<T> inputX = armnnUtils::QuantizedVector<T>({
       8, 8, 4,
       6, 1, 3,
       8, 8, 3,
       8, 9, 8,
       5, 4, 4,

       1, 8, 5,
       7, 1, 1,
       8, 7, 9,
       3, 2, 7,
       8, 5, 3
    }, qScale, qOffset);

    std::vector<T> inputY = armnnUtils::QuantizedVector<T>({
        6, 2, 3, 2,
        6, 2, 2, 8,
        3, 7, 8, 1,

        7, 2, 9, 5,
        2, 3, 1, 3,
        2, 7, 7, 5
    }, qScale, qOffset);

    std::vector<T> outputExpected = armnnUtils::QuantizedVector<T>({
        108, 60, 72, 84,
        51, 35, 44, 23,
        105, 53, 64, 83,
        126, 90, 106, 96,
        66, 46, 55, 46,

        33, 61, 52, 54,
        53, 24, 71, 43,
        88, 100, 142, 106,
        39, 61, 78, 56,
        72, 52, 98, 70
    },qScale, qOffset);

    return BatchMatMulTestImpl<ArmnnType, T, 3>(workloadFactory,
                                                memoryManager,
                                                tensorHandleFactory,
                                                descriptor,
                                                inputX,
                                                inputY,
                                                outputExpected,
                                                inputXInfo,
                                                inputYInfo,
                                                outputInfo);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::BFloat16>, 3>
BatchMatMul3DNonSquareTest<armnn::DataType::BFloat16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
BatchMatMul3DNonSquareTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
BatchMatMul3DNonSquareTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
BatchMatMul3DNonSquareTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
BatchMatMul3DNonSquareTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
BatchMatMul3DNonSquareTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);