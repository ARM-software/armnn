//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BroadcastToTestImpl.hpp"
#include <vector>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/WorkloadTestUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>

#include <armnn/BackendHelper.hpp>

#include <armnnUtils/QuantizeHelper.hpp>
#include <doctest/doctest.h>

namespace
{
template<typename T, std::size_t NumDims>
LayerTestResult<T, NumDims> BroadcastToTestImpl(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory,
        armnn::BroadcastToDescriptor descriptor,
        armnn::TensorInfo& inputInfo,
        armnn::TensorInfo& outputInfo,
        std::vector<T>& inputData,
        std::vector<T>& expectedOutputData)
{

    CHECK(descriptor.m_BroadcastToShape == outputInfo.GetShape());

    LayerTestResult<T, NumDims> result(outputInfo);
    std::vector<T> outputActual(outputInfo.GetNumElements());

    armnn::BroadcastToQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = std::move(descriptor);
    armnn::WorkloadInfo workloadInfo;

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, inputHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, outputHandle.get());

    const armnn::BackendId& backend = workloadFactory.GetBackendId();
    armnn::LayerSupportHandle handle = armnn::GetILayerSupportByBackendId(backend);

    auto workload = workloadFactory.CreateWorkload(armnn::LayerType::BroadcastTo, queueDescriptor, workloadInfo);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputData.data());

    workload->PostAllocationConfigure();
    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(outputActual.data(), outputHandle.get());
    return LayerTestResult<T, NumDims>(outputActual,
                                       expectedOutputData,
                                       outputHandle->GetShape(),
                                       outputInfo.GetShape());
}
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 1> BroadcastTo1dTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BroadcastToDescriptor(armnn::TensorShape( {1, 4} ));

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 1, 1 };
    armnn::TensorShape outputShape = { 1, 4 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
            {
                    1.f
            }, qScale, qOffset);

    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
            {
                    1.f, 1.f,
                    1.f, 1.f
            }, qScale, qOffset);

    return BroadcastToTestImpl<T, 1>(workloadFactory,
                              memoryManager,
                              tensorHandleFactory,
                              descriptor,
                              inputInfo,
                              outputInfo,
                              input,
                              expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> BroadcastTo2dAxis0Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BroadcastToDescriptor(armnn::TensorShape({ 4, 3 }));

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 1, 3 };
    armnn::TensorShape outputShape = { 4, 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
            {
                    0.f, 1.f, 2.f
            }, qScale, qOffset);


    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
            {
                    0.f, 1.f, 2.f,
                    0.f, 1.f, 2.f,
                    0.f, 1.f, 2.f,
                    0.f, 1.f, 2.f
            }, qScale, qOffset);

    return BroadcastToTestImpl<T, 2>(workloadFactory,
                              memoryManager,
                              tensorHandleFactory,
                              descriptor,
                              inputInfo,
                              outputInfo,
                              input,
                              expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> BroadcastTo2dAxis1Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BroadcastToDescriptor(armnn::TensorShape({ 3, 4 }));

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 3, 1 };
    armnn::TensorShape outputShape = { 3, 4 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
            {
                    0.f, 1.f, 2.f
            }, qScale, qOffset);


    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
            {
                    0.f, 0.f, 0.f, 0.f,
                    1.f, 1.f, 1.f, 1.f,
                    2.f, 2.f, 2.f, 2.f
            }, qScale, qOffset);

    return BroadcastToTestImpl<T, 2>(workloadFactory,
                                     memoryManager,
                                     tensorHandleFactory,
                                     descriptor,
                                     inputInfo,
                                     outputInfo,
                                     input,
                                     expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> BroadcastTo3dAxis0Test(armnn::IWorkloadFactory& workloadFactory,
                                        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BroadcastToDescriptor(armnn::TensorShape({ 2, 1, 3 }));

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 1, 1, 3 };
    armnn::TensorShape outputShape = { 2, 1, 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);
    std::vector<T> input = armnnUtils::QuantizedVector<T>(
            {
                    1.1f, 2.12f, 3.3f
            }, qScale, qOffset);


    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
            {
                    1.1f, 2.12f, 3.3f,
                    1.1f, 2.12f, 3.3f
            }, qScale, qOffset);

    return BroadcastToTestImpl<T, 3>(workloadFactory,
                              memoryManager,
                              tensorHandleFactory,
                              descriptor,
                              inputInfo,
                              outputInfo,
                              input,
                              expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> BroadcastTo3dAxis1Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BroadcastToDescriptor(armnn::TensorShape({ 1, 3, 3 }));

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 1, 1, 3 };
    armnn::TensorShape outputShape = { 1, 3, 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);
    std::vector<T> input = armnnUtils::QuantizedVector<T>(
            {
                    1.1f, 2.12f, 3.3f
            }, qScale, qOffset);


    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
            {
                    1.1f, 2.12f, 3.3f,
                    1.1f, 2.12f, 3.3f,
                    1.1f, 2.12f, 3.3f
            }, qScale, qOffset);

    return BroadcastToTestImpl<T, 3>(workloadFactory,
                                     memoryManager,
                                     tensorHandleFactory,
                                     descriptor,
                                     inputInfo,
                                     outputInfo,
                                     input,
                                     expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> BroadcastTo3dAxis2Test(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BroadcastToDescriptor(armnn::TensorShape({ 1, 3, 3 }));

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 1, 3, 1 };
    armnn::TensorShape outputShape = { 1, 3, 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);
    std::vector<T> input = armnnUtils::QuantizedVector<T>(
            {
                    1.1f, 2.12f, 3.3f
            }, qScale, qOffset);


    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
            {
                    1.1f, 1.1f, 1.1f,
                    2.12f, 2.12f, 2.12f,
                    3.3f, 3.3f, 3.3f
            }, qScale, qOffset);

    return BroadcastToTestImpl<T, 3>(workloadFactory,
                                     memoryManager,
                                     tensorHandleFactory,
                                     descriptor,
                                     inputInfo,
                                     outputInfo,
                                     input,
                                     expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> BroadcastTo4dTest(armnn::IWorkloadFactory& workloadFactory,
                                             const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                             const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::BroadcastToDescriptor(armnn::TensorShape({ 3, 1, 2, 3 }));

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 1, 1, 1, 3 };
    armnn::TensorShape outputShape = { 3, 1, 2, 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
            {
                    0.f,  1.f,  2.f
            }, qScale, qOffset);


    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
            {
                    0.f,  1.f,  2.f,
                    0.f,  1.f,  2.f,
                    0.f,  1.f,  2.f,
                    0.f,  1.f,  2.f,
                    0.f,  1.f,  2.f,
                    0.f,  1.f,  2.f
            }, qScale, qOffset);

    return BroadcastToTestImpl<T, 4>(workloadFactory,
                              memoryManager,
                              tensorHandleFactory,
                              descriptor,
                              inputInfo,
                              outputInfo,
                              input,
                              expectedOutput);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 1>
BroadcastTo1dTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
BroadcastTo2dAxis0Test<armnn::DataType::Float32>(
                      armnn::IWorkloadFactory& workloadFactory,
                      const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                      const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
BroadcastTo2dAxis1Test<armnn::DataType::Float32>(
                       armnn::IWorkloadFactory& workloadFactory,
                       const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                       const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
BroadcastTo3dAxis0Test<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
BroadcastTo3dAxis1Test<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
BroadcastTo3dAxis2Test<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
BroadcastTo4dTest<armnn::DataType::Float32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 1>
BroadcastTo1dTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
BroadcastTo2dAxis0Test<armnn::DataType::Float16>(
                       armnn::IWorkloadFactory& workloadFactory,
                       const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                       const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
BroadcastTo2dAxis1Test<armnn::DataType::Float16>(
                       armnn::IWorkloadFactory& workloadFactory,
                       const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                       const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
BroadcastTo3dAxis0Test<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
BroadcastTo3dAxis1Test<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
BroadcastTo3dAxis2Test<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
BroadcastTo4dTest<armnn::DataType::Float16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 1>
BroadcastTo1dTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
BroadcastTo2dAxis0Test<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
BroadcastTo2dAxis1Test<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
BroadcastTo3dAxis0Test<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
BroadcastTo3dAxis1Test<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
BroadcastTo3dAxis2Test<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
BroadcastTo4dTest<armnn::DataType::QAsymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 1>
BroadcastTo1dTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
BroadcastTo2dAxis0Test<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
BroadcastTo2dAxis1Test<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
BroadcastTo3dAxis0Test<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
BroadcastTo3dAxis1Test<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
BroadcastTo3dAxis2Test<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
BroadcastTo4dTest<armnn::DataType::QAsymmU8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 1>
BroadcastTo1dTest<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 2>
BroadcastTo2dAxis0Test<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 2>
BroadcastTo2dAxis1Test<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 3>
BroadcastTo3dAxis0Test<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 3>
BroadcastTo3dAxis1Test<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 3>
BroadcastTo3dAxis2Test<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 4>
BroadcastTo4dTest<armnn::DataType::QSymmS8>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 1>
BroadcastTo1dTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
BroadcastTo2dAxis0Test<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
BroadcastTo2dAxis1Test<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
BroadcastTo3dAxis0Test<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
BroadcastTo3dAxis1Test<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
BroadcastTo3dAxis2Test<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
BroadcastTo4dTest<armnn::DataType::QSymmS16>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 1>
BroadcastTo1dTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 2>
BroadcastTo2dAxis0Test<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 2>
BroadcastTo2dAxis1Test<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 3>
BroadcastTo3dAxis0Test<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 3>
BroadcastTo3dAxis1Test<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 3>
BroadcastTo3dAxis2Test<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 4>
BroadcastTo4dTest<armnn::DataType::Signed32>(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory);
