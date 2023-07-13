//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TileTestImpl.hpp"
#include <vector>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/WorkloadTestUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>

#include <armnn/BackendHelper.hpp>

#include <armnnUtils/QuantizeHelper.hpp>

namespace
{
template<typename T, std::size_t NumDims>
LayerTestResult<T, NumDims> TileTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::TileDescriptor descriptor,
    armnn::TensorInfo& inputInfo,
    armnn::TensorInfo& outputInfo,
    std::vector<T>& inputData,
    std::vector<T>& expectedOutputData)
{

    LayerTestResult<T, NumDims> result(outputInfo);
    std::vector<T> outputActual(outputInfo.GetNumElements());

    armnn::TileQueueDescriptor queueDescriptor;
    queueDescriptor.m_Parameters = std::move(descriptor);
    armnn::WorkloadInfo workloadInfo;

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    AddInputToWorkload(queueDescriptor, workloadInfo, inputInfo, inputHandle.get());
    AddOutputToWorkload(queueDescriptor, workloadInfo, outputInfo, outputHandle.get());

    const armnn::BackendId& backend = workloadFactory.GetBackendId();
    armnn::LayerSupportHandle handle = armnn::GetILayerSupportByBackendId(backend);

    auto workload = workloadFactory.CreateWorkload(armnn::LayerType::Tile, queueDescriptor, workloadInfo);

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
LayerTestResult<T, 1> Tile1dTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::TileDescriptor( std::vector<uint32_t>{ 3 } );

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 3 };
    armnn::TensorShape outputShape = { 9 };

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
        0.f, 1.f, 2.f
    }, qScale, qOffset);

    return TileTestImpl<T, 1>(workloadFactory,
                              memoryManager,
                              tensorHandleFactory,
                              descriptor,
                              inputInfo,
                              outputInfo,
                              input,
                              expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 2> Tile2dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                 const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::TileDescriptor(std::vector<uint32_t>{ 2, 2 });

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 2, 3 };
    armnn::TensorShape outputShape = { 4, 6 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
    {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f
    }, qScale, qOffset);


    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
    {
        0.f, 1.f, 2.f, 0.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 3.f, 4.f, 5.f,

        0.f, 1.f, 2.f, 0.f, 1.f, 2.f,
        3.f, 4.f, 5.f, 3.f, 4.f, 5.f
    }, qScale, qOffset);

    return TileTestImpl<T, 2>(workloadFactory,
                              memoryManager,
                              tensorHandleFactory,
                              descriptor,
                              inputInfo,
                              outputInfo,
                              input,
                              expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 3> Tile3dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                 const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::TileDescriptor(std::vector<uint32_t>{ 1, 2, 1 });

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 2, 2, 3 };
    armnn::TensorShape outputShape = { 2, 4, 3 };

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);
    std::vector<T> input = armnnUtils::QuantizedVector<T>(
    {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,

        6.f,  7.f,  8.f,
        9.f, 10.f, 11.f
    }, qScale, qOffset);


    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
    {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,

        6.f,  7.f,  8.f,
        9.f, 10.f, 11.f,
        6.f,  7.f,  8.f,
        9.f, 10.f, 11.f
    }, qScale, qOffset);

    return TileTestImpl<T, 3>(workloadFactory,
                              memoryManager,
                              tensorHandleFactory,
                              descriptor,
                              inputInfo,
                              outputInfo,
                              input,
                              expectedOutput);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> Tile4dTest(armnn::IWorkloadFactory& workloadFactory,
                                 const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                 const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    auto descriptor = armnn::TileDescriptor(std::vector<uint32_t>{ 2, 1, 1, 1 });

    float qScale = 1.0f;
    int32_t qOffset = 0;

    armnn::TensorShape inputShape  = { 2, 2, 2, 3};
    armnn::TensorShape outputShape = { 4, 2, 2, 3};

    armnn::TensorInfo inputInfo(inputShape, ArmnnType);
    armnn::TensorInfo outputInfo(outputShape, ArmnnType);

    std::vector<T> input = armnnUtils::QuantizedVector<T>(
    {
        0.f,  1.f,  2.f,
        3.f,  4.f,  5.f,

        6.f,  7.f,  8.f,
        9.f, 10.f, 11.f,

        0.f,  1.f,  2.f,
        3.f,  4.f,  5.f,

        6.f,  7.f,  8.f,
        9.f, 10.f, 11.f
    }, qScale, qOffset);


    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(
    {
        0.f,  1.f,  2.f,
        3.f,  4.f,  5.f,

        6.f,  7.f,  8.f,
        9.f, 10.f, 11.f,

        0.f,  1.f,  2.f,
        3.f,  4.f,  5.f,

        6.f,  7.f,  8.f,
        9.f, 10.f, 11.f,

        0.f,  1.f,  2.f,
        3.f,  4.f,  5.f,

        6.f,  7.f,  8.f,
        9.f, 10.f, 11.f,

        0.f,  1.f,  2.f,
        3.f,  4.f,  5.f,

        6.f,  7.f,  8.f,
        9.f, 10.f, 11.f
    }, qScale, qOffset);

    return TileTestImpl<T, 4>(workloadFactory,
                              memoryManager,
                              tensorHandleFactory,
                              descriptor,
                              inputInfo,
                              outputInfo,
                              input,
                              expectedOutput);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 1>
Tile1dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 2>
Tile2dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 3>
Tile3dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
Tile4dTest<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 1>
Tile1dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 2>
Tile2dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 3>
Tile3dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
Tile4dTest<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 1>
Tile1dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 2>
Tile2dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 3>
Tile3dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmS8>, 4>
Tile4dTest<armnn::DataType::QAsymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 1>
Tile1dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 2>
Tile2dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 3>
Tile3dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QAsymmU8>, 4>
Tile4dTest<armnn::DataType::QAsymmU8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 1>
Tile1dTest<armnn::DataType::QSymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 2>
Tile2dTest<armnn::DataType::QSymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 3>
Tile3dTest<armnn::DataType::QSymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS8>, 4>
Tile4dTest<armnn::DataType::QSymmS8>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 1>
Tile1dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 2>
Tile2dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 3>
Tile3dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::QSymmS16>, 4>
Tile4dTest<armnn::DataType::QSymmS16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 1>
Tile1dTest<armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 2>
Tile2dTest<armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 3>
Tile3dTest<armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Signed32>, 4>
Tile4dTest<armnn::DataType::Signed32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

