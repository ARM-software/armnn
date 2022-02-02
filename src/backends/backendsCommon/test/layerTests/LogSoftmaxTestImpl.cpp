//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LogSoftmaxTestImpl.hpp"

#include <Half.hpp>
#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>


#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<armnn::DataType ArmnnType,
         std::size_t NumDims,
         typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, NumDims> LogSoftmaxTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::TensorInfo& inputInfo,
    const armnn::TensorInfo& outputInfo,
    const std::vector<float>& inputValues,
    const std::vector<float>& expectedOutputValues,
    armnn::LogSoftmaxQueueDescriptor descriptor,
    float qScale = 1.0f,
    int32_t qOffset = 0)
{
    IgnoreUnused(memoryManager);

    auto inputTensor = armnnUtils::QuantizedVector<T>(inputValues, qScale, qOffset);

    std::vector<T> actualOutput(outputInfo.GetNumElements());
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(expectedOutputValues, qScale, qOffset);

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputInfo);

    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::LogSoftmax,
                                                                                descriptor,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputTensor.data());

    ExecuteWorkload(*workload, memoryManager);

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, NumDims>(actualOutput,
                                       expectedOutput,
                                       outputHandle->GetShape(),
                                       outputInfo.GetShape());

}

} // anonymous namespace

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> LogSoftmaxTest1(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputOutputShape{1, 1, 2, 4};

    armnn::TensorInfo inputTensorInfo(inputOutputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(inputOutputShape, ArmnnType);

    std::vector<float> inputValues
    {
        0.f, -6.f,  2.f, 4.f,
        3.f, -2.f, 10.f, 1.f
    };

    std::vector<float> expectedOutputValues
    {
        -4.14297f, -10.14297f, -2.14297f, -0.14297f,
        -7.00104f, -12.00104f, -0.00105f, -9.00104f
    };

    armnn::LogSoftmaxQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Beta = 1.0f; // default beta
    descriptor.m_Parameters.m_Axis = -1;   // default axis

    return LogSoftmaxTestImpl<ArmnnType, 4>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        inputValues,
        expectedOutputValues,
        descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> LogSoftmaxTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputOutputShape{1, 1, 2, 4};

    armnn::TensorInfo inputTensorInfo(inputOutputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(inputOutputShape, ArmnnType);

    std::vector<float> inputValues
    {
        0.f, -6.f,  2.f, 4.f,
        3.f, -2.f, 10.f, 1.f
    };

    std::vector<float> expectedOutputValues
    {
        -4.14297f, -10.14297f, -2.14297f, -0.14297f,
        -7.00104f, -12.00104f, -0.00105f, -9.00104f
    };

    armnn::LogSoftmaxQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Beta = 1.0f; // default beta
    descriptor.m_Parameters.m_Axis = 3;    // positive axis

    return LogSoftmaxTestImpl<ArmnnType, 4>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        inputValues,
        expectedOutputValues,
        descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> LogSoftmaxTest3(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputOutputShape{1, 1, 2, 4};

    armnn::TensorInfo inputTensorInfo(inputOutputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(inputOutputShape, ArmnnType);

    std::vector<float> inputValues
    {
        0.0f, -0.6f, 0.2f, 0.4f,
        0.3f, -0.2f, 1.0f, 0.1f
    };

    std::vector<float> expectedOutputValues
    {
        -4.14297f, -10.14297f, -2.14297f, -0.14297f,
        -7.00104f, -12.00104f, -0.00105f, -9.00104f
    };

    armnn::LogSoftmaxQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Beta = 10.0f; // non-default beta
    descriptor.m_Parameters.m_Axis = 3;     // positive axis

    return LogSoftmaxTestImpl<ArmnnType, 4>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        inputValues,
        expectedOutputValues,
        descriptor);
}

template<armnn::DataType ArmnnType, typename T>
LayerTestResult<T, 4> LogSoftmaxTest4(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    const armnn::TensorShape inputOutputShape{1, 1, 2, 4};

    armnn::TensorInfo inputTensorInfo(inputOutputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(inputOutputShape, ArmnnType);

    std::vector<float> inputValues
    {
        0.f, -6.f,  2.f, 4.f,
        3.f, -2.f, 10.f, 1.f
    };

    std::vector<float> expectedOutputValues
    {
        -3.048587f, -4.018149f, -8.000336f, -0.048587f,
        -0.048587f, -0.018149f, -0.000335f, -3.048587f
    };

    armnn::LogSoftmaxQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Beta = 1.0f; // default beta
    descriptor.m_Parameters.m_Axis = -2;   // negative axis

    return LogSoftmaxTestImpl<ArmnnType, 4>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        inputValues,
        expectedOutputValues,
        descriptor);
}

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
LogSoftmaxTest1<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
LogSoftmaxTest2<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
LogSoftmaxTest3<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float32>, 4>
LogSoftmaxTest4<armnn::DataType::Float32>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
LogSoftmaxTest1<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
LogSoftmaxTest2<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
LogSoftmaxTest3<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);

template LayerTestResult<armnn::ResolveType<armnn::DataType::Float16>, 4>
LogSoftmaxTest4<armnn::DataType::Float16>(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory);
