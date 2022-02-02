//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "InstanceNormalizationTestImpl.hpp"

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>


#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/DataLayoutUtils.hpp>
#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

namespace
{

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> InstanceNormTestImpl(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    const armnn::TensorInfo& inputTensorInfo,
    const armnn::TensorInfo& outputTensorInfo,
    const std::vector<float>& inputValues,
    const std::vector<float>& expectedOutputValues,
    armnn::InstanceNormalizationQueueDescriptor descriptor,
    float qScale = 0.0f,
    int32_t qOffset = 0)
{
    IgnoreUnused(memoryManager);
    std::vector<T> inputTensor = armnnUtils::QuantizedVector<T>(inputValues, qScale, qOffset);
    std::vector<T> expectedOutput = armnnUtils::QuantizedVector<T>(expectedOutputValues, qScale, qOffset);
    std::vector<T> actualOutput(outputTensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::WorkloadInfo info;

    AddInputToWorkload(descriptor, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload
            = workloadFactory.CreateWorkload(armnn::LayerType::InstanceNormalization, descriptor, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), inputTensor.data());

    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    return LayerTestResult<T, 4>(actualOutput,
                                 expectedOutput,
                                 outputHandle->GetShape(),
                                 outputTensorInfo.GetShape());
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> InstanceNormTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout)
{
    // BatchSize: 2
    // Height: 2
    // Width: 2
    // Channels: 2

    const armnn::TensorShape inputOutputShape{ 2, 2, 2, 2 };

    armnn::TensorInfo inputTensorInfo(inputOutputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(inputOutputShape, ArmnnType);

    std::vector<float> inputValues
        {
            // Batch 0, Height 0, Width 0 x Channel (2)
            0.f,  1.f,
            // Batch 0, Height 0, Width 1 x Channel (2)
            0.f,  2.f,

            // Batch 0, Height 1, Width 0 x Channel (2)
            0.f,  2.f,
            // Batch 0, Height 1, Width 1 x Channel (2)
            0.f,  4.f,

            // Batch 1, Height 0, Width 0 x Channel (2)
            1.f, -1.f,
            // Batch 1, Height 0, Width 1 x Channel (2)
           -1.f,  2.f,

            // Batch 1, Height 1, Width 0 x Channel (2)
           -1.f, -2.f,
            // Batch 1, Height 1, Width 1 x Channel (2)
            1.f,  4.f
        };

    std::vector<float> expectedOutputValues
        {
            // Batch 0, Height 0, Width 0 x Channel (2)
            0.f, -1.1470304f,
            // Batch 0, Height 0, Width 1 x Channel (2)
            0.f, -0.22940612f,
            // Batch 0, Height 1, Width 0 x Channel (2)
            0.f, -0.22940612f,
            // Batch 0, Height 1, Width 1 x Channel (2)
            0.f,  1.6058424f,

            // Batch 1, Height 0, Width 0 x Channel (2)
            0.99995005f, -0.7337929f,
            // Batch 1, Height 0, Width 1 x Channel (2)
           -0.99995005f,  0.52413774f,

            // Batch 1, Height 1, Width 0 x Channel (2)
           -0.99995005f, -1.1531031f,
            // Batch 1, Height 1, Width 1 x Channel (2)
            0.99995005f,  1.3627582f
        };

    if (dataLayout == armnn::DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw(inputTensorInfo, inputValues);
        PermuteTensorNhwcToNchw(outputTensorInfo, expectedOutputValues);
    }

    armnn::InstanceNormalizationQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Eps        = 0.0001f;
    descriptor.m_Parameters.m_Beta       = 0.0f;
    descriptor.m_Parameters.m_Gamma      = 1.0f;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    return InstanceNormTestImpl<ArmnnType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        inputValues,
        expectedOutputValues,
        descriptor);
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> InstanceNormTest2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout)
{
    // BatchSize: 2
    // Height: 2
    // Width: 2
    // Channels: 2

    const armnn::TensorShape inputOutputShape{ 2, 2, 2, 2 };

    armnn::TensorInfo inputTensorInfo(inputOutputShape, ArmnnType);
    armnn::TensorInfo outputTensorInfo(inputOutputShape, ArmnnType);

    std::vector<float> inputValues
        {
            // Batch 0, Height 0, Width 0 x Channel (2)
            0.f,  1.f,
            // Batch 0, Height 0, Width 1 x Channel (2)
            0.f,  2.f,

            // Batch 0, Height 1, Width 0 x Channel (2)
            0.f,  2.f,
            // Batch 0, Height 1, Width 1 x Channel (2)
            0.f,  4.f,

            // Batch 1, Height 0, Width 0 x Channel (2)
            1.f, -1.f,
            // Batch 1, Height 0, Width 1 x Channel (2)
            -1.f,  2.f,

            // Batch 1, Height 1, Width 0 x Channel (2)
            -1.f, -2.f,
            // Batch 1, Height 1, Width 1 x Channel (2)
            1.f,  4.f
        };

    std::vector<float> expectedOutputValues
        {
            // Batch 0, Height 0, Width 0 x Channel (2)
            10.f,     7.7059393f,
            // Batch 0, Height 0, Width 1 x Channel (2)
            10.f,     9.541187f,

            // Batch 0, Height 1, Width 0 x Channel (2)
            10.f,     9.541187f,
            // Batch 0, Height 1, Width 1 x Channel (2)
            10.f,     13.211685f,

            // Batch 1, Height 0, Width 0 x Channel (2)
            11.9999f, 8.532414f,
            // Batch 1, Height 0, Width 1 x Channel (2)
            8.0001f,  11.048275f,

            // Batch 1, Height 1, Width 0 x Channel (2)
            8.0001f,  7.693794f,
            // Batch 1, Height 1, Width 1 x Channel (2)
            11.9999f, 12.725516f
        };

    if (dataLayout == armnn::DataLayout::NCHW)
    {
        PermuteTensorNhwcToNchw(inputTensorInfo, inputValues);
        PermuteTensorNhwcToNchw(outputTensorInfo, expectedOutputValues);
    }

    armnn::InstanceNormalizationQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Eps        = 0.0001f;
    descriptor.m_Parameters.m_Beta       = 10.0f;
    descriptor.m_Parameters.m_Gamma      = 2.0f;
    descriptor.m_Parameters.m_DataLayout = dataLayout;

    return InstanceNormTestImpl<ArmnnType>(
        workloadFactory,
        memoryManager,
        tensorHandleFactory,
        inputTensorInfo,
        outputTensorInfo,
        inputValues,
        expectedOutputValues,
        descriptor);
}

} // anonymous namespace

LayerTestResult<float, 4> InstanceNormFloat32Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout)
{
    return InstanceNormTest<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<armnn::Half, 4> InstanceNormFloat16Test(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout)
{
    return InstanceNormTest<armnn::DataType::Float16>(workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<float, 4> InstanceNormFloat32Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout)
{
    return InstanceNormTest2<armnn::DataType::Float32>(workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}

LayerTestResult<armnn::Half, 4> InstanceNormFloat16Test2(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory,
    armnn::DataLayout dataLayout)
{
    return InstanceNormTest2<armnn::DataType::Float16>(workloadFactory, memoryManager, tensorHandleFactory, dataLayout);
}
