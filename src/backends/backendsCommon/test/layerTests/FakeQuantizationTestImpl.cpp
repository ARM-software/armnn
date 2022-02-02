//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FakeQuantizationTestImpl.hpp"


#include <armnn/backends/TensorHandle.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

LayerTestResult<float, 2> FakeQuantizationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    constexpr unsigned int width = 2;
    constexpr unsigned int height = 3;

    const armnn::TensorInfo tensorInfo({ height, width }, armnn::DataType::Float32);

    std::vector<float> input =
    {
       -10.0f, -5.0f,
         0.0f,  5.0f,
        10.0f, 10.0f
    };

    std::vector<float> actualOutput(tensorInfo.GetNumElements());
    std::vector<float> expectedOutput(tensorInfo.GetNumElements());

    std::unique_ptr<armnn::ITensorHandle> inputHandle   = tensorHandleFactory.CreateTensorHandle(tensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle  = tensorHandleFactory.CreateTensorHandle(tensorInfo);

    armnn::FakeQuantizationQueueDescriptor data;
    armnn::WorkloadInfo info;

    AddInputToWorkload(data, info, tensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, tensorInfo, outputHandle.get());

    float min = -10.f;
    float max =  10.f;

    data.m_Parameters.m_Min = min;
    data.m_Parameters.m_Max = max;

    armnn::PassthroughTensorHandle refHandle(tensorInfo, expectedOutput.data());
    armnn::FakeQuantizationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadOutput(refData, refInfo, 0, tensorInfo, &refHandle);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateWorkload(armnn::LayerType::FakeQuantization,
                                                                                data,
                                                                                info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), input.data());

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(actualOutput.data(), outputHandle.get());

    expectedOutput =
    {
        0.0f,     63.0f,
        128.0f,   191.0f,
        255.0f,   255.0f
    };

    return LayerTestResult<float, 2>(actualOutput,
                                     expectedOutput,
                                     outputHandle->GetShape(),
                                     tensorInfo.GetShape());
}
