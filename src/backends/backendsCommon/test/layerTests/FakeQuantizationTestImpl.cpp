//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FakeQuantizationTestImpl.hpp"


#include <backendsCommon/CpuTensorHandle.hpp>

#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

LayerTestResult<float, 2> FakeQuantizationTest(
    armnn::IWorkloadFactory& workloadFactory,
    const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
    const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);
    constexpr unsigned int width = 2;
    constexpr unsigned int height = 3;

    const armnn::TensorInfo tensorInfo({height, width },
        armnn::DataType::Float32);

    auto input = MakeTensor<float, 2>(tensorInfo, std::vector<float>({
       -10.0f, -5.0f,
         0.0f,  5.0f,
        10.0f, 10.0f
    }));

    LayerTestResult<float, 2> ret(tensorInfo);

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

    armnn::PassthroughCpuTensorHandle refHandle(tensorInfo, &ret.outputExpected[0][0]);
    armnn::FakeQuantizationQueueDescriptor refData = data;
    armnn::WorkloadInfo refInfo = info;
    SetWorkloadOutput(refData, refInfo, 0, tensorInfo, &refHandle);

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateFakeQuantization(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0]);

    workload->PostAllocationConfigure();
    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0], outputHandle.get());

    ret.outputExpected = MakeTensor<float, 2>(tensorInfo, std::vector<float>({
        0.0f,     63.0f,
        128.0f,   191.0f,
        255.0f,   255.0f
    }));

    return ret;
}
