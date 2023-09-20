//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnnTestUtils/LayerTestResult.hpp>

#include <armnnUtils/QuantizeHelper.hpp>
#include <ResolveType.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<LayerTestResult<T,4>> AddMulAddTest(armnn::IWorkloadFactory& workloadFactory,
                                                const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                                                const armnn::ITensorHandleFactory& tensorHandleFactory,
                                                bool addOutput)
{
    using namespace armnn;
    IgnoreUnused(memoryManager);

    TensorInfo input0TensorInfo({ 1, 2, 2, 3 }, ArmnnType);
    TensorInfo input1TensorInfo({ 1, 2, 2, 3 }, ArmnnType);
    TensorInfo mulInput1TensorInfo({ 3 }, ArmnnType);
    TensorInfo addInput1TensorInfo({ 3 }, ArmnnType);

    TensorInfo output0TensorInfo({ 1, 2, 2, 3 }, ArmnnType);
    TensorInfo output1TensorInfo({ 1, 2, 2, 3 }, ArmnnType);

    if (IsQuantizedType<T>())
    {
        input0TensorInfo.SetQuantizationScale(0.25f);
        input0TensorInfo.SetQuantizationOffset(10);
        input1TensorInfo.SetQuantizationScale(0.25f);
        input1TensorInfo.SetQuantizationOffset(11);
        mulInput1TensorInfo.SetQuantizationScale(0.25f);
        mulInput1TensorInfo.SetQuantizationOffset(12);
        addInput1TensorInfo.SetQuantizationScale(0.25f);
        addInput1TensorInfo.SetQuantizationOffset(13);

        output0TensorInfo.SetQuantizationScale(0.5f);
        output0TensorInfo.SetQuantizationOffset(14);
        output1TensorInfo.SetQuantizationScale(0.5f);
        output1TensorInfo.SetQuantizationOffset(15);
    }

    std::vector<float> input0Data
    {
         0.0f,  0.0f,  0.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -2.0f, -2.0f, -2.0f
    };
    std::vector<float> input1Data
    {
         0.0f,  0.0f,  0.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -2.0f, -2.0f, -2.0f
    };
    std::vector<float> mulInput1Data
    {
         2.0f, 1.0f, 1.0f
    };
    std::vector<float> addInput1Data
    {
         3.0f, 0.0f, 0.0f
    };
    std::vector<float> output0ExpectedData =
    {
         0.0f,  0.0f,  0.0f,
         2.0f,  2.0f,  2.0f,
        -2.0f, -2.0f, -2.0f,
        -4.0f, -4.0f, -4.0f
    };

    std::vector<float> output1ExpectedData =
    {
         3.0f,  0.0f,  0.0f,
         7.0f,  2.0f,  2.0f,
        -1.0f, -2.0f, -2.0f,
        -5.0f, -4.0f, -4.0f
    };

    std::vector<T> input0 = armnnUtils::QuantizedVector<T>(input0Data,
                                                           input0TensorInfo.GetQuantizationScale(),
                                                           input0TensorInfo.GetQuantizationOffset());

    std::vector<T> input1 = armnnUtils::QuantizedVector<T>(input1Data,
                                                           input1TensorInfo.GetQuantizationScale(),
                                                           input1TensorInfo.GetQuantizationOffset());

    std::vector<T> mulInput1 = armnnUtils::QuantizedVector<T>(mulInput1Data,
                                                              mulInput1TensorInfo.GetQuantizationScale(),
                                                              mulInput1TensorInfo.GetQuantizationOffset());

    std::vector<T> addInput1 = armnnUtils::QuantizedVector<T>(addInput1Data,
                                                              addInput1TensorInfo.GetQuantizationScale(),
                                                              addInput1TensorInfo.GetQuantizationOffset());

    std::vector<T> output0Expected = armnnUtils::QuantizedVector<T>(output0ExpectedData,
                                                                    output0TensorInfo.GetQuantizationScale(),
                                                                    output0TensorInfo.GetQuantizationOffset());

    std::vector<T> output1Expected = armnnUtils::QuantizedVector<T>(output1ExpectedData,
                                                                    output1TensorInfo.GetQuantizationScale(),
                                                                    output1TensorInfo.GetQuantizationOffset());

    std::vector<T> output0Actual(output0TensorInfo.GetNumElements());
    std::vector<T> output1Actual(output1TensorInfo.GetNumElements());

    std::unique_ptr<ITensorHandle> input0Handle = tensorHandleFactory.CreateTensorHandle(input0TensorInfo);
    std::unique_ptr<ITensorHandle> input1Handle = tensorHandleFactory.CreateTensorHandle(input1TensorInfo);
    std::unique_ptr<ITensorHandle> mulInput1Handle = tensorHandleFactory.CreateTensorHandle(mulInput1TensorInfo);
    std::unique_ptr<ITensorHandle> addInput1Handle = tensorHandleFactory.CreateTensorHandle(addInput1TensorInfo);
    std::unique_ptr<ITensorHandle> output0Handle = tensorHandleFactory.CreateTensorHandle(output0TensorInfo);
    std::unique_ptr<ITensorHandle> output1Handle = tensorHandleFactory.CreateTensorHandle(output1TensorInfo);

    uint32_t numOutputs = addOutput ? 2 : 1;
    FusedDescriptor descriptor(4, numOutputs, FusedKernelType::AddMulAdd);
    FusedQueueDescriptor fusedQueueDescriptor;
    fusedQueueDescriptor.m_Parameters = descriptor;
    WorkloadInfo info;
    AddInputToWorkload (fusedQueueDescriptor, info, input0TensorInfo, input0Handle.get());
    AddInputToWorkload (fusedQueueDescriptor, info, input1TensorInfo, input1Handle.get());
    AddInputToWorkload (fusedQueueDescriptor, info, mulInput1TensorInfo, mulInput1Handle.get());
    AddInputToWorkload (fusedQueueDescriptor, info, addInput1TensorInfo, addInput1Handle.get());
    if (addOutput)
    {
        AddOutputToWorkload(fusedQueueDescriptor, info, output0TensorInfo, output0Handle.get());
    }
    AddOutputToWorkload(fusedQueueDescriptor, info, output1TensorInfo, output1Handle.get());

    if (addOutput)
    {
        AddOutputToWorkload(fusedQueueDescriptor, info, output0TensorInfo, output0Handle.get());
    }
    AddOutputToWorkload(fusedQueueDescriptor, info, output1TensorInfo, output1Handle.get());

    std::unique_ptr<IWorkload> workload = workloadFactory.CreateWorkload(LayerType::Fused,
                                                                         fusedQueueDescriptor,
                                                                         info);

    input0Handle->Allocate();
    input1Handle->Allocate();
    mulInput1Handle->Allocate();
    addInput1Handle->Allocate();
    if (addOutput)
    {
        output0Handle->Allocate();
    }
    output1Handle->Allocate();

    CopyDataToITensorHandle(input0Handle.get(), input0.data());
    CopyDataToITensorHandle(input1Handle.get(), input1.data());
    CopyDataToITensorHandle(mulInput1Handle.get(), mulInput1.data());
    CopyDataToITensorHandle(addInput1Handle.get(), addInput1.data());

    workload->Execute();

    CopyDataFromITensorHandle(output1Actual.data(), output1Handle.get());
    LayerTestResult<T,4> ret1(output1Actual,
                              output1Expected,
                              output1Handle->GetShape(),
                              output1TensorInfo.GetShape());

    std::vector<LayerTestResult<T,4>> ret = {ret1};

    if (addOutput)
    {
        CopyDataFromITensorHandle(output0Actual.data(), output0Handle.get());
        LayerTestResult<T,4> ret0(output0Actual,
                                  output0Expected,
                                  output0Handle->GetShape(),
                                  output0TensorInfo.GetShape());
        ret = {ret0, ret1};
    }
    return ret;
}
