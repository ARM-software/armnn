//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerTestResult.hpp"

#include <QuantizeHelper.hpp>
#include <ResolveType.hpp>


#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadFactoryHelper.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <test/TensorHelpers.hpp>

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
LayerTestResult<T, 4> PreluTest(
        armnn::IWorkloadFactory& workloadFactory,
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ITensorHandleFactory& tensorHandleFactory)
{
    IgnoreUnused(memoryManager);

    armnn::TensorInfo inputTensorInfo ({ 1, 2, 2, 3 }, ArmnnType);
    armnn::TensorInfo alphaTensorInfo ({ 1, 1, 1, 3 }, ArmnnType);
    armnn::TensorInfo outputTensorInfo({ 1, 2, 2, 3 }, ArmnnType);

    if (armnn::IsQuantizedType<T>())
    {
        inputTensorInfo.SetQuantizationScale(0.25f);
        inputTensorInfo.SetQuantizationOffset(128);
        alphaTensorInfo.SetQuantizationScale(0.25f);
        alphaTensorInfo.SetQuantizationOffset(50);
        outputTensorInfo.SetQuantizationScale(0.5f);
        outputTensorInfo.SetQuantizationOffset(120);
    }

    std::vector<float> inputData
    {
        // Expected quantized values:
        // 128, 128, 128, 132, 132, 132, 124, 124, 124, 120, 120, 120
        0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -2.0f, -2.0f, -2.0f
    };
    std::vector<float> alphaData
    {
        // Expected quantized values:
        // 50, 54, 58
        0.0f, 1.0f, 2.0f
    };
    std::vector<float> outputExpectedData =
    {
        // Expected quantized values:
        // 20, 120, 120, 122, 122, 122, 120, 118, 116, 120, 116, 112
       0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, -1.0f, -2.0f, 0.0f, -2.0f, -4.0f
    };

    auto input = MakeTensor<T, 4>(inputTensorInfo,
                                  armnnUtils::QuantizedVector<T>(inputData,
                                                                 inputTensorInfo.GetQuantizationScale(),
                                                                 inputTensorInfo.GetQuantizationOffset()));

    auto alpha = MakeTensor<T, 4>(alphaTensorInfo,
                                  armnnUtils::QuantizedVector<T>(alphaData,
                                                                 alphaTensorInfo.GetQuantizationScale(),
                                                                 alphaTensorInfo.GetQuantizationOffset()));

    LayerTestResult<T, 4> result(outputTensorInfo);
    result.outputExpected =
        MakeTensor<T, 4>(outputTensorInfo,
                         armnnUtils::QuantizedVector<T>(outputExpectedData,
                                                        outputTensorInfo.GetQuantizationScale(),
                                                        outputTensorInfo.GetQuantizationOffset()));

    std::unique_ptr <armnn::ITensorHandle> inputHandle  = tensorHandleFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> alphaHandle  = tensorHandleFactory.CreateTensorHandle(alphaTensorInfo);
    std::unique_ptr <armnn::ITensorHandle> outputHandle = tensorHandleFactory.CreateTensorHandle(outputTensorInfo);

    armnn::PreluQueueDescriptor descriptor;
    armnn::WorkloadInfo info;
    AddInputToWorkload (descriptor, info, inputTensorInfo,  inputHandle.get());
    AddInputToWorkload (descriptor, info, alphaTensorInfo,  alphaHandle.get());
    AddOutputToWorkload(descriptor, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreatePrelu(descriptor, info);

    inputHandle->Allocate();
    alphaHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);
    CopyDataToITensorHandle(alphaHandle.get(), &alpha[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&result.output[0][0][0][0], outputHandle.get());

    return result;
}
