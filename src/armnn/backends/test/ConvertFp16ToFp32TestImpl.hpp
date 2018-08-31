//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/TypesUtils.hpp>

#include <backends/WorkloadInfo.hpp>
#include <backends/CpuTensorHandle.hpp>

#include <test/TensorHelpers.hpp>

#include <Half.hpp>

LayerTestResult<float, 4> SimpleConvertFp16ToFp32Test(armnn::IWorkloadFactory& workloadFactory)
{
    using namespace half_float::literal;

    const armnn::TensorInfo inputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float16);
    const armnn::TensorInfo outputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float32);

    auto input = MakeTensor<armnn::Half, 4>(inputTensorInfo,
        { -37.5_h, -15.2_h, -8.76_h, -2.0_h, -1.5_h, -1.3_h, -0.5_h, -0.4_h, 0.0_h,
          1.0_h, 0.4_h, 0.5_h, 1.3_h, 1.5_h, 2.0_h, 8.76_h, 15.2_h, 37.5_h });

    LayerTestResult<float, 4> ret(outputTensorInfo);
    ret.outputExpected = MakeTensor<float, 4>(outputTensorInfo,
        { -37.5f, -15.2f, -8.76f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f, 0.0f,
          1.0f, 0.4f, 0.5f, 1.3f, 1.5f, 2.0f, 8.76f, 15.2f, 37.5f });

    std::unique_ptr<armnn::ITensorHandle> inputHandle = workloadFactory.CreateTensorHandle(inputTensorInfo);
    std::unique_ptr<armnn::ITensorHandle> outputHandle = workloadFactory.CreateTensorHandle(outputTensorInfo);

    armnn::ConvertFp16ToFp32QueueDescriptor data;
    armnn::WorkloadInfo info;
    AddInputToWorkload(data, info, inputTensorInfo, inputHandle.get());
    AddOutputToWorkload(data, info, outputTensorInfo, outputHandle.get());

    std::unique_ptr<armnn::IWorkload> workload = workloadFactory.CreateConvertFp16ToFp32(data, info);

    inputHandle->Allocate();
    outputHandle->Allocate();

    CopyDataToITensorHandle(inputHandle.get(), &input[0][0][0][0]);

    workload->Execute();

    CopyDataFromITensorHandle(&ret.output[0][0][0][0], outputHandle.get());

    return ret;
}
