//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/backends/ITensorHandle.hpp>

#include "SampleDynamicAdditionWorkload.hpp"
#include "SampleTensorHandle.hpp"

namespace sdb // sample dynamic backend
{

inline const armnn::TensorInfo& GetTensorInfo(const armnn::ITensorHandle* tensorHandle)
{
    // We know that reference workloads use RefTensorHandles for inputs and outputs
    const SampleTensorHandle* sampleTensorHandle =
        static_cast<const SampleTensorHandle*>(tensorHandle);
    return sampleTensorHandle->GetTensorInfo();
}

const float* GetInputTensorData(unsigned int idx, const armnn::AdditionQueueDescriptor& data)
{
    const armnn::ITensorHandle* tensorHandle = data.m_Inputs[idx];
    return reinterpret_cast<const float*>(tensorHandle->Map());
}

float* GetOutputTensorData(unsigned int idx, const armnn::AdditionQueueDescriptor& data)
{
    armnn::ITensorHandle* tensorHandle = data.m_Outputs[idx];
    return reinterpret_cast<float*>(tensorHandle->Map());
}

SampleDynamicAdditionWorkload::SampleDynamicAdditionWorkload(const armnn::AdditionQueueDescriptor& descriptor,
                                                             const armnn::WorkloadInfo& info)
    : BaseWorkload(descriptor, info)
{}

void SampleDynamicAdditionWorkload::Execute() const
{
    const armnn::TensorInfo& info = GetTensorInfo(m_Data.m_Inputs[0]);
    unsigned int num = info.GetNumElements();

    const float* inputData0 = GetInputTensorData(0, m_Data);
    const float* inputData1 = GetInputTensorData(1, m_Data);
    float* outputData       = GetOutputTensorData(0, m_Data);

    for (unsigned int i = 0; i < num; ++i)
    {
        outputData[i] = inputData0[i] + inputData1[i];
    }
}

} // namespace sdb // sample dynamic backend
