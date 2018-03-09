//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <armnn/Tensor.hpp>
#include <backends/WorkloadInfo.hpp>

namespace armnn
{
class ITensorHandle;
}

template <typename QueueDescriptor>
void AddInputToWorkload(QueueDescriptor& descriptor,
    armnn::WorkloadInfo& info,
    const armnn::TensorInfo& tensorInfo,
    armnn::ITensorHandle* tensorHandle)
{
    descriptor.m_Inputs.push_back(tensorHandle);
    info.m_InputTensorInfos.push_back(tensorInfo);
}

template <typename QueueDescriptor>
void AddOutputToWorkload(QueueDescriptor& descriptor,
    armnn::WorkloadInfo& info,
    const armnn::TensorInfo& tensorInfo,
    armnn::ITensorHandle* tensorHandle)
{
    descriptor.m_Outputs.push_back(tensorHandle);
    info.m_OutputTensorInfos.push_back(tensorInfo);
}

template <typename QueueDescriptor>
void SetWorkloadInput(QueueDescriptor& descriptor,
    armnn::WorkloadInfo& info,
    unsigned int index,
    const armnn::TensorInfo& tensorInfo,
    armnn::ITensorHandle* tensorHandle)
{
    descriptor.m_Inputs[index] = tensorHandle;
    info.m_InputTensorInfos[index] = tensorInfo;
}

template <typename QueueDescriptor>
void SetWorkloadOutput(QueueDescriptor& descriptor,
    armnn::WorkloadInfo& info,
    unsigned int index,
    const armnn::TensorInfo& tensorInfo,
    armnn::ITensorHandle* tensorHandle)
{
    descriptor.m_Outputs[index] = tensorHandle;
    info.m_OutputTensorInfos[index] = tensorInfo;
}