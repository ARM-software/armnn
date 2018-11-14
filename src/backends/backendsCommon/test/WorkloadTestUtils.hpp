//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>

#include <backendsCommon/IBackendInternal.hpp>
#include <backendsCommon/IMemoryManager.hpp>
#include <backendsCommon/Workload.hpp>
#include <backendsCommon/WorkloadInfo.hpp>

namespace armnn
{
class ITensorHandle;
} // namespace armnn

namespace
{

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

inline void ExecuteWorkload(armnn::IWorkload& workload,
                            const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
                            bool memoryManagementRequested = true)
{
    const bool manageMemory = memoryManager && memoryManagementRequested;

    // Acquire working memory (if needed)
    if (manageMemory)
    {
        memoryManager->Acquire();
    }

    // Execute the workload
    workload.Execute();

    // Release working memory (if needed)
    if (manageMemory)
    {
        memoryManager->Release();
    }
}

} // anonymous namespace
