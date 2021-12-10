//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Tensor.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <armnn/backends/IMemoryManager.hpp>
#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadInfo.hpp>

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

    // Perform PostAllocationConfiguration
    workload.PostAllocationConfigure();

    // Execute the workload
    workload.Execute();

    // Release working memory (if needed)
    if (manageMemory)
    {
        memoryManager->Release();
    }
}

inline armnn::Optional<armnn::DataType> GetBiasTypeFromWeightsType(armnn::Optional<armnn::DataType> weightsType)
{
    if (!weightsType)
    {
        return weightsType;
    }

    switch(weightsType.value())
    {
        case armnn::DataType::BFloat16:
        case armnn::DataType::Float16:
        case armnn::DataType::Float32:
            return weightsType;
        case armnn::DataType::QAsymmS8:
        case armnn::DataType::QAsymmU8:
        case armnn::DataType::QSymmS8:
        case armnn::DataType::QSymmS16:
            return armnn::DataType::Signed32;
        default:
            ARMNN_ASSERT_MSG(false, "GetBiasTypeFromWeightsType(): Unsupported data type.");
    }
    return armnn::EmptyOptional();
}

} // anonymous namespace
