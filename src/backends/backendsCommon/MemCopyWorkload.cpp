//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ResolveType.hpp>

#include "WorkloadUtils.hpp"

#include <armnn/backends/MemCopyWorkload.hpp>
#include <armnn/backends/TensorHandle.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <cstring>

namespace armnn
{

namespace
{

template <typename SrcTensorHandleType, typename DstTensorHandleType>
void GatherTensorHandlePairs(const MemCopyQueueDescriptor& descriptor,
                             std::vector<std::pair<SrcTensorHandleType*, DstTensorHandleType*>>& tensorHandlePairs)
{
    const unsigned int numInputs = static_cast<unsigned int>(descriptor.m_Inputs.size());
    tensorHandlePairs.reserve(numInputs);

    for (unsigned int i = 0; i < numInputs; ++i)
    {
        SrcTensorHandleType* const srcTensorHandle = PolymorphicDowncast<SrcTensorHandleType*>(
            descriptor.m_Inputs[i]);
        DstTensorHandleType* const dstTensorHandle = PolymorphicDowncast<DstTensorHandleType*>(
            descriptor.m_Outputs[i]);

        tensorHandlePairs.emplace_back(srcTensorHandle, dstTensorHandle);
    }
}

} //namespace


CopyMemGenericWorkload::CopyMemGenericWorkload(const MemCopyQueueDescriptor& descriptor,
                                               const WorkloadInfo& info)
    : BaseWorkload<MemCopyQueueDescriptor>(descriptor, info)
{
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

void CopyMemGenericWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "CopyMemGeneric_Execute");

    auto copyFunc = [](void* dst, const void* src, size_t size)
        {
            memcpy(dst, src, size);
        };

    for (const auto& pair : m_TensorHandlePairs)
    {
        CopyTensorContentsGeneric(pair.first, pair.second, copyFunc);
    }
}

void CopyMemGenericWorkload::ExecuteAsync(WorkingMemDescriptor& descriptor)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "CopyMemGeneric_Execute_WorkingMemDescriptor");
    std::vector<TensorHandlePair> tensorHandlePairs;
    GatherTensorHandlePairs(descriptor, tensorHandlePairs);

    auto copyFunc = [](void* dst, const void* src, size_t size)
    {
        memcpy(dst, src, size);
    };

    for (const auto& pair : tensorHandlePairs)
    {
        CopyTensorContentsGeneric(pair.first, pair.second, copyFunc);
    }
}

} //namespace armnn
