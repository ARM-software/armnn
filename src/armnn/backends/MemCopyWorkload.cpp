//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "MemCopyWorkload.hpp"
#include "backends/CpuTensorHandle.hpp"

#if ARMCOMPUTECL_ENABLED
#include "backends/ClTensorHandle.hpp"
#endif

#if ARMCOMPUTENEON_ENABLED
#include "backends/NeonTensorHandle.hpp"
#endif

#include <cstring>
#include <boost/cast.hpp>

namespace armnn
{

namespace
{

template <typename SrcTensorHandleType, typename DstTensorHandleType>
void GatherTensorHandlePairs(const MemCopyQueueDescriptor& descriptor,
                             std::vector<std::pair<SrcTensorHandleType*, DstTensorHandleType*>>& tensorHandlePairs)
{
    const unsigned int numInputs = boost::numeric_cast<unsigned int>(descriptor.m_Inputs.size());
    tensorHandlePairs.reserve(numInputs);

    for (unsigned int i = 0; i < numInputs; ++i)
    {
        SrcTensorHandleType* const srcTensorHandle = boost::polymorphic_downcast<SrcTensorHandleType*>(
            descriptor.m_Inputs[i]);
        DstTensorHandleType* const dstTensorHandle = boost::polymorphic_downcast<DstTensorHandleType*>(
            descriptor.m_Outputs[i]);

        tensorHandlePairs.emplace_back(srcTensorHandle, dstTensorHandle);
    }
}

void CopyFromCpuToCpu(const ConstCpuTensorHandle& srcHandle, CpuTensorHandle& dstHandle)
{
    const unsigned int numBytes = srcHandle.GetTensorInfo().GetNumBytes();
    const void* const input = srcHandle.GetConstTensor<void>();
    void* const output = dstHandle.GetTensor<void>();
    std::memcpy(output, input, numBytes);
}

#if ARMCOMPUTECL_ENABLED || ARMCOMPUTENEON_ENABLED

#include "backends/ArmComputeTensorUtils.hpp"

template <armnn::DataType DataType>
void CopyFromCpuToAclBackend(const ConstCpuTensorHandle& srcHandle, arm_compute::ITensor& dstAclTensor)
{
    using T = ResolveType<DataType>;
    armnn::armcomputetensorutils::CopyArmComputeITensorData(srcHandle.GetConstTensor<T>(), dstAclTensor);
}

template <armnn::DataType DataType>
void CopyFromAclBackendToCpu(const arm_compute::ITensor& srcAclTensor, CpuTensorHandle& dstHandle)
{
    using T = ResolveType<DataType>;
    armnn::armcomputetensorutils::CopyArmComputeITensorData(srcAclTensor, dstHandle.GetTensor<T>());
}

#endif // ARMCOMPUTECL_ENABLED || ARMCOMPUTENEON_ENABLED

}

template <armnn::DataType DataType>
CopyFromCpuToCpuWorkload<DataType>::CopyFromCpuToCpuWorkload(const MemCopyQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
    : TypedWorkload<MemCopyQueueDescriptor, DataType>(descriptor, info)
{
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

template <armnn::DataType DataType>
void CopyFromCpuToCpuWorkload<DataType>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "CopyFromCpuToCpuWorkload_Execute");

    for (const auto& pair : m_TensorHandlePairs)
    {
        CopyFromCpuToCpu(*pair.first, *pair.second);
    }
}

template class CopyFromCpuToCpuWorkload<DataType::Float32>;
template class CopyFromCpuToCpuWorkload<DataType::QuantisedAsymm8>;

#if ARMCOMPUTECL_ENABLED

template <armnn::DataType DataType>
CopyFromCpuToClWorkload<DataType>::CopyFromCpuToClWorkload(const MemCopyQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info)
    : TypedWorkload<MemCopyQueueDescriptor, DataType>(descriptor, info)
{
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

template <armnn::DataType DataType>
void CopyFromCpuToClWorkload<DataType>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "CopyFromCpuToClWorkload_Execute");

    for (const auto& pair : m_TensorHandlePairs)
    {
        IClTensorHandle& handle = *pair.second;

        handle.Map(true);
        CopyFromCpuToAclBackend<DataType>(*pair.first, handle.GetTensor());
        handle.UnMap();
    }
}

template class CopyFromCpuToClWorkload<DataType::Float32>;
template class CopyFromCpuToClWorkload<DataType::QuantisedAsymm8>;


template <armnn::DataType DataType>
CopyFromClToCpuWorkload<DataType>::CopyFromClToCpuWorkload(const MemCopyQueueDescriptor& descriptor,
                                                           const WorkloadInfo& info)
    : TypedWorkload<MemCopyQueueDescriptor, DataType>(descriptor, info)
{
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

template <armnn::DataType DataType>
void CopyFromClToCpuWorkload<DataType>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "CopyFromClToCpuWorkload_Execute");

    for (const auto& pair : m_TensorHandlePairs)
    {
        IClTensorHandle& handle = *pair.first;

        handle.Map(true);
        CopyFromAclBackendToCpu<DataType>(handle.GetTensor(), *pair.second);
        handle.UnMap();
    }
}

template class CopyFromClToCpuWorkload<DataType::Float32>;
template class CopyFromClToCpuWorkload<DataType::QuantisedAsymm8>;

#endif // ARMCOMPUTECL_ENABLED

#if ARMCOMPUTENEON_ENABLED

template <armnn::DataType DataType>
CopyFromCpuToNeonWorkload<DataType>::CopyFromCpuToNeonWorkload(const MemCopyQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info)
    : TypedWorkload<MemCopyQueueDescriptor, DataType>(descriptor, info)
{
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

template <armnn::DataType DataType>
void CopyFromCpuToNeonWorkload<DataType>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "CopyFromCpuToNeonWorkload_Execute");

    for (const auto& pair : m_TensorHandlePairs)
    {
        CopyFromCpuToAclBackend<DataType>(*pair.first, pair.second->GetTensor());
    }
}

template class CopyFromCpuToNeonWorkload<DataType::Float32>;
template class CopyFromCpuToNeonWorkload<DataType::QuantisedAsymm8>;

template <armnn::DataType DataType>
CopyFromNeonToCpuWorkload<DataType>::CopyFromNeonToCpuWorkload(const MemCopyQueueDescriptor& descriptor,
                                                               const WorkloadInfo& info)
    : TypedWorkload<MemCopyQueueDescriptor, DataType>(descriptor, info)
{
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

template <armnn::DataType DataType>
void CopyFromNeonToCpuWorkload<DataType>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "CopyFromNeonToCpuWorkload_Execute");

    for (const auto& pair : m_TensorHandlePairs)
    {
        CopyFromAclBackendToCpu<DataType>(pair.first->GetTensor(), *pair.second);
    }
}

template class CopyFromNeonToCpuWorkload<DataType::Float32>;
template class CopyFromNeonToCpuWorkload<DataType::QuantisedAsymm8>;

#endif // ARMCOMPUTENEON_ENABLED

#if ARMCOMPUTECL_ENABLED && ARMCOMPUTENEON_ENABLED

template <armnn::DataType DataType>
CopyFromNeonToClWorkload<DataType>::CopyFromNeonToClWorkload(const MemCopyQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
    : TypedWorkload<MemCopyQueueDescriptor, DataType>(descriptor, info)
{
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

template <armnn::DataType DataType>
void CopyFromNeonToClWorkload<DataType>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "CopyFromNeonToClWorkload_Execute");

    for (const auto& pair : m_TensorHandlePairs)
    {
        IClTensorHandle& handle = *pair.second;

        handle.Map(true);
        handle.GetTensor().copy_from(pair.first->GetTensor());
        handle.UnMap();
    }
}

template class CopyFromNeonToClWorkload<DataType::Float32>;
template class CopyFromNeonToClWorkload<DataType::QuantisedAsymm8>;

template <armnn::DataType DataType>
CopyFromClToNeonWorkload<DataType>::CopyFromClToNeonWorkload(const MemCopyQueueDescriptor& descriptor,
                                                             const WorkloadInfo& info)
    : TypedWorkload<MemCopyQueueDescriptor, DataType>(descriptor, info)
{
    GatherTensorHandlePairs(descriptor, m_TensorHandlePairs);
}

template <armnn::DataType DataType>
void CopyFromClToNeonWorkload<DataType>::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::GpuAcc, "CopyFromClToNeonWorkload_Execute");

    for (const auto& pair : m_TensorHandlePairs)
    {
        IClTensorHandle& handle = *pair.first;

        handle.Map(true);
        pair.second->GetTensor().copy_from(handle.GetTensor());
        handle.UnMap();
    }
}

template class CopyFromClToNeonWorkload<DataType::Float32>;
template class CopyFromClToNeonWorkload<DataType::QuantisedAsymm8>;

#endif // ARMCOMPUTECL_ENABLED && ARMCOMPUTENEON_ENABLED

}
