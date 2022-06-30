//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPermuteWorkload.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnnUtils/Permute.hpp>

#include <ResolveType.hpp>

namespace armnn
{

template <armnn::DataType DataType>
void RefPermuteWorkload<DataType>::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

template <armnn::DataType DataType>
void RefPermuteWorkload<DataType>::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

template <armnn::DataType DataType>
void RefPermuteWorkload<DataType>::Execute(std::vector<ITensorHandle*> inputs,
                                           std::vector<ITensorHandle*> outputs) const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, GetName() + "_Execute");

    const ITensorHandle*     src      = inputs[0];
    ITensorHandle*           dst      = outputs[0];
    const PermutationVector& mappings = m_Data.m_Parameters.m_DimMappings;

    armnnUtils::Permute(GetTensorInfo(dst).GetShape(), mappings,
                        src->Map(), dst->Map(), sizeof(T));
}

template class RefPermuteWorkload<DataType::BFloat16>;
template class RefPermuteWorkload<DataType::Float16>;
template class RefPermuteWorkload<DataType::Float32>;
template class RefPermuteWorkload<DataType::QAsymmS8>;
template class RefPermuteWorkload<DataType::QAsymmU8>;
template class RefPermuteWorkload<DataType::QSymmS16>;

} //namespace armnn
