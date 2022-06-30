//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefTransposeWorkload.hpp"
#include "RefWorkloadUtils.hpp"

#include <armnnUtils/Transpose.hpp>

#include <ResolveType.hpp>

namespace armnn
{

template <armnn::DataType DataType>
void RefTransposeWorkload<DataType>::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

template <armnn::DataType DataType>
void RefTransposeWorkload<DataType>::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

template <armnn::DataType DataType>
void RefTransposeWorkload<DataType>::Execute(std::vector<ITensorHandle*> inputs,
                                             std::vector<ITensorHandle*> outputs) const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, GetName() + "_Execute");

    const ITensorHandle*     src      = inputs[0];
    ITensorHandle*           dst      = outputs[0];
    const PermutationVector& mappings = m_Data.m_Parameters.m_DimMappings;

    armnnUtils::Transpose(GetTensorInfo(src).GetShape(), mappings, src->Map(), dst->Map(), sizeof(T));
}

template class RefTransposeWorkload<DataType::BFloat16>;
template class RefTransposeWorkload<DataType::Float16>;
template class RefTransposeWorkload<DataType::Float32>;
template class RefTransposeWorkload<DataType::QAsymmS8>;
template class RefTransposeWorkload<DataType::QAsymmU8>;
template class RefTransposeWorkload<DataType::QSymmS16>;

} //namespace armnn
