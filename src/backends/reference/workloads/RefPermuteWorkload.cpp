//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPermuteWorkload.hpp"
#include "RefWorkloadUtils.hpp"

#include <Permute.hpp>
#include <ResolveType.hpp>

namespace armnn
{

template <armnn::DataType DataType>
void RefPermuteWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, GetName() + "_Execute");

    const ITensorHandle*     src      = m_Data.m_Inputs[0];
    ITensorHandle*           dst      = m_Data.m_Outputs[0];
    const PermutationVector& mappings = m_Data.m_Parameters.m_DimMappings;

    armnnUtils::Permute(GetTensorInfo(dst).GetShape(), mappings,
                        src->Map(), dst->Map(), sizeof(T));
}

template class RefPermuteWorkload<DataType::Float16>;
template class RefPermuteWorkload<DataType::Float32>;
template class RefPermuteWorkload<DataType::QuantisedAsymm8>;
template class RefPermuteWorkload<DataType::QuantisedSymm16>;

} //namespace armnn
