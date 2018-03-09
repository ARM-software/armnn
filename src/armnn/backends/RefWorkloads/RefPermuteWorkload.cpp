//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefPermuteWorkload.hpp"
#include "RefWorkloadUtils.hpp"

#include <Permute.hpp>

namespace armnn
{

template <armnn::DataType DataType>
void RefPermuteWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, GetName() + "_Execute");

    const ITensorHandle*     src      = m_Data.m_Inputs[0];
    const ITensorHandle*     dst      = m_Data.m_Outputs[0];
    const PermutationVector& mappings = m_Data.m_Parameters.m_DimMappings;

    armnnUtils::Permute(GetTensorInfo(dst).GetShape(), mappings, GetConstCpuData<T>(src), GetCpuData<T>(dst));
}

template class RefPermuteWorkload<DataType::Float32>;
template class RefPermuteWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
