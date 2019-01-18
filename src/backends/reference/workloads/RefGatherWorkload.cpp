//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefGatherWorkload.hpp"

#include "Gather.hpp"
#include "Profiling.hpp"
#include "RefWorkloadUtils.hpp"
#include "TypeUtils.hpp"

namespace armnn
{

template <armnn::DataType DataType>
void RefGatherWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefGatherWorkload_Execute");

    const TensorInfo& inputInfo0 = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& inputInfo1 = GetTensorInfo(m_Data.m_Inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const T* paramsData = GetInputTensorData<T>(0, m_Data);
    const int32_t* indicesData = GetInputTensorData<int32_t>(1, m_Data);
    T* outputData = GetOutputTensorData<T>(0, m_Data);

    Gather(inputInfo0, inputInfo1, outputInfo, paramsData, indicesData, outputData);
}

template class RefGatherWorkload<DataType::Float32>;
template class RefGatherWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
