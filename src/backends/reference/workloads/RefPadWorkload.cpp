//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPadWorkload.hpp"

#include "Pad.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

#include "TypeUtils.hpp"

#include <vector>

namespace armnn
{

template <armnn::DataType DataType>
void RefPadWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefPadWorkload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const T* inputData = GetInputTensorData<T>(0, m_Data);
    T* outputData = GetOutputTensorData<T>(0, m_Data);


    Pad(inputInfo, outputInfo, m_Data.m_Parameters.m_PadList, inputData, outputData);
}

template class RefPadWorkload<DataType::Float32>;
template class RefPadWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn