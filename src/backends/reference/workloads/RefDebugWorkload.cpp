//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDebugWorkload.hpp"
#include "Debug.hpp"
#include "RefWorkloadUtils.hpp"

#include <TypeUtils.hpp>

namespace armnn
{

template<armnn::DataType DataType>
void RefDebugWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, GetName() + "_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const T* inputData = GetInputTensorData<T>(0, m_Data);
    T* outputData = GetOutputTensorData<T>(0, m_Data);

    Debug(inputInfo, outputInfo, m_Data.m_Parameters, inputData, outputData);
}

template class RefDebugWorkload<DataType::Float32>;
template class RefDebugWorkload<DataType::QuantisedAsymm8>;

} // namespace armnn
