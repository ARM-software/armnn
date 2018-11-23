//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefStridedSliceWorkload.hpp"
#include "StridedSlice.hpp"

#include "RefWorkloadUtils.hpp"
#include "TypeUtils.hpp"

namespace armnn
{

template<armnn::DataType DataType>
void RefStridedSliceWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, GetName() + "_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const T* inputData = GetInputTensorData<T>(0, m_Data);
    T* outputData = GetOutputTensorData<T>(0, m_Data);

    StridedSlice(inputInfo, outputInfo, m_Data.m_Parameters, inputData, outputData);
}

template class RefStridedSliceWorkload<DataType::Float32>;
template class RefStridedSliceWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
