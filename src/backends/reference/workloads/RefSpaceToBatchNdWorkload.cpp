//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefSpaceToBatchNdWorkload.hpp"
#include "SpaceToBatchNd.hpp"

#include "RefWorkloadUtils.hpp"
#include "TypeUtils.hpp"

namespace armnn
{

template<armnn::DataType DataType>
void RefSpaceToBatchNdWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, GetName() + "_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    const T* inputData = GetInputTensorData<T>(0, m_Data);
    T* outputData = GetOutputTensorData<T>(0, m_Data);

    SpaceToBatchNd(inputInfo, outputInfo, m_Data.m_Parameters, inputData, outputData);
}

template class RefSpaceToBatchNdWorkload<DataType::Float32>;
template class RefSpaceToBatchNdWorkload<DataType::QuantisedAsymm8>;

} //namespace armnn
