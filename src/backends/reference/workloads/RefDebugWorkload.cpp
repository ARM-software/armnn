//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDebugWorkload.hpp"
#include "Debug.hpp"
#include "RefWorkloadUtils.hpp"

#include <ResolveType.hpp>

#include <cstring>

namespace armnn
{

template<armnn::DataType DataType>
void RefDebugWorkload<DataType>::Execute() const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, GetName() + "_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);

    const T* inputData = GetInputTensorData<T>(0, m_Data);
    T* outputData = GetOutputTensorData<T>(0, m_Data);

    if (m_Callback)
    {
        m_Callback(m_Data.m_Guid, m_Data.m_SlotIndex, m_Data.m_Inputs[0]);
    }
    else
    {
        Debug(inputInfo, inputData, m_Data.m_Guid, m_Data.m_LayerName, m_Data.m_SlotIndex);
    }

    std::memcpy(outputData, inputData, inputInfo.GetNumElements()*sizeof(T));
}

template<armnn::DataType DataType>
void RefDebugWorkload<DataType>::RegisterDebugCallback(const DebugCallbackFunction& func)
{
    m_Callback = func;
}

template class RefDebugWorkload<DataType::Float32>;
template class RefDebugWorkload<DataType::QuantisedAsymm8>;
template class RefDebugWorkload<DataType::QuantisedSymm16>;

} // namespace armnn
