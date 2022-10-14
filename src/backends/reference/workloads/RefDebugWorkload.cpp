//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
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
    Execute(m_Data.m_Inputs);
}

template<armnn::DataType DataType>
void RefDebugWorkload<DataType>::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs);
}

template<armnn::DataType DataType>
void RefDebugWorkload<DataType>::Execute(std::vector<ITensorHandle*> inputs) const
{
    using T = ResolveType<DataType>;

    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, GetName() + "_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);

    const T* inputData = GetInputTensorData<T>(0, m_Data);
    T* outputData = GetOutputTensorData<T>(0, m_Data);

    if (m_Callback)
    {
        m_Callback(m_Data.m_Guid, m_Data.m_SlotIndex, inputs[0]);
    }
    else
    {
        Debug(inputInfo, inputData, m_Data.m_Guid, m_Data.m_LayerName, m_Data.m_SlotIndex, m_Data.m_LayerOutputToFile);
    }

    std::memcpy(outputData, inputData, inputInfo.GetNumElements()*sizeof(T));
}

template<armnn::DataType DataType>
void RefDebugWorkload<DataType>::RegisterDebugCallback(const DebugCallbackFunction& func)
{
    m_Callback = func;
}

template class RefDebugWorkload<DataType::BFloat16>;
template class RefDebugWorkload<DataType::Float16>;
template class RefDebugWorkload<DataType::Float32>;
template class RefDebugWorkload<DataType::QAsymmU8>;
template class RefDebugWorkload<DataType::QAsymmS8>;
template class RefDebugWorkload<DataType::QSymmS16>;
template class RefDebugWorkload<DataType::QSymmS8>;
template class RefDebugWorkload<DataType::Signed32>;

} // namespace armnn
