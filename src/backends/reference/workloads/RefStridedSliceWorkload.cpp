//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefStridedSliceWorkload.hpp"
#include "RefWorkloadUtils.hpp"
#include "StridedSlice.hpp"

namespace armnn
{

RefStridedSliceWorkload::RefStridedSliceWorkload(const StridedSliceQueueDescriptor& descriptor,
                                                 const WorkloadInfo& info)
    : BaseWorkload(descriptor, info)
{}

void RefStridedSliceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefStridedSliceWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    DataType inputDataType  = inputInfo.GetDataType();
    DataType outputDataType = outputInfo.GetDataType();

    ARMNN_ASSERT(inputDataType == outputDataType);
    IgnoreUnused(outputDataType);

    StridedSlice(inputInfo,
                 m_Data.m_Parameters,
                 m_Data.m_Inputs[0]->Map(),
                 m_Data.m_Outputs[0]->Map(),
                 GetDataTypeSize(inputDataType));
}

void RefStridedSliceWorkload::ExecuteAsync(WorkingMemDescriptor& descriptor)
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefStridedSliceWorkload_Execute_WorkingMemDescriptor");

    const TensorInfo& inputInfo  = GetTensorInfo(descriptor.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(descriptor.m_Outputs[0]);

    DataType inputDataType  = inputInfo.GetDataType();
    DataType outputDataType = outputInfo.GetDataType();

    ARMNN_ASSERT(inputDataType == outputDataType);
    IgnoreUnused(outputDataType);

    StridedSlice(inputInfo,
                 m_Data.m_Parameters,
                 descriptor.m_Inputs[0]->Map(),
                 descriptor.m_Outputs[0]->Map(),
                 GetDataTypeSize(inputDataType));
}

} // namespace armnn
