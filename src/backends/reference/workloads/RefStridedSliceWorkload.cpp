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
    : RefBaseWorkload(descriptor, info)
{}

void RefStridedSliceWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefStridedSliceWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefStridedSliceWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefStridedSliceWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    DataType inputDataType  = inputInfo.GetDataType();
    DataType outputDataType = outputInfo.GetDataType();

    ARMNN_ASSERT(inputDataType == outputDataType);
    IgnoreUnused(outputDataType);

    StridedSlice(inputInfo,
                 m_Data.m_Parameters,
                 inputs[0]->Map(),
                 outputs[0]->Map(),
                 GetDataTypeSize(inputDataType));
}

} // namespace armnn
