//
// Copyright © 2018-2024 Arm Ltd and Contributors. All rights reserved.
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

void RefStridedSliceWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefStridedSliceWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);

    DataType inputDataType  = inputInfo.GetDataType();

    StridedSlice(inputInfo,
                 m_Data.m_Parameters,
                 inputs[0]->Map(),
                 outputs[0]->Map(),
                 GetDataTypeSize(inputDataType));
}

} // namespace armnn
