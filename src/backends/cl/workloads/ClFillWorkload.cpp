//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClFillWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <cl/ClTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <arm_compute/core/Types.h>

namespace armnn
{
using namespace armcomputetensorutils;

ClFillWorkload::ClFillWorkload(const FillQueueDescriptor& descriptor,
                               const WorkloadInfo& info,
                               const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<FillQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClFillWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClFillWorkload", 1, 1);

    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();
    arm_compute::PixelValue pixelValue = GetPixelValue(output.info(), descriptor.m_Parameters.m_Value);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClFillWorkload_configure");
        m_Layer.configure(clCompileContext, &output, pixelValue);
    }
}

void ClFillWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClFillWorkload_Execute", this->GetGuid());
    RunClFunction(m_Layer, CHECK_LOCATION());
}

} // namespace armnn
