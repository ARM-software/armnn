//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonFillWorkload.hpp"

#include <neon/NeonTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEFill.h>

#include "NeonWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

NeonFillWorkload::NeonFillWorkload(const FillQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<FillQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonFillWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonFillWorkload", 1, 1);

    arm_compute::ITensor& output = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    arm_compute::PixelValue pixelValue = GetPixelValue(output.info(), descriptor.m_Parameters.m_Value);

    auto layer = std::make_unique<arm_compute::NEFill>();
    layer->configure(&output, pixelValue);
    m_Layer.reset(layer.release());
}

void NeonFillWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonFillWorkload_Execute", this->GetGuid());
    m_Layer->run();
}

} // namespace armnn
