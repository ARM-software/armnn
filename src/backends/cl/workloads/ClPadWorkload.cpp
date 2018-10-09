//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClPadWorkload.hpp"

#include <backends/cl/ClTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>
#include <arm_compute/core/Types.h>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

ClPadWorkload::ClPadWorkload(const PadQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<PadQueueDescriptor>(descriptor, info)
{
    this->m_Data.ValidateInputsOutputs("ClPadWorkload", 1, 1);

    arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(this->m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(this->m_Data.m_Outputs[0])->GetTensor();
    arm_compute::PaddingList padList = static_cast<arm_compute::PaddingList>(descriptor.m_Parameters.m_PadList);

    m_Layer.configure(&input, &output, padList);
}

void ClPadWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClPadWorkload_Execute");
    m_Layer.run();
}

arm_compute::Status ClPadValidate(const TensorInfo& input,
                                  const TensorInfo& output,
                                  const PadDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);
    arm_compute::PaddingList padList = static_cast<arm_compute::PaddingList>(descriptor.m_PadList);

    const arm_compute::Status aclStatus = arm_compute::CLPadLayer::validate(&aclInputInfo,
                                                                            &aclOutputInfo,
                                                                            padList);

    return aclStatus;
}

} // namespace armnn
