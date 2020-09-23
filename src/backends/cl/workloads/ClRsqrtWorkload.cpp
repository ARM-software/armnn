//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClRsqrtWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <cl/ClTensorHandle.hpp>

namespace armnn
{

arm_compute::Status ClRsqrtWorkloadValidate(const TensorInfo& input, const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInput  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutput = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    return arm_compute::CLRsqrtLayer::validate(&aclInput, &aclOutput);
}

ClRsqrtWorkload::ClRsqrtWorkload(const RsqrtQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<RsqrtQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClRsqrtWorkload", 1, 1);

    arm_compute::ICLTensor& input  = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    m_RsqrtLayer.configure(&input, &output);
}

void ClRsqrtWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClRsqrtWorkload_Execute");
    RunClFunction(m_RsqrtLayer, CHECK_LOCATION());
}

} // namespace armnn
