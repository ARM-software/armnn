//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonReduceWorkload.hpp"

#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <neon/NeonTensorHandle.hpp>

#include "NeonWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status NeonReduceWorkloadValidate(const TensorInfo& input,
                                               const TensorInfo& output,
                                               const ReduceDescriptor& desc)
{
    const arm_compute::TensorInfo aclInputInfo  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);
    if (!desc.m_vAxis.empty() && desc.m_vAxis.size() > 1)
    {
        return arm_compute::Status(arm_compute::ErrorCode::RUNTIME_ERROR,
                                   "NeonReduceWorkload: Reduction is supported only on 1 axis.");
    }

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(aclInputInfo.num_dimensions(),
                                                                          input.GetNumDimensions(),
                                                                          desc.m_vAxis);

    return arm_compute::NEReductionOperation::validate(&aclInputInfo,
                                                       &aclOutputInfo,
                                                       static_cast<unsigned int>(coords[0]),
                                                       ConvertReductionOperationToAcl(desc),
                                                       desc.m_KeepDims);
}

NeonReduceWorkload::NeonReduceWorkload(const ReduceQueueDescriptor& descriptor, const WorkloadInfo& info)
    : BaseWorkload<ReduceQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonReduceWorkload", 1, 1);

    arm_compute::ITensor& input  = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(input.info()->num_dimensions(),
                                                                          info.m_InputTensorInfos[0].GetNumDimensions(),
                                                                          m_Data.m_Parameters.m_vAxis);
    m_Layer.configure(&input,
                      &output,
                      static_cast<unsigned int>(coords[0]),
                      ConvertReductionOperationToAcl(m_Data.m_Parameters),
                      m_Data.m_Parameters.m_KeepDims);
}

void NeonReduceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonReduceWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn
