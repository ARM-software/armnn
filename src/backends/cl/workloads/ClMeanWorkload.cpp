//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClMeanWorkload.hpp"

#include <cl/ClTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClMeanValidate(const TensorInfo& input,
                                   const TensorInfo& output,
                                   const MeanDescriptor& desc)
{
    const arm_compute::TensorInfo aclInputInfo  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(aclInputInfo.num_dimensions(),
                                                                          input.GetNumDimensions(),
                                                                          desc.m_Axis);

    return arm_compute::CLReduceMean::validate(&aclInputInfo, coords, desc.m_KeepDims, &aclOutputInfo);
}

ClMeanWorkload::ClMeanWorkload(const MeanQueueDescriptor& descriptor,
                               const WorkloadInfo& info,
                               const arm_compute::CLCompileContext& clCompileContext)
    : BaseWorkload<MeanQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("ClMeanWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(input.info()->num_dimensions(),
                                                                          info.m_InputTensorInfos[0].GetNumDimensions(),
                                                                          m_Data.m_Parameters.m_Axis);

    m_Layer.configure(clCompileContext, &input, coords, m_Data.m_Parameters.m_KeepDims, &output);
}

void ClMeanWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClMeanWorkload_Execute");
    m_Layer.run();
}

} //namespace armnn
