//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
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
                                   const MeanDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(aclInputInfo.num_dimensions(),
                                                                          input.GetNumDimensions(),
                                                                          descriptor.m_Axis);

    return arm_compute::CLReduceMean::validate(&aclInputInfo, coords, descriptor.m_KeepDims, &aclOutputInfo);
}

ClMeanWorkload::ClMeanWorkload(const MeanQueueDescriptor& descriptor,
                               const WorkloadInfo& info,
                               const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<MeanQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClMeanWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());
    m_Data.ValidateInputsOutputs("ClMeanWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(input.info()->num_dimensions(),
                                                                          info.m_InputTensorInfos[0].GetNumDimensions(),
                                                                          m_Data.m_Parameters.m_Axis);

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClMeanWorkload_configure");
        m_Layer.configure(clCompileContext, &input, coords, m_Data.m_Parameters.m_KeepDims, &output);
    }
}

void ClMeanWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClMeanWorkload_Execute", this->GetGuid());
    m_Layer.run();
}

} //namespace armnn
