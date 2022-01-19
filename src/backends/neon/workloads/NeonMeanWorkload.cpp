//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonMeanWorkload.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <neon/NeonTensorHandle.hpp>

#include "NeonWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status NeonMeanWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const MeanDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo  = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(aclInputInfo.num_dimensions(),
                                                                          input.GetNumDimensions(),
                                                                          descriptor.m_Axis);

    return arm_compute::NEReduceMean::validate(&aclInputInfo, coords, descriptor.m_KeepDims, &aclOutputInfo);
}

NeonMeanWorkload::NeonMeanWorkload(const MeanQueueDescriptor& descriptor, const WorkloadInfo& info)
    : NeonBaseWorkload<MeanQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonMeanWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("NeonMeanWorkload", 1, 1);

    arm_compute::ITensor& input  = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(input.info()->num_dimensions(),
                                                                          info.m_InputTensorInfos[0].GetNumDimensions(),
                                                                          m_Data.m_Parameters.m_Axis);

    m_Layer.configure(&input, coords, m_Data.m_Parameters.m_KeepDims, &output);
}

void NeonMeanWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonMeanWorkload_Execute", this->GetGuid());
    m_Layer.run();
}

} //namespace armnn
