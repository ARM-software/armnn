//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonTransposeWorkload.hpp"
#include <neon/NeonTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/core/Error.h>

namespace armnn
{

arm_compute::Status NeonTransposeWorkloadValidate(const TensorInfo& input,
                                                  const TensorInfo& output,
                                                  const TransposeDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);
    const armnn::PermutationVector& mappings = descriptor.m_DimMappings;

    return arm_compute::NEPermute::validate(&aclInputInfo, &aclOutputInfo,
                                            armcomputetensorutils::BuildArmComputeTransposeVector(mappings));
}

NeonTransposeWorkload::NeonTransposeWorkload(const TransposeQueueDescriptor& descriptor,
                                             const WorkloadInfo& info)
        : NeonBaseWorkload<TransposeQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonTransposeWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs(GetName(), 1, 1);

    const arm_compute::ITensor& input = static_cast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = static_cast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    const armnn::PermutationVector& mappings = m_Data.m_Parameters.m_DimMappings;

    // Run the layer.
    m_PermuteFunction.configure(&input, &output,
                                armcomputetensorutils::BuildArmComputeTransposeVector(mappings));
}

void NeonTransposeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID(GetName() + "_Execute", this->GetGuid());
    m_PermuteFunction.run();
}

} // namespace armnn
