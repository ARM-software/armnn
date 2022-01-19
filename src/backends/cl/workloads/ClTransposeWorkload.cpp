//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClTransposeWorkload.hpp"
#include <cl/ClTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <arm_compute/core/Error.h>

#include "ClWorkloadUtils.hpp"

namespace armnn
{

arm_compute::Status ClTransposeWorkloadValidate(const TensorInfo& input,
                                                const TensorInfo& output,
                                                const TransposeDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);
    const armnn::PermutationVector& mappings = descriptor.m_DimMappings;

    return arm_compute::CLPermute::validate(&aclInputInfo, &aclOutputInfo,
                                            armcomputetensorutils::BuildArmComputeTransposeVector(mappings));
}

ClTransposeWorkload::ClTransposeWorkload(const TransposeQueueDescriptor& descriptor,
                                         const WorkloadInfo& info,
                                         const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<TransposeQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClTransposeWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs(GetName(), 1, 1);

    const arm_compute::ICLTensor& input = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    const armnn::PermutationVector& mappings = m_Data.m_Parameters.m_DimMappings;
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClTransposeWorkload_configure");
        // Run the layer.
        m_PermuteFunction.configure(clCompileContext,
                                    &input,
                                    &output,
                                    armcomputetensorutils::BuildArmComputeTransposeVector(mappings));
    }
}

void ClTransposeWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID(GetName() + "_Execute", this->GetGuid());
    RunClFunction(m_PermuteFunction, CHECK_LOCATION());
}

} // namespace armnn
