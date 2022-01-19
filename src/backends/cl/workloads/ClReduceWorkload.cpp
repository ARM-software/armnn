//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClReduceWorkload.hpp"

#include <cl/ClTensorHandle.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>

#include "ClWorkloadUtils.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClReduceWorkloadValidate(const TensorInfo& input,
                                             const TensorInfo& output,
                                             const ReduceDescriptor& descriptor)
{
    if (descriptor.m_vAxis.size() == 1 || descriptor.m_vAxis.empty())
    {
        const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
        const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

        arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(aclInputInfo.num_dimensions(),
                                                                              input.GetNumDimensions(),
                                                                              descriptor.m_vAxis);

        return arm_compute::CLReductionOperation::validate(&aclInputInfo,
                                                           &aclOutputInfo,
                                                           static_cast<unsigned int>(coords[0]),
                                                           ConvertReductionOperationToAcl(descriptor),
                                                           descriptor.m_KeepDims);
    }
    else
    {
        // Validate layer if there are multiple axes.
        arm_compute::Status status;
        IS_MULTI_AXES_REDUCE_SUPPORTED(ClReduceWorkloadValidate, input, descriptor, status);
        return status;
    }
}

ClReduceWorkload::ClReduceWorkload(const ReduceQueueDescriptor& descriptor, const WorkloadInfo& info)
    : ClBaseWorkload<ReduceQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClReduceWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClReduceWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::Coordinates coords = BuildArmComputeReductionCoordinates(input.info()->num_dimensions(),
                                                                          info.m_InputTensorInfos[0].GetNumDimensions(),
                                                                          m_Data.m_Parameters.m_vAxis);
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClReduceWorkload_configure");
        m_Layer.configure(&input,
                          &output,
                          static_cast<unsigned int>(coords[0]),
                          ConvertReductionOperationToAcl(m_Data.m_Parameters),
                          m_Data.m_Parameters.m_KeepDims);
    }
}

void ClReduceWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClReduceWorkload_Execute", this->GetGuid());
    m_Layer.run();
}

} //namespace armnn
