//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClLogicalOrWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <cl/ClTensorHandle.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClLogicalOrWorkloadValidate(const TensorInfo& input0,
                                                const TensorInfo& input1,
                                                const TensorInfo& output)
{
    const arm_compute::TensorInfo aclInputInfo0 = BuildArmComputeTensorInfo(input0);
    const arm_compute::TensorInfo aclInputInfo1 = BuildArmComputeTensorInfo(input1);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output);

    const arm_compute::Status aclStatus = arm_compute::CLLogicalOr::validate(&aclInputInfo0,
                                                                             &aclInputInfo1,
                                                                             &aclOutputInfo);
    return aclStatus;
}

ClLogicalOrWorkload::ClLogicalOrWorkload(const LogicalBinaryQueueDescriptor& descriptor,
                                         const WorkloadInfo& info,
                                         const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<LogicalBinaryQueueDescriptor>(descriptor, info)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClLogicalOrWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClLogicalOrWorkload", 2, 1);

    arm_compute::ICLTensor& input0 = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& input1 = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ICLTensor& output = PolymorphicDowncast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClLogicalOrWorkload_configure");
        m_LogicalOrLayer.configure(clCompileContext, &input0, &input1, &output);
    }
}

void ClLogicalOrWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClLogicalOrWorkload_Execute", this->GetGuid());
    m_LogicalOrLayer.run();
}

} // namespace armnn
