//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClSoftmaxWorkload.hpp"
#include "ClWorkloadUtils.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <cl/ClTensorHandle.hpp>

namespace armnn
{

arm_compute::Status ClSoftmaxWorkloadValidate(const TensorInfo& input,
                                              const TensorInfo& output,
                                              const SoftmaxDescriptor& descriptor)
{
    const arm_compute::TensorInfo aclInputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(input);
    const arm_compute::TensorInfo aclOutputInfo = armcomputetensorutils::BuildArmComputeTensorInfo(output);

    int aclAxis = ComputeAclAxis(descriptor.m_Axis, input);
    return arm_compute::CLSoftmaxLayer::validate(&aclInputInfo, &aclOutputInfo, descriptor.m_Beta, aclAxis);
}

ClSoftmaxWorkload::ClSoftmaxWorkload(const SoftmaxQueueDescriptor& descriptor,
                                     const WorkloadInfo& info,
                                     std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                                     const arm_compute::CLCompileContext& clCompileContext)
        : ClBaseWorkload<SoftmaxQueueDescriptor>(descriptor, info)
        , m_SoftmaxLayer(memoryManager)
{
    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClSoftmaxWorkload_Construct",
                                         descriptor.m_Parameters,
                                         info,
                                         this->GetGuid());

    m_Data.ValidateInputsOutputs("ClSoftmaxWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<ClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<ClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    int aclAxis = ComputeAclAxis(m_Data.m_Parameters.m_Axis, info.m_InputTensorInfos[0]);
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClSoftmaxWorkload_configure");
        m_SoftmaxLayer.configure(clCompileContext, &input, &output, m_Data.m_Parameters.m_Beta, aclAxis);
    }
}

void ClSoftmaxWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClSoftmaxWorkload_Execute", this->GetGuid());
    RunClFunction(m_SoftmaxLayer, CHECK_LOCATION());
}

} // namespace armnn
